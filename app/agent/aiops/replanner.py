"""
Replanner 节点：重新规划或生成最终响应
基于 LangGraph 官方教程实现
"""

from textwrap import dedent
from typing import Dict, Any, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_deepseek import ChatDeepSeek
from pydantic import BaseModel, Field
from loguru import logger

from app.config import config
from app.tools import get_current_time, retrieve_knowledge, execute_shell_command
from app.agent.mcp_client import get_mcp_client_with_retry
from .state import PlanExecuteState
from .utils import format_tools_description


def _build_skill_context_block(state: PlanExecuteState) -> str:
    """从 state 提取共享 skill 上下文, 拼成给 LLM 看的提示块

    若 planner 没匹配到 skill, 返回空字符串
    """
    matched_skill = state.get("matched_skill")
    if not matched_skill:
        return ""

    skill_display_name = state.get("skill_display_name") or matched_skill
    skill_risk_level = state.get("skill_risk_level") or "unknown"
    skill_allowed_tools = state.get("skill_allowed_tools") or []
    skill_playbook = state.get("skill_playbook") or "(此 Skill 未提供 Playbook)"

    return dedent(f"""

        ## 当前匹配的 Skill 上下文 (与 Planner / Executor 共享)
        - **Skill**: {skill_display_name} ({matched_skill})
        - **风险等级**: {skill_risk_level}
        - **推荐工具**: {', '.join(skill_allowed_tools[:10]) if skill_allowed_tools else '无限制'}

        ## Skill Playbook (重新规划时的参考)
        {skill_playbook}

        ## Skill 重规划约束
        - 新步骤应继续遵循 Playbook 的 Phase 顺序
        - 不要引入 Playbook 之外、风险更高的写操作
        - 若已收集到 Playbook 各 Phase 的关键信息, 优先 respond 而不是 replan
    """).strip()

class Response(BaseModel):
    """最终响应的格式"""
    response: str = Field(description="对用户的最终响应")

class Act(BaseModel):
    """重新规划的输出格式"""
    action: str = Field(
        description="""下一步的行动，必须是以下三种之一：
        - 'continue': 当前计划合理，继续执行下一个步骤
        - 'replan': 当前计划需要调整，提供新的步骤列表
        - 'respond': 计划已完成且信息充足，生成最终响应(如果)"""
    )
    # action 为 'replan' 时，新的步骤列表（会替换当前剩余计划）
    new_steps: List[str] = Field(
        default_factory=list,
        description="新的步骤列表（如果 action 是 'replan'，这些步骤会替换剩余计划）"
    )

# Replanner 提示词
replanner_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            dedent("""
                作为一个重新规划专家，你需要根据已执行的步骤决定下一步行动。

                可用工具列表（用于制定计划时参考）：

                {tools_description}

                {skill_context}

                注意：你的职责是制定或调整计划，实际的工具调用由 Executor 负责执行。

                ⚠️ **强制要求（必须遵守）**：
                - **至少执行 3 个步骤后才能考虑 'respond'**
                - 如果已执行步骤 < 3，只能选择 'continue' 或 'replan'，不能选择 'respond'
                - 这是系统的硬性约束，即使你认为信息已经充足也必须遵守

                你有三个选择（按优先级排序）：

                **1. 'respond' - 信息充足，立即生成最终响应** 【最高优先级】
                   - 使用场景：当前信息已经足够回答用户问题
                   - **强制条件**：已执行步骤 >= 3 【必须满足】
                   - 决策标准：
                     * 已执行步骤 >= 3 且获取了关键信息
                     * 或者已执行步骤 >= 5（无论结果如何）
                     * 或者当前信息完全满足任务需求
                   - ⚠️ 不要等到"完美"才响应，"足够好"就应该立即 respond

                **2. 'continue' - 当前计划合理，继续执行** 【次优先级】
                   - 使用场景：剩余计划合理且必要
                   - 决策标准：剩余步骤确实能提供关键信息
                   - ⚠️ 如果剩余步骤不是"必需"的，应选择 respond

                **3. 'replan' - 当前计划有严重问题** 【最低优先级，谨慎使用】
                   - 使用场景：原计划明显错误或遗漏关键步骤
                   - ⚠️ **严格限制**：
                     * 新步骤数量必须 <= 当前剩余步骤数
                     * 优先简化计划，不要添加不必要的步骤
                     * 总步骤数已执行 >= 5 次时，禁止 replan，只能 respond

                评估标准：
                - 当前信息是否已经足够解决用户问题？【最关键】
                - 已执行步骤是否成功获取了核心信息？
                - 剩余步骤是否真的"必需"？
                - 已执行步骤数是否过多（>= 5）？如果是，立即 respond

                **决策优先级口诀：** 
                "优先结束 > 保持不变 > 调整计划"
                "信息足够就响应，不要追求完美"
                "至少三步才能结束，这是铁律"
            """).strip(),
        ),
        ("placeholder", "{messages}"),
    ]
)

# 最终响应生成提示词
response_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            dedent("""
                根据原始任务和已执行步骤的结果，生成一个全面的最终响应。

                响应要求：
                - 清晰、结构化
                - 基于实际数据，不要编造
                - 如果某些步骤失败，要诚实说明
                - 使用 Markdown 格式
            """).strip(),
        ),
        ("placeholder", "{messages}"),
    ]
)

async def replanner(state: PlanExecuteState) -> Dict[str, Any]:
    """
    重新规划节点：决定是继续、调整计划还是生成最终响应

    三种决策：
    1. continue - 继续执行当前计划
    2. replan - 调整计划（替换剩余步骤）
    3. respond - 生成最终响应

    ⚠️ 强制规则：必须至少执行 3 个步骤才能输出 respond
    """
    logger.info("=== Replanner：重新规划 ===")

    input_text = state.get("input", "")
    plan = state.get("plan", [])
    past_steps = state.get("past_steps", [])
    executed_count = len(past_steps)

    logger.info(f"剩余计划步骤: {len(plan)}")
    logger.info(f"已执行步骤: {executed_count}")

    # 读取由 planner 注入的共享 skill 状态
    matched_skill = state.get("matched_skill")
    if matched_skill:
        logger.info(f"📌 使用共享 Skill 状态: {matched_skill}")
    skill_context = _build_skill_context_block(state)

    # ⚠️ 强制限制 1：如果已执行步骤 < 3，禁止 respond
    MIN_EXECUTED_STEPS = 3
    if executed_count < MIN_EXECUTED_STEPS:
        logger.info(f"⚠️ 强制要求：已执行 {executed_count} 个步骤 < {MIN_EXECUTED_STEPS}，禁止 respond，必须继续执行")

    # ⚠️ 强制限制 2：如果已执行步骤过多，直接生成响应
    MAX_STEP = 8
    if executed_count > MAX_STEP:
        logger.warning(f"已执行 {executed_count} 个步骤，超过最大限制 {MAX_STEP}，强制生成最终响应")
        llm = ChatDeepSeek(
            model=config.rag_model,
            api_key=config.DEEPSEEK_API_KEY,
            temperature=0
        )
        return await _generate_response(state, llm)

    # 获取可用的工具列表
    try:
        # 获取本地工具
        local_tools = [
            get_current_time,
            retrieve_knowledge,
            execute_shell_command
        ]

        # 获取 MCP 工具
        mcp_client = await get_mcp_client_with_retry()
        mcp_tools = await mcp_client.get_tools()

        # 合并所有工具
        all_tools = local_tools + mcp_tools
        logger.info(f"可用工具数量: 本地 {len(local_tools)} + MCP {len(mcp_tools)}")

        # 格式化工具描述
        tools_description = format_tools_description(all_tools)
    except Exception as e:
        logger.warning(f"获取工具列表失败: {e}")
        tools_description = "无法获取工具列表"

    # 创建 LLM
    llm = ChatDeepSeek(
        model=config.rag_model,
        api_key=config.DEEPSEEK_API_KEY,
        temperature=0
    )

    # 格式化已执行的步骤
    steps_summary = "\n".join([
        f"步骤: {step}\n结果: {result[:300]}..."
        for step, result in past_steps
    ])

    # 如果还有剩余计划，进行决策
    if plan:
        logger.info("还有剩余计划，评估下一步行动")

        replanner_chain = replanner_prompt | llm.with_structured_output(Act)

        # 添加强制要求的提示信息
        force_continue_hint = ""
        if executed_count < MIN_EXECUTED_STEPS:
            force_continue_hint = f"\n\n⚠️ **强制要求**：当前已执行 {executed_count} 个步骤，必须至少执行 {MIN_EXECUTED_STEPS} 个步骤才能生成响应。你只能选择 'continue' 或 'replan'，绝对不能选择 'respond'！"

        try:
            messages = [
                ("user", f"原始任务: {input_text}"),
                ("user", f"已执行的步骤:\n{steps_summary}"),
                ("user", f"剩余计划: {', '.join(plan)}"),
                ("user", f"⚠️ 重要提示：已执行 {executed_count} 个步骤，请优先考虑是否信息已足够生成响应（respond）"),
                ("user", f"⚠️ 强制要求：至少执行 {MIN_EXECUTED_STEPS} 个步骤才能 respond，当前为 {executed_count} 个步骤{force_continue_hint}")
            ]

            act = await replanner_chain.ainvoke({
                "messages": messages,
                "tools_description": tools_description,
                "skill_context": skill_context,
            })

            # 处理返回结果
            if isinstance(act, Act):
                action = act.action
                new_steps = act.new_steps
            else:
                # 如果返回的是字典
                action = act.get("action", "continue")  # type: ignore
                new_steps = act.get("new_steps", [])  # type: ignore

            logger.info(f"Replanner 决策: {action}")

            # ⚠️ 强制检查：如果已执行步骤 < 3 且决策是 respond，强制改为 continue
            if action == "respond" and executed_count < MIN_EXECUTED_STEPS:
                logger.warning(
                    f"⚠️ 违反强制要求：已执行 {executed_count} 个步骤 < {MIN_EXECUTED_STEPS}，"
                    f"决策 '{action}' 被强制改为 'continue'"
                )
                action = "continue"
                logger.info("强制继续执行")

            if action == "respond":
                # 二次检查：确保已执行步骤 >= 3
                if executed_count >= MIN_EXECUTED_STEPS:
                    logger.info(f"决定生成最终响应（已执行 {executed_count} 个步骤，符合要求）")
                    return await _generate_response(state, llm)
                else:
                    # 不应该走到这里，因为上面已经强制改过，但为了安全再检查一次
                    logger.warning(f"⚠️ 异常：仍尝试 respond，但已执行步骤={executed_count}，强制改为 continue")
                    return {}
            elif action == "replan":
                # ⚠️ 强制限制：新步骤数不能超过当前剩余步骤数
                if len(new_steps) > len(plan):
                    logger.warning(
                        f"新步骤数 {len(new_steps)} > 剩余步骤数 {len(plan)}，"
                        f"强制截断为 {len(plan)} 个步骤"
                    )
                    new_steps = new_steps[:len(plan)]
                # ⚠️ 二次检查：如果已执行步骤 >= 5，禁止 replan
                if executed_count >= 5:
                    logger.warning(f"已执行 {executed_count} 个步骤，禁止重新规划，强制继续执行")
                    return {}

                logger.info(f"决定调整计划，新步骤数量: {len(new_steps)}")
                if new_steps:
                    # 替换剩余计划
                    return {"plan": new_steps}
                else:
                    logger.warning("replan 但未提供新步骤，继续执行原计划")
                    return {}

            else:  # action == "continue"
                logger.info("决定继续执行当前计划")
                return {}  # 不修改状态，继续执行

        except Exception as e:
            logger.error(f"重新规划失败: {e}, 继续执行剩余计划")
            return {}

    else:
        # 没有剩余计划
        # ⚠️ 强制检查：即使没有剩余计划，也必须执行至少 3 个步骤才能响应
        if executed_count >= MIN_EXECUTED_STEPS:
            logger.info(f"计划已执行完毕，已执行 {executed_count} 个步骤，生成最终响应")
            return await _generate_response(state, llm)
        else:
            # 执行步骤不足，不能结束，需要生成新计划
            logger.warning(
                f"计划已执行完毕，但只执行了 {executed_count} 个步骤 < {MIN_EXECUTED_STEPS}，"
                f"需要继续规划新步骤"
            )
            # 返回空字典让 graph 继续，但因为没有计划，需要 planner 生成新计划
            # 这里可以触发一个 replan 信号
            # 生成一个简单的继续计划
            fallback_plan = [f"继续收集相关信息来回答: {input_text}"]
            logger.info(f"生成后备计划: {fallback_plan}")
            return {"plan": fallback_plan}


async def _generate_response(state: PlanExecuteState, llm: ChatDeepSeek) -> Dict[str, Any]:
    """生成最终响应 - 会带上共享 skill 上下文, 让响应风格与 Skill Playbook 对齐"""
    logger.info("生成最终响应...")

    input_text = state.get("input", "")
    past_steps = state.get("past_steps", [])
    executed_count = len(past_steps)
    matched_skill = state.get("matched_skill")
    skill_display_name = state.get("skill_display_name") or matched_skill
    skill_risk_level = state.get("skill_risk_level")

    # 安全检查：确保至少执行了 3 个步骤
    MIN_EXECUTED_STEPS = 3
    if executed_count < MIN_EXECUTED_STEPS:
        logger.error(f"❌ 严重错误：尝试生成响应但只执行了 {executed_count} 个步骤 < {MIN_EXECUTED_STEPS}")
        return {
            "response": f"⚠️ 系统提示：当前已执行 {executed_count} 个步骤，但系统要求至少执行 {MIN_EXECUTED_STEPS} 个步骤才能生成完整响应。请继续执行更多任务。"
        }

    # 格式化执行历史
    execution_history = "\n\n".join([
        f"### 步骤: {step}\n**结果:**\n{result}"
        for step, result in past_steps
    ])

    response_gen = response_prompt | llm.with_structured_output(Response)

    try:
        messages = [
            ("user", f"原始任务: {input_text}"),
            ("user", f"执行历史:\n{execution_history}"),
        ]
        # 若有共享 skill 状态, 一并喂给最终响应生成
        if matched_skill:
            messages.append((
                "user",
                f"本次诊断使用的 Skill: {skill_display_name} ({matched_skill}), "
                f"风险等级: {skill_risk_level or 'unknown'}. "
                "请在结论中体现该 Skill 的 Phase 结构与处置建议."
            ))
        messages.append(("user", "请基于以上信息生成全面的最终响应"))

        response_obj = await response_gen.ainvoke({"messages": messages})

        # 处理返回结果
        if isinstance(response_obj, Response):
            final_result = response_obj.response
        else:
            final_result = response_obj.get("response")

        logger.info(f"最终响应生成完成，长度: {len(final_result)}")

        return {"response": final_result}

    except Exception as e:
        logger.error(f"生成响应失败: {e}")
        # 生成简单的后备响应
        fallback_response = f"""# 任务执行结果
        
            ## 原始任务
            {input_text}
        
            ## 执行的步骤（共 {executed_count} 个）
            {_format_simple_steps(past_steps)}
        
            ## 说明
            由于系统异常，无法生成完整响应。以上是已收集的信息。
            """
        return {"response": fallback_response}

def _format_simple_steps(past_steps: list) -> str:
    """格式化步骤列表（简单版）"""
    if not past_steps:
        return "无"

    formatted = []
    for i, (step, result) in enumerate(past_steps, 1):
        result_preview = result[:200] + "..." if len(result) > 200 else result
        formatted.append(f"{i}. **{step}**\n   {result_preview}\n")

    return "\n".join(formatted)
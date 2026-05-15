"""
CLI Replanner 节点：重新规划或生成最终响应
专门用于命令行任务的重新规划和结果总结
"""

from textwrap import dedent
from typing import Dict, Any, List, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_deepseek import ChatDeepSeek
from pydantic import BaseModel, Field
from loguru import logger

from app.config import config
from app.tools.cli_tools import execute_shell_command
from .state import PlanExecuteState
from .utils import format_tools_description


class CLIAct(BaseModel):
    """CLI 重新规划的输出格式"""
    action: str = Field(
        description="""下一步的行动，必须是以下三种之一：
        - 'continue': 命令执行成功，继续执行下一个命令
        - 'replan': 当前命令执行失败或需要调整，提供新的命令
        - 'respond': 所有命令已执行完成，生成最终响应"""
    )
    # action 为 'replan' 时，新的命令列表（会替换当前剩余命令）
    new_commands: List[str] = Field(
        default_factory=list,
        description="新的命令列表（如果 action 是 'replan'，这些命令会替换剩余命令）"
    )
    analysis: str = Field(
        default="",
        description="对当前执行结果的分析，说明为什么要继续、重新规划或结束"
    )


class CLIResponse(BaseModel):
    """CLI 最终响应的格式"""
    response: str = Field(description="对用户的最终响应，基于命令执行结果用自然语言回答")
    summary: str = Field(description="执行摘要，简要说明执行了哪些命令")
    suggestions: Optional[str] = Field(default=None, description="如果发现问题，提供建议")


def _build_cli_skill_context_block(state: PlanExecuteState) -> str:
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
        - 新命令应继续遵循 Playbook 的 Phase 顺序
        - 不要引入 Playbook 之外、风险更高的写操作
        - 若已收集到 Playbook 各 Phase 的关键信息, 优先 respond 而不是 replan
    """).strip()


# CLI Replanner 提示词
cli_replanner_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            dedent("""
                你是一个专业的 Windows 命令行任务评估专家。你需要根据已执行的命令结果，决定下一步行动。

                ## 可用工具
                {tools_description}

                {skill_context}

                ## 决策规则

                ### 1. 'respond' - 生成最终响应 【最高优先级】
                适用场景：
                - 所有计划中的命令都已执行完成
                - 已执行命令数量 >= 5（避免无限循环）
                - 当前信息已经足够回答用户问题
                - 连续多次执行失败

                ### 2. 'continue' - 继续执行下一个命令 【次优先级】
                适用场景：
                - 当前命令执行成功
                - 还有剩余命令需要执行
                - 剩余命令确实能提供有用信息

                ### 3. 'replan' - 调整命令计划 【谨慎使用】
                适用场景：
                - 当前命令执行失败，需要尝试替代方案
                - 当前命令结果不理想，需要更精确的命令
                - 发现新的信息需要进一步探索

                限制条件：
                - 新命令数量 <= 当前剩余命令数
                - 已执行命令 < 5 次时才能 replan
                - 每个问题最多 replan 2 次

                ## 分析要点
                根据命令输出判断：
                - 端口检查：端口是否被占用？哪个进程占用？
                - 进程检查：进程是否运行？运行状态如何？
                - 系统资源：资源使用率是否正常？有无异常？
                - 文件操作：文件是否存在？内容是否符合预期？
                - 网络诊断：网络是否连通？延迟是否正常？

                ## 输出要求
                必须输出包含以下字段的 JSON：
                - action: 决策动作
                - new_commands: 新命令列表（仅在 replan 时需要）
                - analysis: 对当前结果的分析说明

                ## 示例

                用户问题："查看8004端口是否正在使用"
                已执行命令：Get-NetTCPConnection -LocalPort 8004
                输出结果：空（没有输出）
                分析：端口未被占用
                决策：respond - 直接告诉用户端口未被占用

                用户问题："检查系统CPU使用情况"
                已执行命令：Get-Counter "\\Processor(_Total)\\% Processor Time"
                输出结果：显示CPU使用率85%
                分析：CPU使用率较高，需要进一步排查
                剩余命令：Get-CimInstance Win32_OperatingSystem, Get-PSDrive
                决策：continue - 继续执行内存和磁盘检查

                用户问题："查看nginx进程"
                已执行命令：Get-Process nginx
                输出结果：找不到nginx进程
                分析：nginx未运行，但之前计划中只有检查命令
                决策：replan - 添加启动nginx的命令
                新命令：Start-Service nginx

                **重要提示：优先使用 respond 尽快给用户反馈，不要过度执行命令**
            """).strip(),
        ),
        ("placeholder", "{messages}"),
    ]
)

# 最终响应生成提示词
cli_response_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            dedent("""
                你是一个专业的 Windows 运维助手。请根据用户的问题和命令执行结果，生成一个清晰、有用的最终响应。

                ## 响应要求
                1. **回答要直接**：直接回答用户的问题，不要绕弯子
                2. **突出重点**：如果发现问题，要明确指出
                3. **提供建议**：如果存在问题，给出解决方案
                4. **格式美观**：使用 markdown 格式，让信息更易读
                5. **数据准确**：基于命令输出，不要编造数据

                ## 响应结构
                ### 执行摘要
                简要说明执行了哪些命令

                ### 关键发现
                列出最重要的发现（端口状态、进程情况、资源使用等）

                ### 详细结果
                逐条说明每个命令的输出

                ### 建议（如果需要）
                如果发现问题，给出解决建议

                ## 示例

                用户问题："查看8004端口是否正在使用"
                命令输出：Get-NetTCPConnection -LocalPort 8004 返回空
                响应：

                ## 执行摘要
                已执行端口检查命令

                ## 关键发现
                ✅ 8004端口当前未被使用

                ## 详细结果
                命令: Get-NetTCPConnection -LocalPort 8004

                结果: 没有进程监听8004端口

                ## 建议
                该端口可用，您可以放心使用。


                ## 注意事项
                - 如果命令执行失败，诚实说明并提供可能的原因
                - 输出内容过长时，适当截断但保留关键信息
                - 使用友好的语气，让用户容易理解
            """).strip(),
        ),
        ("placeholder", "{messages}"),
    ]
)

async def cli_replanner(state: PlanExecuteState) -> Dict[str, Any]:
    """
    CLI 重新规划节点：决定是继续、调整命令还是生成最终响应

    三种决策：
    1. continue - 继续执行下一个命令
    2. replan - 调整命令计划（替换剩余命令）
    3. respond - 生成最终响应
    """
    logger.info("=== CLI Replanner：重新规划 ===")

    input_text = state.get("input", "")
    plan = state.get("plan", [])
    past_steps = state.get("past_steps", [])
    replan_count = state.get("replan_count", 0)  # 记录重新规划次数

    logger.info(f"剩余命令数: {len(plan)}")
    logger.info(f"已执行命令数: {len(past_steps)}")
    logger.info(f"重新规划次数: {replan_count}")

    # 读取由 planner 注入的共享 skill 状态
    matched_skill = state.get("matched_skill")
    if matched_skill:
        logger.info(f"📌 使用共享 Skill 状态: {matched_skill}")
    skill_context_block = _build_cli_skill_context_block(state)

    # ========== 强制结束条件 ==========

    # 条件1：已执行命令过多，强制生成响应
    MAX_STEPS = 10
    if len(past_steps) >= MAX_STEPS:
        logger.warning(f"已执行 {len(past_steps)} 个命令，超过最大限制 {MAX_STEPS}，强制生成最终响应")
        llm = ChatDeepSeek(
            model=config.rag_model,
            api_key=config.DEEPSEEK_API_KEY,
            temperature=0
        )
        return await _generate_cli_response(state, llm)

    # 条件2：重新规划次数过多，强制生成响应
    MAX_REPLAN = 3
    if replan_count >= MAX_REPLAN:
        logger.warning(f"已经重新规划 {replan_count} 次，超过最大限制 {MAX_REPLAN}，强制生成响应")
        llm = ChatDeepSeek(
            model=config.rag_model,
            api_key=config.DEEPSEEK_API_KEY,
            temperature=0
        )
        return await _generate_cli_response(state, llm)

    # 条件3：没有剩余命令，生成响应
    if not plan:
        logger.info("所有命令已执行完毕，生成最终响应")
        llm = ChatDeepSeek(
            model=config.rag_model,
            api_key=config.DEEPSEEK_API_KEY,
            temperature=0
        )
        return await _generate_cli_response(state, llm)

    # ========== 决策阶段 ==========

    # 获取最后一个执行的命令结果
    last_step = past_steps[-1] if past_steps else None
    last_command = last_step[0] if last_step else ""
    last_result = last_step[1] if last_step else ""

    logger.info(f"最后一个命令: {last_command[:100]}")
    logger.info(f"命令结果长度: {len(last_result)} 字符")

    # 创建 LLM
    llm = ChatDeepSeek(
        model=config.rag_model,
        api_key=config.DEEPSEEK_API_KEY,
        temperature=0
    )

    # 格式化工具描述
    tools_description = format_tools_description([execute_shell_command])

    # 格式化已执行的命令历史
    steps_summary = []
    for i, (cmd, result) in enumerate(past_steps, 1):
        # 截断过长的结果
        result_preview = result[:300] + "..." if len(result) > 300 else result
        steps_summary.append(f"命令{i}: {cmd}\n结果: {result_preview}")

    # 获取待执行的命令预览
    remaining_commands = plan[:3]  # 只显示前3个
    remaining_preview = "\n".join([f"  - {cmd[:80]}" for cmd in remaining_commands])
    if len(plan) > 3:
        remaining_preview += f"\n  ... 还有 {len(plan) - 3} 个命令"

    # 构建决策消息
    decision_messages = [
        ("user", f"原始任务: {input_text}"),
        ("user", f"已执行的命令:\n{chr(10).join(steps_summary)}"),
        ("user", f"剩余命令:\n{remaining_preview}"),
        ("user", f"重新规划次数: {replan_count}"),
    ]

    # 如果最后一个命令执行失败，添加特别提示
    if "失败" in last_result or "错误" in last_result or "error" in last_result.lower():
        decision_messages.append(("user", "⚠️ 最后一个命令执行失败，请考虑是否需要重新规划（replan）或直接生成响应（respond）"))

    try:
        # 调用 LLM 进行决策
        replanner_chain = cli_replanner_prompt | llm.with_structured_output(CLIAct)

        act = await replanner_chain.ainvoke({
            "messages": decision_messages,
            "tools_description": tools_description,
            "skill_context": skill_context_block,
        })

        # 处理返回结果
        if isinstance(act, CLIAct):
            action = act.action
            new_commands = act.new_commands
            analysis = act.analysis
        else:
            action = act.get("action", "continue")  # type: ignore
            new_commands = act.get("new_commands", [])  # type: ignore
            analysis = act.get("analysis", "")  # type: ignore

        logger.info(f"Replanner 决策: {action}")
        logger.info(f"决策分析: {analysis}")

        # ========== 处理不同决策 ==========

        if action == "respond":
            logger.info("信息充足，生成最终响应")
            return await _generate_cli_response(state, llm)


        elif action == "replan":

            # 检查重新规划次数限制

            if replan_count >= MAX_REPLAN:
                logger.warning(f"已达到最大重新规划次数 {MAX_REPLAN}，强制生成响应")

                return await _generate_cli_response(state, llm)

            if not new_commands:
                logger.warning("replan 但未提供新命令，继续执行原计划")

                return {}

            logger.info(f"重新规划，新命令数: {len(new_commands)}")

            for i, cmd in enumerate(new_commands[:5], 1):  # 只打印前5个

                logger.info(f"  新命令{i}: {cmd[:100]}...")

            if len(new_commands) > 5:
                logger.info(f"  ... 还有 {len(new_commands) - 5} 个命令")

            # 直接替换计划，不进行数量截断

            return {
                "plan": new_commands,
                "replan_count": replan_count + 1
            }

        else:  # action == "continue"
            logger.info("继续执行下一个命令")
            return {}  # 不修改状态，继续执行

    except Exception as e:
        logger.error(f"重新规划失败: {e}", exc_info=True)

        # 发生错误时，如果还有剩余命令，继续执行
        if plan:
            logger.info("规划失败，继续执行剩余命令")
            return {}
        else:
            # 没有剩余命令，生成简单响应
            return await _generate_cli_response(state, llm)


async def _generate_cli_response(state: PlanExecuteState, llm: ChatDeepSeek) -> Dict[str, Any]:
    """
    生成最终响应（基于命令执行结果）
    """
    logger.info("生成 CLI 最终响应...")

    input_text = state.get("input", "")
    past_steps = state.get("past_steps", [])
    plan = state.get("plan", [])

    # 如果没有执行任何命令
    if not past_steps:
        return {
            "response": f"未能执行任何命令来回答您的问题：{input_text}\n\n可能原因：\n- 没有生成合适的命令\n- 用户取消了所有命令执行",
            "response_type": "error"
        }

    # 格式化执行历史
    execution_details = []
    success_count = 0
    fail_count = 0

    for i, (command, result) in enumerate(past_steps, 1):
        # 判断是否成功
        is_success = "失败" not in result and "错误" not in result and "error" not in result.lower()
        if is_success:
            success_count += 1
        else:
            fail_count += 1

        # 截断过长的结果
        result_preview = result[:1000] if len(result) > 1000 else result
        execution_details.append(f"### 命令 {i}\n**执行:** `{command}`\n**结果:**\n```\n{result_preview}\n```")

    # 检查是否有剩余未执行的命令
    remaining_info = ""
    if plan:
        remaining_info = f"\n\n**注意：** 还有 {len(plan)} 个命令未执行（用户可能已跳过或退出）"

    response_chain = cli_response_prompt | llm.with_structured_output(CLIResponse)

    try:
        messages = [
            ("user", f"用户问题: {input_text}"),
            ("user", f"执行了 {len(past_steps)} 个命令（成功 {success_count} 个，失败 {fail_count} 个）"),
            ("user", f"执行详情:\n{chr(10).join(execution_details)}{remaining_info}"),
        ]

        # 若有共享 skill 状态, 喂给最终响应生成
        matched_skill = state.get("matched_skill")
        if matched_skill:
            skill_display_name = state.get("skill_display_name") or matched_skill
            skill_risk_level = state.get("skill_risk_level") or "unknown"
            messages.append((
                "user",
                f"本次诊断使用的 Skill: {skill_display_name} ({matched_skill}), "
                f"风险等级: {skill_risk_level}. "
                "请在结论中体现该 Skill 的 Phase 结构与处置建议."
            ))

        messages.append(("user", "请根据以上信息生成最终响应"))

        response_obj = await response_chain.ainvoke({"messages": messages})

        # 处理返回结果
        if isinstance(response_obj, CLIResponse):
            final_response = response_obj.response
            summary = response_obj.summary
            suggestions = response_obj.suggestions
        else:
            final_response = response_obj.get("response", "")
            summary = response_obj.get("summary", "")
            suggestions = response_obj.get("suggestions", "")

        logger.info(f"最终响应生成完成，长度: {len(final_response)}")

        # 如果有建议，附加到响应中
        if suggestions:
            final_response += f"\n\n## 💡 建议\n{suggestions}"

        return {
            "response": final_response,
            "summary": summary,
            "execution_summary": {
                "total_commands": len(past_steps),
                "success_count": success_count,
                "fail_count": fail_count,
                "remaining_count": len(plan)
            }
        }

    except Exception as e:
        logger.error(f"生成响应失败: {e}")

        # 生成简单的后备响应
        fallback_response = _generate_fallback_response(input_text, past_steps, plan)
        return {"response": fallback_response}


def _generate_fallback_response(input_text: str, past_steps: List, plan: List) -> str:
    """生成后备响应（当 LLM 生成失败时）"""

    # 分析最后一条命令的结果
    if past_steps:
        last_command, last_result = past_steps[-1]

        # 简单的关键词匹配
        if "端口" in input_text or "port" in input_text.lower():
            if "no output" in last_result.lower() or not last_result.strip():
                return f"检查结果显示：端口未被使用。"
            elif "LISTEN" in last_result:
                return f"检查结果显示：端口正在使用中。\n\n详细信息：\n{last_result[:500]}"

        elif "进程" in input_text or "process" in input_text.lower():
            if "grep" in last_command:
                lines = [l for l in last_result.split('\n') if l and 'grep -v' not in l]
                if lines:
                    return f"找到 {len(lines)} 个相关进程：\n{last_result[:500]}"
                else:
                    return f"未找到相关进程。"

        elif "cpu" in input_text.lower() or "内存" in input_text:
            return f"系统资源检查结果：\n{last_result[:500]}"

    # 通用响应
    executed_count = len(past_steps)
    return f"""## 任务执行结果

**原始问题：** {input_text}

**执行摘要：**
- 共执行 {executed_count} 个命令
- 剩余 {len(plan)} 个命令未执行

**命令输出：**
{chr(10).join([f"### {cmd}\n```\n{result[:300]}\n```" for cmd, result in past_steps[:3]])}

**说明：**
由于系统处理限制，以上为原始命令输出。如需更详细的分析，请重新提问。
"""


# 辅助函数：快速判断是否应该结束
def should_terminate(state: PlanExecuteState) -> bool:
    """
    快速判断是否应该终止执行

    用于在循环中提前终止
    """
    past_steps = state.get("past_steps", [])
    plan = state.get("plan", [])

    # 条件1：没有剩余计划
    if not plan:
        return True

    # 条件2：已执行命令过多
    if len(past_steps) >= 10:
        return True

    # 条件3：连续失败
    if len(past_steps) >= 3:
        last_three = past_steps[-3:]
        all_failed = all("失败" in result or "错误" in result for _, result in last_three)
        if all_failed:
            logger.warning("连续3个命令失败，终止执行")
            return True

    return False
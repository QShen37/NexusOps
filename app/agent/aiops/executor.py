"""
Executor 节点：执行单个步骤
基于 LangGraph 官方教程实现

集成共享 Skill 状态: 从 state 中读取 planner 写入的 skill 上下文,
依据 skill 的 playbook 与 allowed_tools 引导执行
"""

from typing import Dict, Any, List, Optional
from textwrap import dedent
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_deepseek import ChatDeepSeek
from langgraph.prebuilt import ToolNode
from loguru import logger

from app.config import config
from app.tools import get_current_time, retrieve_knowledge, execute_shell_command
from app.agent.mcp_client import get_mcp_client_with_retry
from .state import PlanExecuteState


def _filter_tools_by_skill(all_tools: List[Any], allowed: Optional[List[str]]) -> List[Any]:
    """根据 skill.allowed_tools 过滤工具

    若 allowed 为空或没有匹配项, 则返回全部工具 (避免无工具可用)
    """
    if not allowed:
        return all_tools

    allowed_set = {name.strip() for name in allowed if name and name.strip()}
    filtered = [t for t in all_tools if getattr(t, "name", None) in allowed_set]

    if not filtered:
        logger.warning(
            f"skill.allowed_tools 没有匹配到任何已注册工具 ({allowed}), 回退使用全部工具"
        )
        return all_tools

    logger.info(
        f"基于 skill.allowed_tools 过滤工具: {len(filtered)}/{len(all_tools)} 个可用"
    )
    return filtered


def _build_executor_system_prompt(state: PlanExecuteState) -> str:
    """根据 state 中的 skill 上下文构建 Executor 的 system prompt"""
    matched_skill = state.get("matched_skill")
    skill_display_name = state.get("skill_display_name")
    skill_playbook = state.get("skill_playbook")
    skill_risk_level = state.get("skill_risk_level")
    skill_allowed_tools = state.get("skill_allowed_tools") or []

    base = dedent("""
        你是一个能力强大的运维助手, 负责执行计划中的具体步骤。

        通用执行原则:
        1. 理解步骤目标
        2. 选择合适的工具 (如果步骤已指定工具, 优先使用)
        3. 调用工具获取真实数据
        4. 返回清晰、准确的执行结果

        硬性要求:
        - 工具调用失败时, 说明失败原因, 不要伪造数据
        - 只返回实际获取到的信息, 严禁编造
        - 专注于当前步骤, 不要提前推进或跳过
    """).strip()

    if not matched_skill:
        return base

    # 拼接 skill 上下文
    skill_block = dedent(f"""

        ## 当前匹配的 Skill 上下文
        - **Skill**: {skill_display_name or matched_skill} ({matched_skill})
        - **风险等级**: {skill_risk_level or 'unknown'}
        - **推荐工具**: {', '.join(skill_allowed_tools[:10]) if skill_allowed_tools else '无限制'}

        ## Skill Playbook (执行参考)
        {skill_playbook or '(此 Skill 未提供 Playbook)'}

        ## Skill 执行约束
        - 优先选择 Playbook 中提到的工具
        - 严格遵守 Playbook 中的注意事项 (如"生产环境慎重启"等)
        - 风险等级为 high 的写操作要先在结果中给出警告
    """).strip()

    return base + "\n\n" + skill_block


async def executor(state: PlanExecuteState) -> Dict[str, Any]:
    """
    执行节点：执行计划中的下一个步骤

    使用 LangGraph 的 ToolNode 自动处理工具调用
    集成共享 skill 状态: 读取 matched_skill / skill_playbook / skill_allowed_tools
    """
    logger.info("=== Executor：执行步骤 ===")

    plan = state.get("plan", [])
    matched_skill = state.get("matched_skill")
    skill_allowed_tools = state.get("skill_allowed_tools")

    if matched_skill:
        logger.info(f"📌 使用共享 Skill 状态: {matched_skill}")

    # 如果计划为空，不执行
    if not plan:
        logger.info("计划为空，跳过执行")
        return {}

    # 取出第一个步骤
    task = plan[0]
    logger.info(f"当前任务: {task}")

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
        logger.info(f"可用工具数量: 本地 {len(local_tools)} + MCP {len(mcp_tools)}")

        # 合并所有工具
        all_tools = local_tools + mcp_tools

        # 根据 skill.allowed_tools 过滤可用工具
        active_tools = _filter_tools_by_skill(all_tools, skill_allowed_tools)

        # 创建 LLM
        llm = ChatDeepSeek(
            model=config.rag_model,
            api_key=config.DEEPSEEK_API_KEY,
            temperature=0
        )
        llm_with_tools = llm.bind_tools(active_tools)

        # 创建工具节点（自动执行工具调用）
        tool_node = ToolNode(active_tools)

        # 构建消息（带 skill 上下文的 system prompt）
        system_prompt = _build_executor_system_prompt(state)
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"请执行以下任务: {task}")
        ]

        # 第一步：LLM 决定是否调用工具
        llm_response = await llm_with_tools.ainvoke(messages)
        logger.info(f"LLM 响应类型: {type(llm_response)}")

        # 第二步：如果有工具调用，执行工具
        if hasattr(llm_response, "tool_calls") and llm_response.tool_calls:
            logger.info(f"检测到 {len(llm_response.tool_calls)} 个工具调用")

            # 使用 ToolNode 自动执行工具
            messages.append(llm_response)
            tool_messages = await tool_node.ainvoke({"messages": messages})

            # 第三步：将工具结果返回给 LLM 生成最终答案
            messages.extend(tool_messages["messages"])
            final_response = await llm_with_tools.ainvoke(messages)
            result = final_response.content if hasattr(final_response, 'content') else str(final_response)
        else:
            # 没有工具调用，直接使用 LLM 的输出
            logger.info("LLM 未调用工具，直接返回结果")
            result = llm_response.content if hasattr(llm_response, 'content') else str(llm_response)

        logger.info(f"步骤执行完成，结果长度: {len(result)}")

        # 返回更新：移除已执行的步骤，添加执行历史
        # skill 相关字段不修改, 由 state 自然透传给下一节点
        return {
            "plan": plan[1:],  # 移除第一个步骤
            "past_steps": [(task, result)],  # 使用 operator.add 追加
        }

    except Exception as e:
        logger.error(f"执行步骤失败: {e}", exc_info=True)
        return {
            "plan": plan[1:],
            "past_steps": [(task, f"执行失败: {str(e)}")],
        }

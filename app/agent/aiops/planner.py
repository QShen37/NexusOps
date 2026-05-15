"""
Planner 节点：制定执行计划
集成 Skills 能力，让计划更智能、更符合运维场景
"""

from textwrap import dedent
from typing import List, Dict, Any, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_deepseek import ChatDeepSeek
from pydantic import BaseModel, Field
from loguru import logger

from app.config import config
from app.tools import get_current_time, retrieve_knowledge, execute_shell_command
from app.agent.mcp_client import get_mcp_client_with_retry
from .state import PlanExecuteState
from .utils import format_tools_description
from .skill_manager import SkillManager, Skill


class Plan(BaseModel):
    """计划的输出格式"""
    steps: List[str] = Field(
        description="完成任务所需的不同步骤。这些步骤应该按顺序执行，每一步都建立在前一步的基础上。"
    )
    suggested_skill: Optional[str] = Field(
        default=None,
        description="推荐使用的 Skill 名称（如果适用）"
    )


class EnhancedPlanner:
    """
    增强版 Planner - 集成 Skills 能力

    特点：
    1. 自动识别问题类型，匹配最合适的 Skill
    2. 基于 Skill 的 Playbook 制定执行计划
    3. 结合经验文档和工具能力
    """

    def __init__(self):
        self.skill_manager = SkillManager()
        self.llm = None

    def _get_llm(self):
        """懒加载 LLM"""
        if self.llm is None:
            self.llm = ChatDeepSeek(
                model=config.rag_model,
                api_key=config.DEEPSEEK_API_KEY,
                temperature=0
            )
        return self.llm


async def planner(state: PlanExecuteState) -> Dict[str, Any]:
    """
    规划节点：根据用户输入生成执行计划

    增强功能：
    1. 自动匹配最合适的 Skill
    2. 基于 Skill Playbook 制定计划
    3. 结合经验文档和可用工具
    """
    logger.info("=== Planner：制定执行计划（集成 Skills） ===")

    input_text = state.get("input", "")
    logger.info(f"用户输入: {input_text}")

    try:
        # ========== Step 1: 匹配最合适的 Skill ==========
        logger.info("🔍 Step 1: 匹配最合适的 Skill...")
        skill_manager = SkillManager()
        matched_skill = skill_manager.match_skill(input_text)

        skill_context = ""
        if matched_skill:
            logger.info(f"✓ 匹配到 Skill: {matched_skill.display_name} ({matched_skill.name})")
            logger.info(f"  描述: {matched_skill.description}")
            logger.info(f"  触发词: {', '.join(matched_skill.triggers[:5])}")

            # 构建 Skill 上下文
            skill_context = dedent(f"""
                ## 匹配到的诊断 Skill
                
                **Skill 名称**: {matched_skill.display_name}
                **描述**: {matched_skill.description}
                **风险等级**: {matched_skill.risk_level}
                
                ### Playbook 内容:
                {matched_skill.content}
                
                ### 该 Skill 推荐使用的工具:
                {', '.join(matched_skill.allowed_tools[:10])}
                
                ---
                **重要**: 请参考上述 Playbook 来制定执行计划。Playbook 中定义了标准的诊断流程，
                你的计划应该遵循 Playbook 中的 Phase 顺序。
            """).strip()
        else:
            logger.info("⚠ 未匹配到特定 Skill，使用通用规划模式")

        # ========== Step 2: 查询内部文档获取相关经验 ==========
        logger.info("📚 Step 2: 查询内部文档...")
        experience_docs = ""
        try:
            # 结合 Skill 信息增强查询
            enhanced_query = input_text
            if matched_skill:
                enhanced_query = f"{input_text} {matched_skill.name} {matched_skill.display_name}"

            context_str = await retrieve_knowledge.ainvoke({"query": enhanced_query})
            if context_str and context_str.strip():
                experience_docs = context_str
                logger.info(f"✓ 找到相关经验文档，长度: {len(experience_docs)}")
            else:
                logger.info("未找到相关经验文档")
        except Exception as e:
            logger.warning(f"查询内部文档失败: {e}")

        # ========== Step 3: 获取可用工具列表 ==========
        logger.info("🔧 Step 3: 获取可用工具...")
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

        # ========== Step 4: 构建增强的 Prompt ==========
        planner_prompt = build_enhanced_planner_prompt()

        # 格式化经验文档上下文
        if experience_docs:
            experience_context = dedent(f"""
                ## 相关经验文档
    
                以下是从知识库中检索到的相关经验和最佳实践，请参考这些经验制定执行计划：
    
                {experience_docs}
    
                ---
            """).strip()
        else:
            experience_context = ""

        # ========== Step 5: 生成计划 ==========
        logger.info("🤖 Step 4: LLM 生成计划...")
        llm = ChatDeepSeek(
            model=config.rag_model,
            api_key=config.DEEPSEEK_API_KEY,
            temperature=0
        )

        planner_chain = planner_prompt | llm.with_structured_output(Plan)

        # 调用 LLM 生成计划
        plan_result = await planner_chain.ainvoke({
            "messages": [("user", input_text)],
            "tools_description": tools_description,
            "experience_context": experience_context,
            "skill_context": skill_context,
            "skill_name": matched_skill.name if matched_skill else "generic_oncall"
        })

        # 提取步骤列表
        if isinstance(plan_result, Plan):
            plan_steps = plan_result.steps
            suggested_skill = plan_result.suggested_skill
        else:
            plan_steps = plan_result.get("steps", [])
            suggested_skill = plan_result.get("suggested_skill")

        # 如果没有建议 skill 但匹配到了，使用匹配的 skill
        if not suggested_skill and matched_skill:
            suggested_skill = matched_skill.name

        logger.info(f"✓ 计划已生成，共 {len(plan_steps)} 个步骤")
        if suggested_skill:
            logger.info(f"💡 推荐使用 Skill: {suggested_skill}")

        for i, step in enumerate(plan_steps, 1):
            logger.info(f"  步骤{i}: {step}")

        # ========== Step 6: 返回增强的计划 + 共享 Skill 状态 ==========
        # 将 skill 完整上下文写入 state, 供 executor / replanner 读取
        return {
            "plan": plan_steps,
            "suggested_skill": suggested_skill,
            "matched_skill": matched_skill.name if matched_skill else None,
            "skill_playbook": matched_skill.content if matched_skill else None,
            "skill_display_name": matched_skill.display_name if matched_skill else None,
            "skill_description": matched_skill.description if matched_skill else None,
            "skill_allowed_tools": matched_skill.allowed_tools if matched_skill else None,
            "skill_risk_level": matched_skill.risk_level if matched_skill else None,
        }

    except Exception as e:
        logger.error(f"生成计划失败: {e}", exc_info=True)
        # 返回一个基于 Skill 的默认计划
        return await fallback_plan(state)


def build_enhanced_planner_prompt() -> ChatPromptTemplate:
    """构建增强版 Planner 提示词（集成 Skills）"""

    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                dedent("""
                    作为一个专家级别的运维规划者，你需要将复杂的运维问题分解为可执行的步骤。
                    
                    ## 你的职责
                    你的职责是制定计划，实际的工具调用由 Executor 负责执行。
                    
                    ## 可用工具列表（供参考）
                    {tools_description}
                    
                    ## Skill 指导（如果有匹配的 Skill）
                    {skill_context}
                    
                    ## 经验文档参考
                    {experience_context}
                    
                    ## 规划原则
                    
                    1. **优先使用 Skill Playbook**：如果提供了匹配的 Skill，请严格遵循其 Playbook 中的 Phase 顺序制定计划。
                    
                    2. **Skill 的典型 Phase 结构**：
                       - Phase 1: 摸底/信息收集（检查状态、获取概览）
                       - Phase 2: 健康度检查（深入分析指标）
                       - Phase 3: 根因分析（定位问题）
                       - Phase 4: 处置建议（给出解决方案）
                    
                    3. **步骤设计原则**：
                       - 每个步骤要明确：做什么、用什么工具、期望得到什么
                       - 步骤之间要有清晰的依赖关系
                       - 步骤描述要具体、可操作
                       - 优先执行信息收集步骤，再进行根因分析
                    
                    4. **工具使用建议**：
                       - 根据 Playbook 中推荐的 allowed_tools 选择工具
                       - 如果 Playbook 中提到的工具不可用，寻找替代方案
                       - 对于生产环境，优先使用只读工具
                    
                    5. **风险意识**：
                       - 高风险操作（重启、删除）应该作为建议而非自动执行
                       - 需要在计划中标注风险等级
                       - 对写操作要有明确的警告
                    
                    6. **经验复用**：
                       - 参考经验文档中的类似案例
                       - 如果经验文档中有 SOP，优先遵循
                    
                    ## 输出格式
                    输出一个包含步骤列表和推荐 Skill 的计划。
                    
                    - 如果问题适合用特定 Skill 解决，在 suggested_skill 字段填写 Skill 名称
                    - 如果问题需要多个 Skill 配合，在步骤中说明
                    
                    ## 示例
                    
                    用户输入："我的 Docker 容器一直在重启"
                    
                    输出：
                    步骤1: 使用 docker_ps 工具获取所有容器列表和状态
                    步骤2: 针对重启的容器，使用 docker_inspect 查看退出码和重启次数
                    步骤3: 使用 docker_logs 获取容器最近 100 行日志，查找 ERROR/OOM 关键字
                    步骤4: 分析退出码（137=OOM, 139=段错误）定位问题
                    步骤5: 根据诊断结果给出建议（增加内存限制、修复配置等）
                    
                    suggested_skill: "container_diagnosis"
                """).strip(),
            ),
            ("placeholder", "{messages}"),
        ]
    )


async def fallback_plan(state: PlanExecuteState) -> Dict[str, Any]:
    """兜底计划生成 - 同时通过 SkillManager 自动匹配 skill 上下文"""
    input_text = state.get("input", "")
    input_lower = input_text.lower()

    # 简单的规则匹配生成计划
    if any(k in input_lower for k in ["容器", "docker", "container"]):
        plan_steps = [
            "使用 docker_ps 查看容器状态",
            "使用 docker_inspect 检查异常容器配置",
            "使用 docker_logs 分析错误日志",
            "根据退出码给出诊断建议"
        ]
        suggested_skill = "container_diagnosis"
    elif any(k in input_lower for k in ["cpu", "内存", "磁盘", "进程"]):
        plan_steps = [
            "获取系统 CPU 和内存使用情况",
            "列出占用资源最高的进程",
            "检查磁盘使用率",
            "分析资源瓶颈并给出优化建议"
        ]
        suggested_skill = "host_resource_diagnosis"
    elif any(k in input_lower for k in ["网络", "ping", "连接", "超时"]):
        plan_steps = [
            "使用 ping 测试基础连通性",
            "检查 DNS 解析是否正常",
            "测试目标端口是否开放",
            "分析网络延迟和丢包情况"
        ]
        suggested_skill = "network_diagnosis"
    else:
        plan_steps = [
            "收集系统基础信息",
            "分析用户描述的问题现象",
            "查询相关经验文档",
            "给出排查建议"
        ]
        suggested_skill = "generic_oncall"

    # 兜底分支也通过 SkillManager 加载 skill 详情, 与正常路径保持一致
    skill = SkillManager().get_skill(suggested_skill)
    return {
        "plan": plan_steps,
        "suggested_skill": suggested_skill,
        "matched_skill": skill.name if skill else suggested_skill,
        "skill_playbook": skill.content if skill else None,
        "skill_display_name": skill.display_name if skill else None,
        "skill_description": skill.description if skill else None,
        "skill_allowed_tools": skill.allowed_tools if skill else None,
        "skill_risk_level": skill.risk_level if skill else None,
    }
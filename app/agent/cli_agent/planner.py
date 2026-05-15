"""
CLI Planner 节点：为命令行任务制定执行计划

集成 Skills 能力：
1. 自动识别问题类型, 匹配最合适的 Skill
2. 基于 Skill 的 Playbook 制定命令计划
3. 将 Skill 上下文写入 state, 供 executor / replanner 复用
"""

from textwrap import dedent
from typing import List, Dict, Any, Optional
import re
from langchain_core.prompts import ChatPromptTemplate
from langchain_deepseek import ChatDeepSeek
from pydantic import BaseModel, Field
from loguru import logger

from app.config import config
from app.tools.cli_tools import execute_shell_command
from app.agent.aiops.skill_manager import SkillManager
from .state import PlanExecuteState
from .utils import format_tools_description


# 匹配 LLM 可能塞进 step 的「步骤前缀」: 步骤1: / Step 1. / 1. / 1) / - 等
# 支持两种独立形式: 1) "- " 这种纯列表 bullet  2) "步骤1:" / "1." / "Step 2." 这种带数字编号
_STEP_PREFIX_RE = re.compile(
    r"^\s*(?:"
    r"[-*•]\s+"  # 纯 markdown 列表 bullet, 例如 "- docker ps"
    r"|"
    r"(?:[-*•]\s*)?(?:步骤|step|阶段|phase)?\s*[#]?\s*\d+[\.\)、:：]?\s*"  # 带数字编号
    r")",
    re.IGNORECASE,
)
# 匹配 markdown 代码块围栏
_CODE_FENCE_RE = re.compile(r"^```[\w\-]*\s*|\s*```$")
# 匹配纯 markdown 标题
_MARKDOWN_HEADER_RE = re.compile(r"^#+\s+")


def _looks_like_shell_command(text: str) -> bool:
    """判断字符串看起来是不是有效 shell 命令

    - 必须包含至少一个 ASCII 字母 (cmdlet / 命令名都是英文)
    - 不能整段是纯中文描述
    """
    if not text:
        return False
    # 至少有一个 ASCII 字母序列(命令名/参数)
    if not re.search(r"[A-Za-z]", text):
        return False
    # 砍掉空白后, 第一个非空白 token 必须是英文/符号开头(命令名)
    # 不能是 "查看..." "检查..." 这种纯中文开头
    first_char = text.lstrip()[:1]
    if first_char and "一" <= first_char <= "鿿":
        return False
    return True


def _clean_command_step(raw: Any) -> str:
    """把 LLM 返回的 step 还原成可执行的 shell 命令字符串

    清理逻辑:
    - 非字符串直接转字符串
    - 去掉首尾空白 / 反引号 / 代码块围栏
    - 去掉 "步骤N:" / "Step N." / "1." / "1)" 等前缀
    - 去掉 "PS> " / "PS C:\\> " / "$ " / "> " 等 prompt 前缀
    - 整段是注释 (# .../ // ...) 或纯 markdown 标题, 视为无效 step 返回 ""
    - 整段是纯中文描述 (没有任何 ASCII 字母, 或以中文起头), 视为无效 step 返回 ""
    - 多行命令保留(PowerShell 支持反引号续行), 但去掉每行的 prompt 前缀
    """
    if raw is None:
        return ""
    if not isinstance(raw, str):
        raw = str(raw)

    text = raw.strip().strip("`").strip()
    # 去 markdown 代码块围栏
    text = _CODE_FENCE_RE.sub("", text).strip()
    if not text:
        return ""

    # 整段是 markdown header 或纯注释 → 丢弃
    if _MARKDOWN_HEADER_RE.match(text):
        return ""
    stripped_for_check = text.lstrip()
    if stripped_for_check.startswith(("# ", "// ")) and "\n" not in text:
        return ""

    # 多行 step 逐行清理 prompt 前缀, 删除全注释行
    cleaned_lines = []
    for line in text.splitlines():
        line = line.rstrip()
        # 去掉 step prefix(只在第一行尝试)
        if not cleaned_lines:
            line = _STEP_PREFIX_RE.sub("", line, count=1)
        # 去掉 shell prompt 前缀
        line = re.sub(r"^(?:PS\s*[A-Z]:\\?[^>]*>|PS>|>>>|\$|>)\s*", "", line)
        if not line.strip():
            continue
        # 跳过纯注释行
        if re.match(r"^\s*(#|//)\s+", line):
            continue
        cleaned_lines.append(line)

    result = "\n".join(cleaned_lines).strip()
    if not result:
        return ""

    # 兜底检查: 不像 shell 命令的步骤丢掉, 防止把"检查端口"当命令执行
    if not _looks_like_shell_command(result):
        logger.warning(f"step 不像 shell 命令, 丢弃: {result[:80]}")
        return ""

    return result


class CLIPlan(BaseModel):
    """CLI 计划的输出格式"""
    steps: List[str] = Field(
        description="完成任务所需执行的 shell 命令。每个命令都应该独立、可执行，并按照正确的顺序排列。"
    )
    suggested_skill: Optional[str] = Field(
        default=None,
        description="推荐使用的 Skill 名称(如果适用)"
    )


def _build_cli_planner_prompt() -> ChatPromptTemplate:
    """构建增强版 CLI Planner 提示词(集成 Skills)"""

    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                dedent("""
                    你是一个专业的 Windows 系统运维专家和命令行规划者。
                    你的任务是将用户的自然语言需求, 转化为一系列可执行的 shell 命令。

                    ## 可用工具
                    你只能使用 execute_shell_command 工具来执行命令。

                    ## Skill 指导(如果有匹配的 Skill)
                    {skill_context}

                    ## 核心原则
                    1. **优先遵循 Skill Playbook**: 若有匹配的 Skill, 严格按 Playbook 的 Phase 顺序拆解命令
                    2. **命令要精确**: 每个命令应该直接解决用户需求的一个方面
                    3. **优先级排序**: 先执行查询类命令, 再执行操作类命令
                    4. **考虑依赖**: 若后续命令依赖前一个命令的结果, 确保顺序正确
                    5. **限制输出**: 对可能输出大量数据的命令, 添加 Select-Object / findstr / Format-Table 等限制
                    6. **安全性优先**: 优先使用只读命令(查看、查询), 避免破坏性操作
                    7. **命令组合**: 合理使用管道(|)组合多个简单命令

                    ## 常见场景示例(注意: 每个 step 都是原始 shell 命令字符串, 不带任何前缀和注释)

                    ### 1. 端口检查
                    用户需求："查看8004端口是否正在使用"
                    steps:
                      - Get-NetTCPConnection -LocalPort 8004 -ErrorAction SilentlyContinue
                      - netstat -ano | findstr :8004

                    ### 2. 进程检查
                    用户需求："查看nginx进程是否在运行"
                    steps:
                      - Get-Process nginx -ErrorAction SilentlyContinue
                      - tasklist | findstr nginx

                    ### 3. 系统资源检查
                    用户需求："检查系统CPU、内存、磁盘使用情况"
                    steps:
                      - Get-Counter '\\Processor(_Total)\\% Processor Time'
                      - Get-CimInstance Win32_OperatingSystem | Select-Object TotalVisibleMemorySize,FreePhysicalMemory
                      - Get-PSDrive -PSProvider FileSystem

                    ### 4. 容器检查 (会触发 container_diagnosis Skill)
                    用户需求："我的 milvus 容器是不是挂了"
                    steps:
                      - docker ps -a --format "table {{.Names}}\t{{.Status}}\t{{.Image}}"
                      - docker inspect milvus --format '{{.State.Status}}/{{.State.ExitCode}}/{{.RestartCount}}'
                      - docker logs milvus --tail 100

                    ## 输出格式硬性要求 (违反则计划被丢弃, 必须严格遵守!)
                    1. **每个 step 必须是一条原始 shell 命令**, 直接复制到 PowerShell 即可执行
                    2. **禁止前缀**: 不能写 "步骤1:" / "Step 1." / "1." / "1)" / "- " 等任何编号或列表前缀
                    3. **禁止注释**: 不能写 "# xxx" / "// xxx" / "<!-- xxx -->" 等注释行作为 step
                    4. **禁止代码块**: 不能用 ``` 包裹命令; 不能加 markdown 标题
                    5. **禁止纯描述**: 不能写"检查端口"、"查看进程"等纯中文描述, 只能写真正的命令
                    6. **禁止 prompt 前缀**: 不能写 "PS> " / "PS C:\\>" / "$ " / "> " 等 shell 提示符
                    7. 若问题适合用特定 Skill 解决, 在 suggested_skill 字段填写 Skill 名称

                    ❌ 错误示例(不要这样写):
                      - "步骤1: Get-NetTCPConnection -LocalPort 8004"
                      - "# 先检查端口"
                      - "PS> Get-Process nginx"
                      - "1. tasklist | findstr nginx"

                    ✅ 正确示例(应该这样写):
                      - "Get-NetTCPConnection -LocalPort 8004"
                      - "Get-Process nginx -ErrorAction SilentlyContinue"
                      - "tasklist | findstr nginx"

                    ## 注意事项
                    - 避免生成破坏性命令(Remove-Item -Recurse -Force、format、diskpart clean 等)
                    - 避免使用管理员权限命令, 除非用户明确要求
                    - 如果 Playbook 中有"生产环境慎重启"等约束, 不要把写操作放进计划中
                """).strip(),
            ),
            ("placeholder", "{messages}"),
        ]
    )


async def cli_planner(state: PlanExecuteState) -> Dict[str, Any]:
    """
    CLI 规划节点：根据用户输入生成需要执行的 shell 命令列表

    流程:
    1. 通过 SkillManager 匹配最合适的 Skill
    2. 把 Skill Playbook 注入到 prompt
    3. LLM 基于 Playbook + 用户输入生成命令计划
    4. 将 Skill 上下文写入 state, 供后续节点复用
    """
    logger.info("=== CLI Planner：制定命令行执行计划(集成 Skills) ===")

    input_text = state.get("input", "")
    logger.info(f"用户输入: {input_text}")

    past_steps = state.get("past_steps", [])
    if past_steps:
        logger.info(f"已有历史步骤: {len(past_steps)} 个")

    # ========== Step 1: 匹配 Skill ==========
    logger.info("🔍 Step 1: 匹配最合适的 Skill...")
    skill_manager = SkillManager()
    matched_skill = skill_manager.match_skill(input_text)

    skill_context = ""
    if matched_skill:
        logger.info(
            f"✓ 匹配到 Skill: {matched_skill.display_name} "
            f"({matched_skill.name}), 风险={matched_skill.risk_level}"
        )
        skill_context = dedent(f"""
            ## 匹配到的诊断 Skill
            - **Skill 名称**: {matched_skill.display_name} ({matched_skill.name})
            - **描述**: {matched_skill.description}
            - **风险等级**: {matched_skill.risk_level}
            - **推荐工具(白名单)**: {', '.join(matched_skill.allowed_tools[:10]) if matched_skill.allowed_tools else '无限制'}

            ### Playbook 内容:
            {matched_skill.content}

            ---
            **重要**: 请严格按 Playbook 的 Phase 顺序生成命令。Playbook 是该类问题的标准排查流程。
        """).strip()
    else:
        logger.info("⚠ 未匹配到特定 Skill, 使用通用规划模式")

    # ========== Step 2: LLM 生成计划 ==========
    try:
        tools = [execute_shell_command]
        tools_description = format_tools_description(tools)

        llm = ChatDeepSeek(
            model=config.rag_model,
            api_key=config.DEEPSEEK_API_KEY,
            temperature=0
        )

        planner_prompt = _build_cli_planner_prompt()
        planner_chain = planner_prompt | llm.with_structured_output(CLIPlan)

        # 构建用户消息(包含历史上下文)
        user_message = input_text
        if past_steps:
            history_summary = "\n".join([
                f"已执行步骤 {i + 1}: {step}"
                for i, step in enumerate(past_steps)
            ])
            user_message = (
                f"历史执行记录:\n{history_summary}\n\n"
                f"当前继续任务:\n{input_text}\n\n"
                "请根据已执行的结果, 规划剩余需要执行的命令。"
            )

        logger.info("🤖 Step 2: LLM 生成命令计划...")
        plan_result = await planner_chain.ainvoke({
            "messages": [("user", user_message)],
            "tools_description": tools_description,
            "skill_context": skill_context or "(本次未匹配到特定 Skill, 按通用经验规划即可)",
        })

        # 提取步骤列表
        if isinstance(plan_result, CLIPlan):
            plan_steps = plan_result.steps
            suggested_skill = plan_result.suggested_skill
        else:
            plan_steps = plan_result.get("steps", [])  # type: ignore
            suggested_skill = plan_result.get("suggested_skill")  # type: ignore

        # 验证和优化命令
        validated_steps = []
        for step in plan_steps:
            cleaned = _clean_command_step(step)
            if not cleaned:
                continue
            dangerous_patterns = [
                "rm -rf /", "rm -rf /*", "mkfs", "dd if=",
                ":(){ :|:& };:", "> /dev/sda"
            ]
            if any(pattern in cleaned for pattern in dangerous_patterns):
                logger.warning(f"检测到潜在危险命令: {cleaned}")
                cleaned = f"[危险命令] {cleaned}"
            validated_steps.append(cleaned)

        # 没建议则回退到匹配的 skill
        if not suggested_skill and matched_skill:
            suggested_skill = matched_skill.name

        logger.info(f"✓ 命令计划已生成, 共 {len(validated_steps)} 个命令")
        if suggested_skill:
            logger.info(f"💡 推荐使用 Skill: {suggested_skill}")
        for i, step in enumerate(validated_steps, 1):
            logger.info(f"  命令{i}: {step[:100]}{'...' if len(step) > 100 else ''}")

        return {
            "plan": validated_steps,
            "suggested_skill": suggested_skill,
            "matched_skill": matched_skill.name if matched_skill else None,
            "skill_playbook": matched_skill.content if matched_skill else None,
            "skill_display_name": matched_skill.display_name if matched_skill else None,
            "skill_description": matched_skill.description if matched_skill else None,
            "skill_allowed_tools": matched_skill.allowed_tools if matched_skill else None,
            "skill_risk_level": matched_skill.risk_level if matched_skill else None,
        }

    except Exception as e:
        logger.error(f"生成命令计划失败: {e}", exc_info=True)
        return await _fallback_cli_plan(state, matched_skill)


async def _fallback_cli_plan(
    state: PlanExecuteState,
    matched_skill: Optional[Any] = None,
) -> Dict[str, Any]:
    """LLM 失败时的兜底命令计划"""
    input_text = state.get("input", "")
    input_lower = input_text.lower()

    if "端口" in input_lower or "port" in input_lower:
        import re
        port_match = re.search(r'(\d{4,5})', input_text)
        port = port_match.group(1) if port_match else "8000"
        default_plan = [
            f"Get-NetTCPConnection -LocalPort {port} -ErrorAction SilentlyContinue",
            f"netstat -ano | findstr :{port}",
        ]
    elif "进程" in input_lower or "process" in input_lower:
        import re
        proc_match = re.search(r'([a-zA-Z_][a-zA-Z0-9_]*)', input_text)
        proc_name = proc_match.group(1) if proc_match else "python"
        default_plan = [
            f"Get-Process {proc_name} -ErrorAction SilentlyContinue",
            f"tasklist | findstr {proc_name}",
        ]
    elif "cpu" in input_lower or "内存" in input_lower or "memory" in input_lower:
        default_plan = [
            "Get-Counter '\\Processor(_Total)\\% Processor Time'",
            "Get-CimInstance Win32_OperatingSystem | Select-Object TotalVisibleMemorySize,FreePhysicalMemory",
        ]
    elif "磁盘" in input_lower or "disk" in input_lower:
        default_plan = ["Get-PSDrive -PSProvider FileSystem"]
    elif "日志" in input_lower or "log" in input_lower:
        default_plan = ["Get-EventLog -LogName System -EntryType Error -Newest 50"]
    elif any(k in input_lower for k in ["容器", "docker", "container"]):
        default_plan = [
            'docker ps -a --format "table {{.Names}}\t{{.Status}}\t{{.Image}}"',
            "docker stats --no-stream",
        ]
    else:
        default_plan = [
            "Get-Date",
            "Get-ComputerInfo | Select-Object CsName,OsName,OsVersion",
        ]

    logger.info(f"使用兜底计划: {default_plan}")

    return {
        "plan": default_plan,
        "suggested_skill": matched_skill.name if matched_skill else None,
        "matched_skill": matched_skill.name if matched_skill else None,
        "skill_playbook": matched_skill.content if matched_skill else None,
        "skill_display_name": matched_skill.display_name if matched_skill else None,
        "skill_description": matched_skill.description if matched_skill else None,
        "skill_allowed_tools": matched_skill.allowed_tools if matched_skill else None,
        "skill_risk_level": matched_skill.risk_level if matched_skill else None,
    }

"""
CLI Planner 节点：为命令行任务制定执行计划
基于 LangGraph 官方教程实现
"""

from textwrap import dedent
from typing import List, Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_deepseek import ChatDeepSeek
from pydantic import BaseModel, Field
from loguru import logger

from app.config import config
from app.tools.cli_tools import execute_shell_command
from .state import PlanExecuteState
from .utils import format_tools_description


class CLIPlan(BaseModel):
    """CLI 计划的输出格式"""
    steps: List[str] = Field(
        description="完成任务所需执行的 shell 命令。每个命令都应该独立、可执行，并按照正确的顺序排列。"
    )


# CLI Planner 提示词
cli_planner_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            dedent("""
                你是一个专业的 Windows 系统运维专家和命令行规划者。你的任务是将用户的自然语言需求，转化为一系列可执行的 shell 命令。

                ## 可用工具
                你只能使用 execute_shell_command 工具来执行命令。这个工具可以执行任意 shell 命令。

                ## 核心原则
                1. **命令要精确**：每个命令应该直接解决用户需求的一个方面
                2. **优先级排序**：先执行查询类命令，再执行操作类命令
                3. **考虑依赖**：如果后续命令依赖前一个命令的结果，确保顺序正确
                4. **限制输出**：对于可能输出大量数据的命令，添加 Select-Object、findstr、Format-Table 等限制
                5. **安全性优先**：优先使用只读命令（查看、查询），避免破坏性操作
                6. **命令组合**：合理使用管道（|）组合多个简单命令

                ## 常见场景的命令规划示例

                ### 1. 端口检查
                用户需求："查看8004端口是否正在使用"
                规划：
                步骤1: Get-NetTCPConnection -LocalPort 8004 -ErrorAction SilentlyContinue
                步骤2: netstat -ano | findstr :8004

                ### 2. 进程检查
                用户需求："查看nginx进程是否在运行"
                规划：
                步骤1: Get-Process nginx -ErrorAction SilentlyContinue
                步骤2: tasklist | findstr nginx

                ### 3. 系统资源检查
                用户需求："检查系统CPU、内存、磁盘使用情况"
                规划：
                步骤1: Get-Counter '\\Processor(_Total)\\% Processor Time'
                步骤2: Get-CimInstance Win32_OperatingSystem | Select-Object TotalVisibleMemorySize,FreePhysicalMemory
                步骤3: Get-PSDrive -PSProvider FileSystem

                ### 4. 日志分析
                用户需求："查看最近的系统错误日志"
                规划：
                步骤1: Get-EventLog -LogName System -EntryType Error -Newest 100

                ### 5. 文件操作
                用户需求："查找当前目录下所有的Python文件"
                规划：
                步骤1: Get-ChildItem -Path . -Recurse -Filter "*.py"

                ### 6. 网络诊断
                用户需求："检查网络连通性和端口开放情况"
                规划：
                步骤1: Test-Connection google.com -Count 4
                步骤2: Get-NetTCPConnection -State Listen

                ### 7. 性能分析
                用户需求："找出占用CPU最高的进程"
                规划：
                步骤1: Get-Process | Sort-Object CPU -Descending | Select-Object -First 10

                ### 8. 服务状态
                用户需求："查看所有运行中的docker容器"
                规划：
                步骤1: docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Image}}"

                ## 输出要求
                1. 每个步骤必须是一个完整的、可执行的 shell 命令
                2. 命令应该是独立的，不依赖前一个步骤的输出（除非明确需要）
                3. 如果前一个命令的输出是后续命令的输入，使用管道或临时文件
                4. 对于复杂的任务，可以将命令拆分为多个简单的步骤

                ## 注意事项
                - 避免生成破坏性命令（Remove-Item -Recurse -Force、format、diskpart clean 等）
                - 避免使用管理员权限命令，除非用户明确要求
                - 对于需要多步操作的复杂任务，要详细规划每一步
                - 如果用户需求不明确，生成一个询问性质的信息收集命令

                现在，请根据用户的需求，制定详细的命令行执行计划。
            """).strip()
        ),
        ("placeholder", "{messages}"),
    ]
)


async def cli_planner(state: PlanExecuteState) -> Dict[str, Any]:
    """
    CLI 规划节点：根据用户输入生成需要执行的 shell 命令列表

    流程：
    1. 分析用户需求
    2. 生成对应的 shell 命令列表
    3. 按顺序返回命令计划

    Args:
        state: 当前状态，包含用户的输入

    Returns:
        包含计划步骤的字典
    """
    logger.info("=== CLI Planner：制定命令行执行计划 ===")

    input_text = state.get("input", "")
    logger.info(f"用户输入: {input_text}")

    # 获取历史步骤（如果有）
    past_steps = state.get("past_steps", [])
    if past_steps:
        logger.info(f"已有历史步骤: {len(past_steps)} 个")

    try:
        # 准备可用工具描述
        tools = [execute_shell_command]
        tools_description = format_tools_description(tools)

        # 创建 LLM
        llm = ChatDeepSeek(
            model=config.rag_model,
            api_key=config.DEEPSEEK_API_KEY,
            temperature=0  # 使用较低温度确保计划稳定性
        )

        # 创建规划链
        planner_chain = cli_planner_prompt | llm.with_structured_output(CLIPlan)

        # 构建用户消息（包含历史上下文）
        user_message = input_text
        if past_steps:
            history_summary = "\n".join([
                f"已执行步骤 {i + 1}: {step}"
                for i, step in enumerate(past_steps)
            ])
            user_message = f"""
            历史执行记录：
            {history_summary}

            当前继续任务：
            {input_text}

            请根据已执行的结果，规划剩余需要执行的命令。
            """

        # 调用 LLM 生成计划
        logger.info("正在生成命令执行计划...")
        plan_result = await planner_chain.ainvoke({
            "messages": [("user", user_message)],
            "tools_description": tools_description
        })

        # 提取步骤列表
        if isinstance(plan_result, CLIPlan):
            plan_steps = plan_result.steps
        else:
            # 如果返回的是字典，提取 steps 字段
            plan_steps = plan_result.get("steps", [])  # type: ignore

        # 验证和优化命令
        validated_steps = []
        for step in plan_steps:
            # 移除多余的空格和换行
            step = step.strip()
            # 检查是否为空
            if not step:
                continue
            # 检查危险命令模式
            dangerous_patterns = [
                "rm -rf /", "rm -rf /*", "mkfs", "dd if=",
                ":(){ :|:& };:", "> /dev/sda"
            ]
            is_dangerous = any(pattern in step for pattern in dangerous_patterns)
            if is_dangerous:
                logger.warning(f"检测到潜在危险命令: {step}")
                # 添加警告标记，但不阻止（执行时会再次请求确认）
                step = f"[危险命令] {step}"
            validated_steps.append(step)

        logger.info(f"命令计划已生成，共 {len(validated_steps)} 个命令")
        for i, step in enumerate(validated_steps, 1):
            logger.info(f"命令{i}: {step[:100]}{'...' if len(step) > 100 else ''}")

        return {
            "plan": validated_steps,
            "current_step_index": 0,  # 当前执行到第几步
            "command_results": []  # 存储命令执行结果
        }

    except Exception as e:
        logger.error(f"生成命令计划失败: {e}", exc_info=True)

        # 根据输入类型返回默认计划
        default_plan = []

        # 智能判断默认计划
        input_lower = input_text.lower()
        if "端口" in input_lower or "port" in input_lower:
            # 提取端口号
            import re
            port_match = re.search(r'(\d{4,5})', input_text)
            port = port_match.group(1) if port_match else "8000"
            default_plan = [
                f"lsof -i:{port} 2>/dev/null || netstat -tuln | grep {port}"
            ]
        elif "进程" in input_lower or "process" in input_lower:
            # 提取进程名
            import re
            proc_match = re.search(r'([a-zA-Z_][a-zA-Z0-9_]*)', input_text)
            proc_name = proc_match.group(1) if proc_match else "python"
            default_plan = [
                f"ps aux | grep {proc_name} | grep -v grep"
            ]
        elif "cpu" in input_lower or "内存" in input_lower or "memory" in input_lower:
            default_plan = [
                "top -bn1 | head -20",
                "free -h"
            ]
        elif "磁盘" in input_lower or "disk" in input_lower or "df" in input_lower:
            default_plan = [
                "df -h"
            ]
        elif "日志" in input_lower or "log" in input_lower:
            default_plan = [
                "tail -n 100 /var/log/syslog 2>/dev/null || tail -n 100 /var/log/messages 2>/dev/null"
            ]
        else:
            # 通用默认计划
            default_plan = [
                "echo '正在收集系统信息...'",
                "uname -a",
                "uptime"
            ]

        logger.info(f"使用默认计划: {default_plan}")
        return {
            "plan": default_plan,
            "current_step_index": 0,
            "command_results": []
        }


# 专门用于复杂任务的 Planner
async def cli_complex_planner(state: PlanExecuteState) -> Dict[str, Any]:
    """
    复杂任务规划器：用于需要多步骤、有依赖关系的复杂命令行任务

    适用场景：
    - 需要综合分析多个命令结果
    - 命令之间有数据依赖
    - 需要根据前一个命令的结果决定后续命令
    """
    logger.info("=== CLI Complex Planner：制定复杂任务计划 ===")

    input_text = state.get("input", "")

    complex_plan_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                dedent("""
                    你是一个专家级的运维自动化工程师。对于复杂的命令行任务，你需要制定详细的执行计划。

                    ## 复杂任务特点
                    1. 需要执行多个相关的命令
                    2. 后续命令依赖于前面命令的输出
                    3. 可能需要进行条件判断
                    4. 需要综合分析多个数据源

                    ## 规划策略
                    对于复杂任务，你应该：
                    1. 将任务分解为多个阶段
                    2. 每个阶段包含一组相关的命令
                    3. 明确阶段之间的数据传递
                    4. 考虑错误处理和回滚机制

                    ## 输出格式
                    每个步骤应该清晰说明：
                    - 步骤目标
                    - 具体命令
                    - 预期输出
                    - 如何传递给下一步

                    请为复杂任务生成详细的执行计划。
                """).strip(),
            ),
            ("placeholder", "{messages}"),
        ]
    )

    try:
        llm = ChatDeepSeek(
            model=config.rag_model,
            api_key=config.DEEPSEEK_API_KEY,
            temperature=0
        )

        planner_chain = complex_plan_prompt | llm.with_structured_output(CLIPlan)

        plan_result = await planner_chain.ainvoke({
            "messages": [("user", input_text)],
            "tools_description": format_tools_description([execute_shell_command])
        })

        plan_steps = plan_result.steps if isinstance(plan_result, CLIPlan) else plan_result.get("steps", [])

        logger.info(f"复杂任务计划已生成，共 {len(plan_steps)} 个阶段")

        return {
            "plan": plan_steps,
            "is_complex": True,
            "current_phase": 0
        }

    except Exception as e:
        logger.error(f"生成复杂任务计划失败: {e}")
        return {
            "plan": [],
            "is_complex": False
        }
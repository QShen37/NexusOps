"""
CLI Executor 节点：执行 Windows PowerShell/CMD 命令

特性:
- 自动模式: 直接执行所有命令
- 交互模式: 通过 cli_service 把命令推送给前端, 等待前端用户确认
- 危险命令检测: 在 confirm 事件中带上 danger 标记
- 共享 Skill 上下文: 从 state 读取 planner 注入的 skill 信息
"""

import re
from typing import Dict, Any, Optional, Tuple
from textwrap import dedent
from loguru import logger

from app.tools.cli_tools import execute_confirmed_command
from .state import PlanExecuteState


# Windows 危险命令模式 - 用于在 confirm 事件中给前端标红
DANGEROUS_PATTERNS = [
    (r"Remove-Item\s+.*-Recurse.*-Force", "⚠️ 极度危险：递归强制删除文件！", True),
    (r"rd\s+/s\s+/q", "⚠️ 危险：将递归删除目录！", True),
    (r"del\s+/f\s+/s\s+/q", "⚠️ 危险：将强制删除多个文件！", True),
    (r"format\s+[A-Z]:", "⚠️ 极度危险：格式化磁盘操作！", True),
    (r"diskpart", "⚠️ 危险：磁盘分区操作可能导致数据丢失！", True),
    (r"shutdown\s+/s", "⚠️ 注意：系统即将关机！", True),
    (r"Stop-Computer", "⚠️ 注意：PowerShell 关机命令！", True),
    (r"Restart-Computer", "⚠️ 注意：系统即将重启！", True),
    (r"taskkill\s+/f", "⚠️ 注意：强制终止进程！", False),
    (r"Stop-Process\s+.*-Force", "⚠️ 注意：强制停止进程！", False),
    (r"Set-ExecutionPolicy", "⚠️ 注意：正在修改 PowerShell 执行策略！", False),
    (r"sc\s+delete", "⚠️ 危险：删除 Windows 服务！", True),
    (r"reg\s+delete", "⚠️ 危险：删除注册表项！", True),
]

WARNING_PATTERNS = [
    (r"del\s+", "⚠️ 注意：删除操作会永久移除文件"),
    (r"move\s+", "⚠️ 注意：文件移动操作"),
    (r"copy\s+", "⚠️ 注意：文件复制操作"),
    (r"taskkill", "⚠️ 注意：终止进程操作"),
    (r"Stop-Service", "⚠️ 注意：停止服务操作"),
    (r"Restart-Service", "⚠️ 注意：重启服务操作"),
    (r">\s*\S+", "⚠️ 注意：重定向输出可能覆盖文件"),
]


def _check_command_dangerous(command: str) -> Tuple[bool, str]:
    """检查 Windows 命令是否危险, 返回 (is_dangerous, warning_message)"""
    for pattern, warning, is_dangerous in DANGEROUS_PATTERNS:
        if re.search(pattern, command, re.IGNORECASE):
            return is_dangerous, warning

    for pattern, warning in WARNING_PATTERNS:
        if re.search(pattern, command, re.IGNORECASE):
            return False, warning

    return False, ""


def _wrap_powershell(command: str) -> str:
    """命令未指定 shell 时自动包一层 powershell -Command"""
    if not command.lower().startswith(("powershell", "cmd")):
        return f'powershell -Command "{command}"'
    return command


def _normalize_result(raw_result: Any) -> str:
    """统一把命令结果转成 string"""
    if raw_result is None:
        return ""
    if isinstance(raw_result, str):
        return raw_result
    if isinstance(raw_result, bytes):
        return raw_result.decode("utf-8", errors="ignore")
    if isinstance(raw_result, dict):
        import json
        return json.dumps(raw_result, ensure_ascii=False, indent=2)
    return str(raw_result)


def _run_command(command: str, timeout: int = 30) -> str:
    """运行已确认的命令并返回 string 化的结果"""
    final_command = _wrap_powershell(command)
    try:
        raw_result = execute_confirmed_command(final_command, timeout=timeout)
        return _normalize_result(raw_result)
    except Exception as e:
        return f"命令执行失败: {str(e)}"


async def cli_executor(state: PlanExecuteState) -> Dict[str, Any]:
    """
    CLI 执行节点：执行 Windows PowerShell/CMD 命令

    - auto_mode=True: 直接执行
    - auto_mode=False: 通过 cli_service 把命令推送给前端, 等待用户确认
    """
    logger.info("=== CLI Executor：执行命令 ===")

    plan = state.get("plan", [])
    auto_mode = state.get("auto_mode", False)
    session_id = state.get("session_id", "default")
    matched_skill = state.get("matched_skill")

    if matched_skill:
        logger.info(f"📌 使用共享 Skill 状态: {matched_skill}")

    if not plan:
        logger.info("计划为空, 跳过执行")
        return {}

    # 获取当前命令
    current_command = plan[0]
    past_steps_count = len(state.get("past_steps", []))
    current_step_index = past_steps_count + 1
    total_steps = len(plan) + past_steps_count

    logger.info(f"执行命令 [{current_step_index}/{total_steps}]: {current_command[:100]}")

    # 获取步骤描述
    step_descriptions = state.get("step_descriptions") or {}
    step_description = step_descriptions.get(current_step_index, "")

    final_command = current_command
    edited_command: Optional[str] = None
    new_auto_mode = auto_mode

    # ========== 交互模式: 请求前端确认 ==========
    if not auto_mode:
        is_dangerous, warning = _check_command_dangerous(current_command)

        # 懒加载 service, 避免循环引用
        from app.services.cli_service import cli_service

        confirmation = await cli_service.request_confirmation(
            session_id=session_id,
            command=current_command,
            step_index=current_step_index,
            total_steps=total_steps,
            step_description=step_description,
            dangerous=is_dangerous,
            warning=warning,
            matched_skill=matched_skill,
            skill_display_name=state.get("skill_display_name"),
            skill_risk_level=state.get("skill_risk_level"),
        )

        action = (confirmation or {}).get("action", "skip")
        edited_command = (confirmation or {}).get("edited_command") or None
        new_auto_mode = bool((confirmation or {}).get("auto_mode", False))

        logger.info(
            f"前端确认结果: action={action}, edited={edited_command is not None}, "
            f"auto_mode={new_auto_mode}"
        )

        if action == "quit":
            return {
                "plan": [],
                "past_steps": [(current_command, "任务被用户终止")],
                "task_status": "terminated",
                "response": "用户终止了任务执行",
            }

        if action == "skip":
            logger.info(f"用户跳过命令: {current_command}")
            return {
                "plan": plan[1:],
                "past_steps": [(current_command, "⏭️ 用户跳过此命令")],
                "auto_mode": new_auto_mode,
            }

        # execute / edit 都会走到这里
        if edited_command:
            final_command = edited_command
            logger.info(f"使用前端编辑后的命令: {final_command}")
        else:
            final_command = current_command

    # ========== 执行命令 ==========
    logger.info(f"🔧 正在执行命令: {final_command[:120]}")
    result = _run_command(final_command, timeout=30)
    logger.info(f"命令执行完成, 结果长度: {len(result)}")

    execution_record = {
        "command": current_command,
        "edited_command": edited_command,
        "executed_command": final_command,
        "result": result,
        "status": "success" if "失败" not in result and "错误" not in result else "failed",
    }

    return {
        "plan": plan[1:],
        "past_steps": [(current_command, result)],
        "execution_records": state.get("execution_records", []) + [execution_record],
        "auto_mode": new_auto_mode,
    }


async def cli_batch_executor(state: PlanExecuteState) -> Dict[str, Any]:
    """批量执行器：一次性执行多个 Windows 命令(不询问确认)"""
    logger.info("=== CLI Batch Executor：批量执行命令 ===")

    plan = state.get("plan", [])
    if not plan:
        logger.info("计划为空, 跳过执行")
        return {}

    results = []
    for i, command in enumerate(plan, 1):
        logger.info(f"批量执行 [{i}/{len(plan)}]: {command[:100]}")
        result = _run_command(command, timeout=30)
        results.append((command, result))

    return {
        "plan": [],
        "past_steps": results,
        "batch_results": results,
    }


def format_command_output(result: str, max_lines: int = 50) -> str:
    """格式化命令输出"""
    lines = result.split('\n')
    if len(lines) > max_lines:
        truncated = '\n'.join(lines[:max_lines])
        return f"{truncated}\n... (共 {len(lines)} 行, 已截断)"
    return result


def extract_key_info(command: str, output: str) -> str:
    """从 Windows 命令输出中提取关键信息"""
    command_lower = command.lower()

    if "get-nettcpconnection" in command_lower or "netstat" in command_lower:
        if "listen" in output.lower():
            return "端口正在监听中"
        elif "established" in output.lower():
            return "端口存在活跃连接"
        elif not output.strip():
            return "端口未被使用"
        return format_command_output(output, max_lines=10)

    elif "get-process" in command_lower or "tasklist" in command_lower:
        lines = [l for l in output.split('\n') if l.strip()]
        if len(lines) > 1:
            return f"找到 {len(lines) - 1} 个相关进程"
        elif lines:
            return "找到相关进程"
        return "未找到相关进程"

    elif "get-psdrive" in command_lower or "get-volume" in command_lower:
        lines = output.split('\n')
        return "磁盘使用情况:\n" + '\n'.join(lines[:10])

    elif "win32_operatingsystem" in command_lower:
        return "已获取内存使用情况"

    elif "get-counter" in command_lower:
        return "已获取 CPU 使用率"

    elif "get-service" in command_lower:
        if "running" in output.lower():
            return "服务正在运行"
        elif "stopped" in output.lower():
            return "服务已停止"
        return format_command_output(output, max_lines=10)

    elif "docker ps" in command_lower:
        lines = output.split('\n')
        if len(lines) > 1:
            return f"当前运行 {len(lines) - 1} 个 Docker 容器"
        return "没有运行中的 Docker 容器"

    elif "get-eventlog" in command_lower:
        return "已获取系统错误日志"

    return format_command_output(output, max_lines=10)

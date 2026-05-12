"""CLI工具 - 让模型自主生成和执行 Windows PowerShell/CMD 命令"""

import subprocess
import re
import json
from typing import Tuple, Optional
from langchain_core.tools import tool
from loguru import logger
from dataclasses import dataclass


@dataclass
class CommandResult:
    """命令执行结果"""
    success: bool
    stdout: str
    stderr: str
    return_code: int
    command: str


# Windows 危险命令黑名单
DANGEROUS_PATTERNS = [
    (r"Remove-Item\s+.*-Recurse.*-Force", "递归强制删除文件"),
    (r"rd\s+/s\s+/q", "递归删除目录"),
    (r"del\s+/f\s+/s\s+/q", "强制删除文件"),
    (r"format\s+[A-Z]:", "格式化磁盘"),
    (r"diskpart", "磁盘分区操作"),
    (r"shutdown\s+/s", "系统关机"),
    (r"Stop-Computer", "PowerShell 关机命令"),
    (r"Restart-Computer", "系统重启"),
    (r"taskkill\s+/f", "强制结束进程"),
    (r"Stop-Process\s+.*-Force", "强制停止进程"),
    (r"sc\s+delete", "删除 Windows 服务"),
    (r"reg\s+delete", "删除注册表"),
]


def _check_command_dangerous_level(command: str) -> Tuple[bool, str, str]:
    """检查命令危险等级"""
    cmd_lower = command.lower()

    for pattern, warning in DANGEROUS_PATTERNS:
        if re.search(pattern, cmd_lower, re.IGNORECASE):
            return True, "dangerous", f"⚠️ 危险命令检测: {warning}"

    return False, "safe", ""


def _execute_shell_command(
        command: str,
        capture_output: bool = True,
        timeout: int = 30,
        working_dir: Optional[str] = None
) -> CommandResult:
    """
    Windows PowerShell 命令执行内核
    """

    try:
        logger.info(f"执行命令: {command}")

        # 自动补 PowerShell
        if not command.lower().startswith(("powershell", "cmd")):
            command = command.strip()

        result = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-Command",
                command
            ],
            shell=False,
            capture_output=capture_output,
            text=True,
            encoding="utf-8",
            errors="ignore",
            cwd=working_dir,
            timeout=timeout
        )

        stdout = (result.stdout or "").strip()
        stderr = (result.stderr or "").strip()

        cmd_result = CommandResult(
            success=result.returncode == 0,
            stdout=stdout,
            stderr=stderr,
            return_code=result.returncode,
            command=command
        )

        if cmd_result.success:
            logger.info(f"命令执行成功: {command[:100]}")
        else:
            logger.error(
                f"命令执行失败: {command[:100]} | "
                f"returncode={result.returncode} | "
                f"stderr={stderr[:300]}"
            )

        return cmd_result

    except subprocess.TimeoutExpired:
        logger.error(f"命令执行超时 ({timeout}s): {command}")

        return CommandResult(
            success=False,
            stdout="",
            stderr=f"命令执行超时（超过 {timeout} 秒）",
            return_code=-1,
            command=command
        )

    except Exception as e:
        logger.exception("命令执行异常")

        return CommandResult(
            success=False,
            stdout="",
            stderr=f"命令执行错误: {str(e)}",
            return_code=-1,
            command=command
        )


def _format_command_output(
        result: CommandResult,
        max_output_lines: int = 200
) -> str:
    """
    格式化命令输出
    """

    if result.success:

        output = result.stdout

        if not output:
            return "命令执行成功，但没有输出内容。"

        lines = output.splitlines()

        if len(lines) > max_output_lines:
            output = "\n".join(lines[:max_output_lines])
            output += f"\n... (输出已截断，共 {len(lines)} 行)"

        return output

    else:

        error_text = result.stderr if result.stderr else "未知错误"

        return (
            f"命令执行失败\n"
            f"返回码: {result.return_code}\n"
            f"错误信息:\n{error_text}"
        )


@tool
def execute_shell_command(command: str, timeout: int = 30) -> str:
    """
    执行 Windows PowerShell/CMD 命令

    重要：
    该工具只负责生成待确认命令。
    实际执行由 execute_confirmed_command 完成。

    Args:
        command: PowerShell/CMD 命令
        timeout: 超时时间

    Returns:
        str
    """

    try:
        logger.info(
            f"Shell命令工具调用: command='{command}', timeout={timeout}"
        )

        is_dangerous, danger_level, warning = (
            _check_command_dangerous_level(command)
        )

        payload = {
            "type": "confirm_required",
            "command": command,
            "timeout": timeout,
            "dangerous": is_dangerous,
            "warning": warning
        }

        return json.dumps(payload, ensure_ascii=False)

    except Exception as e:
        logger.exception("Shell命令工具调用失败")

        return json.dumps({
            "type": "error",
            "message": str(e)
        }, ensure_ascii=False)


def execute_confirmed_command(
        command: str,
        timeout: int = 30
) -> str:
    """
    执行已确认命令

    Args:
        command: 已确认命令
        timeout: 超时时间

    Returns:
        str
    """

    try:
        result = _execute_shell_command(
            command=command,
            timeout=timeout
        )

        formatted_output = _format_command_output(result)

        response = {
            "success": result.success,
            "command": command,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "return_code": result.return_code,
            "formatted_output": formatted_output
        }

        return json.dumps(
            response,
            ensure_ascii=False,
            indent=2
        )

    except Exception as e:
        logger.exception("执行确认命令失败")

        return json.dumps({
            "success": False,
            "command": command,
            "stdout": "",
            "stderr": str(e),
            "return_code": -1,
            "formatted_output": f"执行命令时发生错误: {str(e)}"
        }, ensure_ascii=False, indent=2)
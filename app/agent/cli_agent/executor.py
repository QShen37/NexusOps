"""
CLI Executor 节点：执行 Windows PowerShell/CMD 命令
支持执行前用户确认、危险命令警告、命令编辑等功能
"""

from typing import Dict, Any, Optional, Tuple
from loguru import logger

from app.tools.cli_tools import  execute_confirmed_command
from .state import PlanExecuteState


class CLIExecutor:
    """CLI 命令执行器 - 带用户确认机制"""

    def __init__(self):
        self.pending_command: Optional[Dict[str, Any]] = None
        self.waiting_confirmation: bool = False

    def _check_command_dangerous(self, command: str) -> Tuple[bool, str]:
        """
        检查 Windows 命令是否危险
        """
        dangerous_patterns = [
            (r"Remove-Item\s+.*-Recurse.*-Force", "⚠️ 极度危险：递归强制删除文件！"),
            (r"rd\s+/s\s+/q", "⚠️ 危险：将递归删除目录！"),
            (r"del\s+/f\s+/s\s+/q", "⚠️ 危险：将强制删除多个文件！"),
            (r"format\s+[A-Z]:", "⚠️ 极度危险：格式化磁盘操作！"),
            (r"diskpart", "⚠️ 危险：磁盘分区操作可能导致数据丢失！"),
            (r"shutdown\s+/s", "⚠️ 注意：系统即将关机！"),
            (r"Stop-Computer", "⚠️ 注意：PowerShell 关机命令！"),
            (r"Restart-Computer", "⚠️ 注意：系统即将重启！"),
            (r"taskkill\s+/f", "⚠️ 注意：强制终止进程！"),
            (r"Stop-Process\s+.*-Force", "⚠️ 注意：强制停止进程！"),
            (r"Set-ExecutionPolicy", "⚠️ 注意：正在修改 PowerShell 执行策略！"),
            (r"sc\s+delete", "⚠️ 危险：删除 Windows 服务！"),
            (r"reg\s+delete", "⚠️ 危险：删除注册表项！"),
        ]

        import re

        for pattern, warning in dangerous_patterns:
            if re.search(pattern, command, re.IGNORECASE):
                return True, warning

        warning_patterns = [
            (r"del\s+", "⚠️ 注意：删除操作会永久移除文件"),
            (r"move\s+", "⚠️ 注意：文件移动操作"),
            (r"copy\s+", "⚠️ 注意：文件复制操作"),
            (r"taskkill", "⚠️ 注意：终止进程操作"),
            (r"Stop-Service", "⚠️ 注意：停止服务操作"),
            (r"Restart-Service", "⚠️ 注意：重启服务操作"),
            (r">\s*\S+", "⚠️ 注意：重定向输出可能覆盖文件"),
        ]

        for pattern, warning in warning_patterns:
            if re.search(pattern, command, re.IGNORECASE):
                return False, warning

        return False, ""

    def _ask_user_confirmation(self, command: str, step_info: str = "") -> Tuple[bool, Optional[str]]:
        """
        询问用户确认

        Returns:
            (confirmed, edited_command)
        """
        print("\n" + "=" * 70)
        print("🔐 需要确认执行命令")
        print("=" * 70)

        if step_info:
            print(f"📋 任务说明: {step_info}")

        print(f"\n💻 将要执行的命令:")
        print(f"   PS> {command}")

        # 检查危险级别
        is_dangerous, warning = self._check_command_dangerous(command)

        if warning:
            print(f"\n{warning}")

            if is_dangerous:
                print("此操作可能造成不可逆的影响！")

        print("\n请选择操作:")
        print("   [Y] 执行命令")
        print("   [N] 跳过此命令")
        print("   [E] 编辑命令后执行")
        print("   [A] 执行所有剩余命令（不再询问）")
        print("   [Q] 退出整个任务")

        while True:
            choice = input("\n请输入选项 (Y/N/E/A/Q): ").strip().upper()

            if choice == 'Y':
                return True, None

            elif choice == 'N':
                print("⏭️  已跳过此命令")
                return False, None

            elif choice == 'E':
                print("\n📝 请输入修改后的命令:")
                new_command = input("PS> ").strip()

                if new_command:
                    print(f"✅ 命令已修改为: {new_command}")
                    return True, new_command
                else:
                    print("❌ 命令不能为空，请重新选择")
                    continue

            elif choice == 'A':
                print("✅ 已开启自动执行模式，后续命令将自动执行")
                return True, None

            elif choice == 'Q':
                print("❌ 用户选择退出任务")
                return None, None

            else:
                print("❌ 无效选项，请输入 Y, N, E, A 或 Q")

    async def execute_command_with_confirmation(
            self,
            command: str,
            step_index: int,
            total_steps: int,
            step_description: str = "",
            auto_mode: bool = False
    ) -> Tuple[bool, str, Optional[str]]:
        """
        执行命令（带确认）

        Returns:
            (executed, result, edited_command)
        """
        logger.info(f"准备执行命令 [{step_index}/{total_steps}]: {command[:100]}")

        # 自动补 PowerShell
        final_command = command

        if not final_command.lower().startswith(("powershell", "cmd")):
            final_command = f'powershell -Command "{final_command}"'

        # 如果不在自动模式，询问用户确认
        if not auto_mode:
            confirmed, edited_command = self._ask_user_confirmation(command, step_description)

            # 用户选择退出
            if confirmed is None:
                return False, "用户选择退出任务", None

            # 用户跳过
            if not confirmed:
                return False, "用户跳过此命令", None

            # 使用编辑后的命令
            if edited_command:
                final_command = edited_command

                if not final_command.lower().startswith(("powershell", "cmd")):
                    final_command = f'powershell -Command "{final_command}"'

        else:
            print(f"\n🤖 [PowerShell 自动模式] 执行命令:")
            print(f"   PS> {command}")

        # 执行命令
        print(f"\n🔧 正在执行命令...")

        try:
            raw_result = execute_confirmed_command(final_command, timeout=30)

            # 统一转换为字符串
            if raw_result is None:
                result = ""

            elif isinstance(raw_result, str):
                result = raw_result

            elif isinstance(raw_result, bytes):
                result = raw_result.decode("utf-8", errors="ignore")

            elif isinstance(raw_result, dict):
                import json
                result = json.dumps(raw_result, ensure_ascii=False, indent=2)

            else:
                result = str(raw_result)

            # 截断过长输出
            if len(result) > 500:
                display_result = result[:500] + "\n... (输出已截断)"
            else:
                display_result = result

            print(f"\n📊 执行结果:")
            print("-" * 50)
            print(display_result)
            print("-" * 50)

            logger.info("命令执行成功")

            return True, result, edited_command

        except Exception as e:
            error_msg = f"命令执行失败: {str(e)}"

            print(f"\n❌ {error_msg}")

            logger.error(error_msg)

            return True, error_msg, edited_command


async def cli_executor(state: PlanExecuteState) -> Dict[str, Any]:
    """
    CLI 执行节点：执行 Windows PowerShell/CMD 命令
    """
    logger.info("=== CLI Executor：执行命令 ===")

    # 获取当前计划
    plan = state.get("plan", [])
    auto_mode = state.get("auto_mode", False)

    # 如果计划为空
    if not plan:
        logger.info("计划为空，跳过执行")
        return {}

    # 获取当前命令
    current_command = plan[0]
    past_steps_count = len(state.get("past_steps", []))
    current_step_index = past_steps_count + 1
    total_steps = len(plan) + past_steps_count

    logger.info(f"执行命令 [{current_step_index}/{total_steps}]: {current_command[:100]}")

    # 创建执行器
    executor = CLIExecutor()

    # 获取步骤描述
    step_descriptions = state.get("step_descriptions", {})
    step_description = step_descriptions.get(current_step_index, "")

    # 执行命令
    executed, result, edited_command = await executor.execute_command_with_confirmation(
        command=current_command,
        step_index=current_step_index,
        total_steps=total_steps,
        step_description=step_description,
        auto_mode=auto_mode
    )

    # 用户退出
    if executed is False and result == "用户选择退出任务":
        return {
            "plan": [],
            "past_steps": [(current_command, "任务被用户终止")],  # 不要手动拼接
            "task_status": "terminated",
            "response": "用户终止了任务执行"
        }

    # 用户跳过
    if not executed:
        logger.info(f"用户跳过命令: {current_command}")

        return {
            "plan": plan[1:],
            "past_steps": [(current_command, f"⏭️ 用户跳过: {result}")],  # 不要手动拼接
            "auto_mode": auto_mode
        }

    # 命令执行完成
    logger.info(f"命令执行完成，结果长度: {len(result)}")

    # 构建执行记录
    execution_record = {
        "command": current_command,
        "edited_command": edited_command if edited_command else None,
        "result": result,
        "status": "success",
        "timestamp": None
    }

    # 关键修复：返回单个元组，让 operator.add 自动追加
    return {
        "plan": plan[1:],
        "past_steps": [(current_command, result)],  # 只返回新步骤，不要手动拼接
        "execution_records": state.get("execution_records", []) + [execution_record],
        "auto_mode": auto_mode
    }


async def cli_batch_executor(state: PlanExecuteState) -> Dict[str, Any]:
    """
    批量执行器：一次性执行多个 Windows 命令
    """
    logger.info("=== CLI Batch Executor：批量执行命令 ===")

    plan = state.get("plan", [])
    past_steps = state.get("past_steps", [])  # 只用于计数

    if not plan:
        logger.info("计划为空，跳过执行")
        return {}

    results = []

    for i, command in enumerate(plan, 1):
        logger.info(f"批量执行 [{i}/{len(plan)}]: {command[:100]}")

        try:
            final_command = command

            if not final_command.lower().startswith(("powershell", "cmd")):
                final_command = f'powershell -Command "{final_command}"'

            result = execute_confirmed_command(final_command, timeout=30)

            results.append((command, result))  # 收集所有结果

            logger.info(f"命令 {i} 执行成功")

        except Exception as e:
            error_msg = f"执行失败: {str(e)}"

            results.append((command, error_msg))

            logger.error(f"命令 {i} 执行失败: {e}")

    # 返回所有结果，让 operator.add 逐条追加
    return {
        "plan": [],
        "past_steps": results,  # 返回整个列表，但注意 operator.add 会追加
        "batch_results": results
    }


def format_command_output(result: str, max_lines: int = 50) -> str:
    """格式化命令输出"""

    lines = result.split('\n')

    if len(lines) > max_lines:
        truncated = '\n'.join(lines[:max_lines])
        return f"{truncated}\n... (共 {len(lines)} 行，已截断)"

    return result


def extract_key_info(command: str, output: str) -> str:
    """从 Windows 命令输出中提取关键信息"""

    command_lower = command.lower()

    # 端口检查
    if "get-nettcpconnection" in command_lower or "netstat" in command_lower:

        if "listen" in output.lower():
            return "端口正在监听中"

        elif "established" in output.lower():
            return "端口存在活跃连接"

        elif not output.strip():
            return "端口未被使用"

        else:
            return format_command_output(output, max_lines=10)

    # 进程检查
    elif "get-process" in command_lower or "tasklist" in command_lower:

        lines = [l for l in output.split('\n') if l.strip()]

        if len(lines) > 1:
            return f"找到 {len(lines) - 1} 个相关进程"

        elif lines:
            return "找到相关进程"

        else:
            return "未找到相关进程"

    # 磁盘检查
    elif "get-psdrive" in command_lower or "get-volume" in command_lower:

        lines = output.split('\n')

        return "磁盘使用情况:\n" + '\n'.join(lines[:10])

    # 内存检查
    elif "win32_operatingsystem" in command_lower:
        return "已获取内存使用情况"

    # CPU 检查
    elif "get-counter" in command_lower:
        return "已获取 CPU 使用率"

    # 服务检查
    elif "get-service" in command_lower:

        if "running" in output.lower():
            return "服务正在运行"

        elif "stopped" in output.lower():
            return "服务已停止"

        else:
            return format_command_output(output, max_lines=10)

    # Docker
    elif "docker ps" in command_lower:

        lines = output.split('\n')

        if len(lines) > 1:
            return f"当前运行 {len(lines) - 1} 个 Docker 容器"

        else:
            return "没有运行中的 Docker 容器"

    # 日志检查
    elif "get-eventlog" in command_lower:
        return "已获取系统错误日志"

    # 默认返回
    return format_command_output(output, max_lines=10)
"""CLI Service - 智能命令行服务，基于 Plan-Execute-Replan 架构"""

from typing import AsyncGenerator, Dict, Any, Optional
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from loguru import logger

from app.agent.cli_agent import PlanExecuteState
from app.agent.cli_agent import cli_planner
from app.agent.cli_agent import cli_executor
from app.agent.cli_agent import cli_replanner, should_terminate


# 节点名称常量
NODE_PLANNER = "cli_planner"
NODE_EXECUTOR = "cli_executor"
NODE_REPLANNER = "cli_replanner"

# 确认状态
CONFIRMATION_PENDING = "pending"
CONFIRMATION_APPROVED = "approved"
CONFIRMATION_REJECTED = "rejected"
CONFIRMATION_EDITED = "edited"
CONFIRMATION_AUTO = "auto"


class CLIService:
    """智能 CLI 服务 - 基于 Plan-Execute-Replan 架构"""

    def __init__(self):
        self.checkpointer = MemorySaver()
        self.graph = self._build_graph()
        self.pending_confirmations = {}  # 存储待确认的命令 {session_id: command_info}
        logger.info("CLI Service 初始化完成")

    # =========================
    # build graph
    # =========================
    def _build_graph(self):
        """构建状态图"""
        workflow = StateGraph(PlanExecuteState)

        # 添加节点
        workflow.add_node(NODE_PLANNER, cli_planner)
        workflow.add_node(NODE_EXECUTOR, cli_executor)
        workflow.add_node(NODE_REPLANNER, cli_replanner)

        # 设置入口点
        workflow.set_entry_point(NODE_PLANNER)

        # 添加边
        workflow.add_edge(NODE_PLANNER, NODE_EXECUTOR)
        workflow.add_edge(NODE_EXECUTOR, NODE_REPLANNER)

        # =========================
        # 条件边：决定继续还是结束
        # =========================
        def should_continue(state: PlanExecuteState):
            """判断是否继续执行"""
            state = state or {}

            # 如果已经有响应，结束流程
            if state.get("response"):
                logger.info("流程结束：已有最终响应")
                return END

            # 检查是否应该终止（连续失败等）
            if should_terminate(state):
                logger.info("流程结束：触发终止条件")
                return END

            # 获取剩余计划
            plan = state.get("plan") or []

            # 如果还有剩余命令，继续执行
            if len(plan) > 0:
                logger.info(f"继续执行剩余 {len(plan)} 个命令")
                return NODE_EXECUTOR

            # 没有剩余命令，结束流程
            logger.info("流程结束：无剩余命令")
            return END

        workflow.add_conditional_edges(
            NODE_REPLANNER,
            should_continue,
            {
                NODE_EXECUTOR: NODE_EXECUTOR,
                END: END
            }
        )

        return workflow.compile(checkpointer=self.checkpointer)

    # =========================
    # execute
    # =========================
    async def execute(
        self,
        user_input: str,
        session_id: str = "default",
        auto_mode: bool = False,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        执行 CLI 任务

        Args:
            user_input: 用户的自然语言输入
            session_id: 会话ID
            auto_mode: 是否自动执行模式（跳过确认）

        Yields:
            事件字典，包含类型、阶段和内容
        """
        logger.info(f"开始 CLI 任务: {user_input}, session_id={session_id}, auto_mode={auto_mode}")

        # 初始化状态
        initial_state: PlanExecuteState = {
            "input": user_input,
            "plan": [],
            "past_steps": [],
            "response": "",
            "replan_count": 0,  # 重新规划次数
            "auto_mode": auto_mode,  # 自动执行模式
            "execution_records": [],  # 执行记录
        }

        config = {"configurable": {"thread_id": session_id}}

        try:
            # 流式执行图
            async for event in self.graph.astream(
                input=initial_state,
                config=config,
                stream_mode="updates"
            ):
                # 跳过空事件
                if not event:
                    continue

                for node, state in event.items():
                    logger.info(f"进入节点: {node}")

                    # 确保 state 不为空
                    state = state or {}

                    # ========== Planner 节点 ==========
                    if node == NODE_PLANNER:
                        plan = state.get("plan", [])
                        yield {
                            "type": "plan",
                            "stage": "planner",
                            "plan": plan,
                            "command_count": len(plan),
                            "message": f"已生成 {len(plan)} 个命令" if plan else "未能生成命令"
                        }

                    # ========== Executor 节点 ==========
                    elif node == NODE_EXECUTOR:
                        past_steps = state.get("past_steps", [])
                        plan = state.get("plan", [])

                        # 获取最后执行的命令结果
                        last_step = past_steps[-1] if past_steps else None

                        if last_step:
                            command, result = last_step
                            yield {
                                "type": "execution",
                                "stage": "executor",
                                "command": command,
                                "result": result[:500] if len(result) > 500 else result,  # 截断过长结果
                                "result_full": result,  # 完整结果
                                "remaining_commands": len(plan),
                                "message": f"命令执行完成" if "失败" not in result else "命令执行失败"
                            }
                        else:
                            yield {
                                "type": "execution",
                                "stage": "executor",
                                "message": "执行器已启动"
                            }

                    # ========== Replanner 节点 ==========
                    elif node == NODE_REPLANNER:
                        response = state.get("response")
                        plan = state.get("plan", [])

                        # 如果有最终响应
                        if response:
                            yield {
                                "type": "report",
                                "stage": "replanner",
                                "report": response,
                                "message": "任务完成"
                            }
                        else:
                            # 重新规划中
                            replan_count = state.get("replan_count", 0)
                            yield {
                                "type": "replan",
                                "stage": "replanner",
                                "plan": plan,
                                "replan_count": replan_count,
                                "remaining_commands": len(plan),
                                "message": f"重新规划中（第 {replan_count} 次）" if replan_count > 0 else "继续执行"
                            }

            # ========== 获取最终状态 ==========
            final_state = self.graph.get_state(config)

            final_response = ""
            execution_summary = {}

            if final_state and final_state.values:
                final_response = final_state.values.get("response", "")
                execution_summary = {
                    "total_commands": len(final_state.values.get("past_steps", [])),
                    "success_count": sum(
                        1 for _, result in final_state.values.get("past_steps", [])
                        if "失败" not in result and "错误" not in result
                    ),
                    "remaining_count": len(final_state.values.get("plan", []))
                }

            yield {
                "type": "complete",
                "response": final_response,
                "summary": execution_summary,
                "message": "所有任务执行完成"
            }

        except Exception as e:
            logger.exception(f"CLI Service 执行失败: {e}")
            yield {
                "type": "error",
                "message": str(e),
                "error_type": type(e).__name__
            }

    # =========================
    # 带用户确认的执行
    # =========================
    async def execute_with_confirmation(
        self,
        user_input: str,
        session_id: str = "default",
        on_confirmation: Optional[callable] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        带用户确认的执行（交互式）

        Args:
            user_input: 用户的自然语言输入
            session_id: 会话ID
            on_confirmation: 确认回调函数，用于处理命令确认

        Yields:
            事件字典
        """
        logger.info(f"开始交互式 CLI 任务: {user_input}, session_id={session_id}")

        # 初始化状态（非自动模式）
        initial_state: PlanExecuteState = {
            "input": user_input,
            "plan": [],
            "past_steps": [],
            "response": "",
            "replan_count": 0,
            "auto_mode": False,
            "execution_records": [],
        }

        config = {"configurable": {"thread_id": session_id}}

        try:
            async for event in self.graph.astream(
                input=initial_state,
                config=config,
                stream_mode="updates"
            ):
                if not event:
                    continue

                for node, state in event.items():
                    state = state or {}

                    # Planner 节点
                    if node == NODE_PLANNER:
                        plan = state.get("plan", [])
                        yield {
                            "type": "plan",
                            "stage": "planner",
                            "plan": plan,
                            "command_count": len(plan),
                            "requires_confirmation": len(plan) > 0,
                            "message": f"请确认是否执行以下 {len(plan)} 个命令"
                        }

                        # 如果有待确认命令，存储到 pending_confirmations
                        if plan:
                            self.pending_confirmations[session_id] = {
                                "commands": plan,
                                "current_index": 0,
                                "status": CONFIRMATION_PENDING
                            }

                    # Executor 节点
                    elif node == NODE_EXECUTOR:
                        past_steps = state.get("past_steps", [])
                        plan = state.get("plan", [])

                        last_step = past_steps[-1] if past_steps else None

                        if last_step:
                            command, result = last_step
                            yield {
                                "type": "execution",
                                "stage": "executor",
                                "command": command,
                                "result": result[:500] if len(result) > 500 else result,
                                "result_full": result,
                                "remaining_commands": len(plan),
                            }

                    # Replanner 节点
                    elif node == NODE_REPLANNER:
                        response = state.get("response")

                        if response:
                            yield {
                                "type": "report",
                                "stage": "replanner",
                                "report": response,
                                "message": "任务完成"
                            }

                            # 清理待确认状态
                            if session_id in self.pending_confirmations:
                                del self.pending_confirmations[session_id]

            # 最终状态
            final_state = self.graph.get_state(config)
            final_response = ""
            if final_state and final_state.values:
                final_response = final_state.values.get("response", "")

            yield {
                "type": "complete",
                "response": final_response,
                "message": "任务完成"
            }

        except Exception as e:
            logger.exception(f"交互式 CLI 执行失败: {e}")
            yield {
                "type": "error",
                "message": str(e)
            }

    # =========================
    # 确认命令
    # =========================
    async def confirm_command(
        self,
        session_id: str,
        confirmed: bool,
        edited_command: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        确认或拒绝待执行的命令

        Args:
            session_id: 会话ID
            confirmed: 是否确认执行
            edited_command: 用户编辑后的命令（可选）

        Returns:
            确认结果
        """
        logger.info(f"用户确认命令: session_id={session_id}, confirmed={confirmed}, edited={edited_command is not None}")

        # 获取待确认的命令信息
        pending = self.pending_confirmations.get(session_id)

        if not pending:
            logger.warning(f"未找到待确认的命令: {session_id}")
            return {
                "success": False,
                "message": "没有待确认的命令"
            }

        if confirmed:
            # 获取当前命令
            current_index = pending.get("current_index", 0)
            commands = pending.get("commands", [])

            if current_index < len(commands):
                current_command = commands[current_index]

                # 更新状态
                if edited_command:
                    pending["status"] = CONFIRMATION_EDITED
                    pending["edited_command"] = edited_command
                else:
                    pending["status"] = CONFIRMATION_APPROVED

                # 更新索引
                pending["current_index"] = current_index + 1

                # 检查是否还有其他命令需要确认
                has_more = pending["current_index"] < len(commands)

                return {
                    "success": True,
                    "message": "命令已确认",
                    "command": edited_command or current_command,
                    "has_more": has_more,
                    "next_command": commands[pending["current_index"]] if has_more else None
                }
            else:
                # 所有命令都已确认
                del self.pending_confirmations[session_id]
                return {
                    "success": True,
                    "message": "所有命令已确认完成",
                    "all_confirmed": True
                }
        else:
            # 用户拒绝
            pending["status"] = CONFIRMATION_REJECTED
            del self.pending_confirmations[session_id]

            return {
                "success": True,
                "message": "命令已被拒绝",
                "rejected": True
            }

    # =========================
    # 获取服务状态
    # =========================
    def get_service_status(self) -> Dict[str, Any]:
        """获取服务状态"""
        return {
            "service": "CLI Service",
            "status": "running",
            "pending_confirmations": len(self.pending_confirmations),
            "sessions": list(self.pending_confirmations.keys())
        }

    # =========================
    # 清理会话
    # =========================
    def cleanup_session(self, session_id: str) -> bool:
        """清理指定会话"""
        if session_id in self.pending_confirmations:
            del self.pending_confirmations[session_id]
            logger.info(f"已清理会话: {session_id}")
            return True
        return False


# 全局实例
cli_service = CLIService()
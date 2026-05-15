"""CLI Service - 智能命令行服务

基于 Plan-Execute-Replan 架构, 通过事件队列把:
- 图节点 (planner / executor / replanner) 的状态更新
- executor 发起的「等待用户确认」请求

合并成一个 SSE 流, 让前端能够实时接收命令计划、执行结果, 并对每条命令进行
确认 / 跳过 / 编辑 / 切换自动模式 / 退出。
"""

import asyncio
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

# 确认 action 集合
ACTION_EXECUTE = "execute"
ACTION_SKIP = "skip"
ACTION_EDIT = "edit"
ACTION_AUTO = "auto"
ACTION_QUIT = "quit"
VALID_ACTIONS = {ACTION_EXECUTE, ACTION_SKIP, ACTION_EDIT, ACTION_AUTO, ACTION_QUIT}

# 等待前端确认的超时时间(秒)
CONFIRMATION_TIMEOUT = 600


class CLIService:
    """智能 CLI 服务 - 基于 Plan-Execute-Replan 架构"""

    def __init__(self):
        self.checkpointer = MemorySaver()
        self.graph = self._build_graph()
        # 每个 session 的事件队列(同时接收 graph 事件 + executor 的确认请求)
        self._event_queues: Dict[str, asyncio.Queue] = {}
        # 每个 session 等待前端响应的 future
        self._confirmation_futures: Dict[str, asyncio.Future] = {}
        logger.info("CLI Service 初始化完成")

    # =========================
    # build graph
    # =========================
    def _build_graph(self):
        """构建状态图"""
        workflow = StateGraph(PlanExecuteState)

        workflow.add_node(NODE_PLANNER, cli_planner)
        workflow.add_node(NODE_EXECUTOR, cli_executor)
        workflow.add_node(NODE_REPLANNER, cli_replanner)

        workflow.set_entry_point(NODE_PLANNER)
        workflow.add_edge(NODE_PLANNER, NODE_EXECUTOR)
        workflow.add_edge(NODE_EXECUTOR, NODE_REPLANNER)

        def should_continue(state: PlanExecuteState):
            state = state or {}
            if state.get("response"):
                logger.info("流程结束: 已有最终响应")
                return END
            if should_terminate(state):
                logger.info("流程结束: 触发终止条件")
                return END
            plan = state.get("plan") or []
            if len(plan) > 0:
                logger.info(f"继续执行剩余 {len(plan)} 个命令")
                return NODE_EXECUTOR
            logger.info("流程结束: 无剩余命令")
            return END

        workflow.add_conditional_edges(
            NODE_REPLANNER,
            should_continue,
            {NODE_EXECUTOR: NODE_EXECUTOR, END: END},
        )

        return workflow.compile(checkpointer=self.checkpointer)

    # =========================
    # 命令确认: 前端 <-> executor 之间的异步桥
    # =========================
    async def request_confirmation(
        self,
        session_id: str,
        command: str,
        step_index: int,
        total_steps: int,
        step_description: str = "",
        dangerous: bool = False,
        warning: str = "",
        matched_skill: Optional[str] = None,
        skill_display_name: Optional[str] = None,
        skill_risk_level: Optional[str] = None,
    ) -> Dict[str, Any]:
        """executor 调用: 把命令推送给前端, 阻塞等待用户响应

        返回的 dict 字段:
        - action: execute / skip / edit / auto / quit
        - edited_command: 若 action 是 edit, 则为用户编辑后的命令
        - auto_mode: 若用户选择 auto, 则为 True
        """
        loop = asyncio.get_event_loop()
        future: asyncio.Future = loop.create_future()
        self._confirmation_futures[session_id] = future

        # 推送 confirm_required 事件给前端
        queue = self._event_queues.get(session_id)
        confirm_event = {
            "type": "confirm_required",
            "stage": "executor",
            "session_id": session_id,
            "command": command,
            "step_index": step_index,
            "total_steps": total_steps,
            "step_description": step_description,
            "dangerous": dangerous,
            "warning": warning,
            "matched_skill": matched_skill,
            "skill_display_name": skill_display_name,
            "skill_risk_level": skill_risk_level,
            "message": "请在前端确认是否执行该命令",
        }
        if queue is not None:
            await queue.put(confirm_event)
        else:
            logger.warning(f"会话 {session_id} 没有事件队列, confirm_required 事件无法推送")

        # 等前端响应
        try:
            result = await asyncio.wait_for(future, timeout=CONFIRMATION_TIMEOUT)
            return result
        except asyncio.TimeoutError:
            logger.warning(f"会话 {session_id} 命令确认超时, 默认跳过")
            return {"action": ACTION_SKIP, "edited_command": None, "auto_mode": False}
        finally:
            self._confirmation_futures.pop(session_id, None)

    async def resolve_confirmation(
        self,
        session_id: str,
        action: str,
        edited_command: Optional[str] = None,
    ) -> Dict[str, Any]:
        """前端调用 (/cli/confirm): 设置确认结果, 唤醒 executor

        action: execute / skip / edit / auto / quit (也兼容 confirmed 布尔值)
        """
        if action not in VALID_ACTIONS:
            return {"success": False, "message": f"非法 action: {action}"}

        future = self._confirmation_futures.get(session_id)
        if future is None:
            return {"success": False, "message": "当前会话没有等待确认的命令"}
        if future.done():
            return {"success": False, "message": "该命令已被处理"}

        auto_mode_after = action == ACTION_AUTO
        normalized_action = ACTION_EXECUTE if action == ACTION_AUTO else action

        # edit 必须带 edited_command
        if normalized_action == ACTION_EDIT and not edited_command:
            return {"success": False, "message": "edit 操作必须提供 edited_command"}

        payload = {
            "action": normalized_action,
            "edited_command": edited_command if normalized_action == ACTION_EDIT else None,
            "auto_mode": auto_mode_after,
        }
        future.set_result(payload)
        logger.info(
            f"会话 {session_id} 收到前端确认: action={action}, "
            f"edited={edited_command is not None}, auto_mode_after={auto_mode_after}"
        )

        return {
            "success": True,
            "message": "确认已提交",
            "action": normalized_action,
            "auto_mode": auto_mode_after,
        }

    # =========================
    # execute (主流式接口)
    # =========================
    async def execute(
        self,
        user_input: str,
        session_id: str = "default",
        auto_mode: bool = False,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        执行 CLI 任务

        把 graph.astream 的节点事件 + executor 的 confirm_required 请求合并到
        一个事件队列里, 按到达顺序 yield 给前端 SSE。
        """
        logger.info(
            f"开始 CLI 任务: {user_input}, session_id={session_id}, auto_mode={auto_mode}"
        )

        # 初始化每个 session 的事件队列
        queue: asyncio.Queue = asyncio.Queue()
        self._event_queues[session_id] = queue

        initial_state: PlanExecuteState = {
            "input": user_input,
            "plan": [],
            "past_steps": [],
            "response": "",
            "replan_count": 0,
            "auto_mode": auto_mode,
            "execution_records": [],
            "session_id": session_id,
            "step_descriptions": {},
        }

        config = {"configurable": {"thread_id": session_id}}

        async def run_graph():
            """后台任务: 跑 graph.astream, 把节点事件 push 到 queue"""
            try:
                async for event in self.graph.astream(
                    input=initial_state,
                    config=config,
                    stream_mode="updates",
                ):
                    if not event:
                        continue
                    for node, state in event.items():
                        logger.info(f"进入节点: {node}")
                        state = state or {}

                        if node == NODE_PLANNER:
                            plan = state.get("plan", [])
                            await queue.put({
                                "type": "plan",
                                "stage": "planner",
                                "plan": plan,
                                "command_count": len(plan),
                                "matched_skill": state.get("matched_skill"),
                                "skill_display_name": state.get("skill_display_name"),
                                "skill_risk_level": state.get("skill_risk_level"),
                                "message": (
                                    f"已生成 {len(plan)} 个命令"
                                    if plan else "未能生成命令"
                                ),
                            })

                        elif node == NODE_EXECUTOR:
                            past_steps = state.get("past_steps", [])
                            plan = state.get("plan", [])
                            last_step = past_steps[-1] if past_steps else None

                            if last_step:
                                command, result = last_step
                                result_str = result if isinstance(result, str) else str(result)
                                await queue.put({
                                    "type": "execution",
                                    "stage": "executor",
                                    "command": command,
                                    "result": result_str[:500] if len(result_str) > 500 else result_str,
                                    "result_full": result_str,
                                    "remaining_commands": len(plan),
                                    "message": (
                                        "命令执行完成"
                                        if "失败" not in result_str else "命令执行失败"
                                    ),
                                })
                            else:
                                await queue.put({
                                    "type": "execution",
                                    "stage": "executor",
                                    "message": "执行器已启动",
                                })

                        elif node == NODE_REPLANNER:
                            response = state.get("response")
                            plan = state.get("plan", [])

                            if response:
                                await queue.put({
                                    "type": "report",
                                    "stage": "replanner",
                                    "report": response,
                                    "message": "任务完成",
                                })
                            else:
                                replan_count = state.get("replan_count", 0)
                                await queue.put({
                                    "type": "replan",
                                    "stage": "replanner",
                                    "plan": plan,
                                    "replan_count": replan_count,
                                    "remaining_commands": len(plan),
                                    "message": (
                                        f"重新规划中(第 {replan_count} 次)"
                                        if replan_count > 0 else "继续执行"
                                    ),
                                })

                # 取最终状态
                final_state = self.graph.get_state(config)
                final_response = ""
                execution_summary = {}
                if final_state and final_state.values:
                    final_response = final_state.values.get("response", "")
                    past_steps = final_state.values.get("past_steps", [])
                    execution_summary = {
                        "total_commands": len(past_steps),
                        "success_count": sum(
                            1 for _, result in past_steps
                            if "失败" not in result and "错误" not in result
                        ),
                        "remaining_count": len(final_state.values.get("plan", [])),
                    }

                await queue.put({
                    "type": "complete",
                    "response": final_response,
                    "summary": execution_summary,
                    "message": "所有任务执行完成",
                })

            except Exception as e:
                logger.exception(f"CLI Service 图执行失败: {e}")
                await queue.put({
                    "type": "error",
                    "message": str(e),
                    "error_type": type(e).__name__,
                })
            finally:
                # 通知主协程结束
                await queue.put(None)

        task = asyncio.create_task(run_graph())

        try:
            while True:
                event = await queue.get()
                if event is None:
                    break
                yield event
                if event.get("type") in ("complete", "error"):
                    break
        finally:
            # 收尾顺序: 先唤醒挂起的 confirmation future, 让 executor 能正常返回,
            # 再 cancel graph 任务, 避免遗留协程
            pending_future = self._confirmation_futures.pop(session_id, None)
            if pending_future is not None and not pending_future.done():
                pending_future.set_result({
                    "action": ACTION_SKIP,
                    "edited_command": None,
                    "auto_mode": False,
                })

            if not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass

            self._event_queues.pop(session_id, None)
            logger.info(f"[会话 {session_id}] 资源已清理")

    # =========================
    # 服务状态 / 清理
    # =========================
    def get_service_status(self) -> Dict[str, Any]:
        return {
            "service": "CLI Service",
            "status": "running",
            "active_sessions": list(self._event_queues.keys()),
            "pending_confirmations": list(self._confirmation_futures.keys()),
        }

    def cleanup_session(self, session_id: str) -> bool:
        cleaned = False
        future = self._confirmation_futures.pop(session_id, None)
        if future is not None:
            if not future.done():
                future.set_result({
                    "action": ACTION_QUIT,
                    "edited_command": None,
                    "auto_mode": False,
                })
            cleaned = True

        if session_id in self._event_queues:
            self._event_queues.pop(session_id, None)
            cleaned = True

        if cleaned:
            logger.info(f"已清理会话: {session_id}")
        return cleaned


# 全局实例
cli_service = CLIService()

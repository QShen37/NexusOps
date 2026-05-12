from typing import AsyncGenerator, Dict, Any
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from loguru import logger

from app.agent.aiops import PlanExecuteState, planner, executor, replanner


NODE_PLANNER = "planner"
NODE_EXECUTOR = "executor"
NODE_REPLANNER = "replanner"


class AIOpsService:

    def __init__(self):
        self.checkpointer = MemorySaver()
        self.graph = self._build_graph()
        logger.info("Plan-Execute-Replan Service 初始化完成")

    # =========================
    # build graph
    # =========================
    def _build_graph(self):

        workflow = StateGraph(PlanExecuteState)

        workflow.add_node(NODE_PLANNER, planner)
        workflow.add_node(NODE_EXECUTOR, executor)
        workflow.add_node(NODE_REPLANNER, replanner)

        workflow.set_entry_point(NODE_PLANNER)

        workflow.add_edge(NODE_PLANNER, NODE_EXECUTOR)
        workflow.add_edge(NODE_EXECUTOR, NODE_REPLANNER)

        # =========================
        # FIX: safe routing
        # =========================
        def should_continue(state: PlanExecuteState):

            state = state or {}

            if state.get("response"):
                logger.info("流程结束：response exists")
                return END

            plan = state.get("plan") or []

            if len(plan) > 0:
                logger.info(f"继续执行剩余 {len(plan)} steps")
                return NODE_EXECUTOR

            logger.info("plan 为空，结束流程")
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
    ) -> AsyncGenerator[Dict[str, Any], None]:

        logger.info(f"开始任务: {user_input}")

        initial_state: PlanExecuteState = {
            "input": user_input,
            "plan": [],
            "past_steps": [],
            "response": ""
        }

        config = {"configurable": {"thread_id": session_id}}

        try:

            async for event in self.graph.astream(
                input=initial_state,
                config=config,
                stream_mode="updates"
            ):

                # =========================
                # FIX 1: event 可能为空
                # =========================
                if not event:
                    continue

                for node, state in event.items():

                    logger.info(f"node={node}")

                    # =========================
                    # FIX 2: state safety
                    # =========================
                    state = state or {}

                    if node == NODE_PLANNER:
                        yield {
                            "type": "plan",
                            "stage": "planner",
                            "plan": state.get("plan", [])
                        }

                    elif node == NODE_EXECUTOR:
                        yield {
                            "type": "executor",
                            "stage": "executor",
                            "state": state
                        }

                    elif node == NODE_REPLANNER:

                        response = state.get("response")

                        if response:
                            yield {
                                "type": "report",
                                "stage": "replanner",
                                "report": response
                            }
                        else:
                            yield {
                                "type": "replan",
                                "stage": "replanner",
                                "plan": state.get("plan", [])
                            }

            # =========================
            # FIX 3: final state safety
            # =========================
            final_state = self.graph.get_state(config)

            final_response = ""
            if final_state and final_state.values:
                final_response = final_state.values.get("response") or ""

            yield {
                "type": "complete",
                "response": final_response
            }

        except Exception as e:
            logger.exception("AIOps execute failed")
            yield {
                "type": "error",
                "message": str(e)
            }


aiops_service = AIOpsService()
import json
from fastapi import APIRouter
from sse_starlette.sse import EventSourceResponse
from loguru import logger

from app.models.aiops import AIOpsRequest
from app.services.aiops_service import aiops_service

router = APIRouter()

DEFAULT_AIOPS_USER_INPUT = """
请检查当前系统 CPU、内存、磁盘使用情况，
如果存在异常进程，请分析原因，
并给出修复建议。
""".strip()


@router.post("/aiops")
async def diagnose_stream(request: AIOpsRequest):
    session_id = request.session_id or "default"
    user_input = (request.user_input or "").strip() or DEFAULT_AIOPS_USER_INPUT
    logger.info(f"[会话 {session_id}] 收到 AIOps 诊断请求（流式）")

    async def event_generator():
        try:
            async for event in aiops_service.execute(
                user_input=user_input,
                session_id=session_id,
            ):
                # 发送事件
                yield {
                    "event": "message",
                    "data": json.dumps(event, ensure_ascii=False)
                }

                # 如果是完成或错误事件，结束流
                if event.get("type") in ["complete", "error"]:
                    break
            logger.info(f"[会话 {session_id}] AIOps 诊断流式响应完成")
        except Exception as e:
            logger.error(f"[会话 {session_id}] AIOps 诊断流式响应异常: {e}", exc_info=True)
            yield {
                "event": "message",
                "data": json.dumps({
                    "type": "error",
                    "stage": "exception",
                    "message": f"诊断异常: {str(e)}"
                }, ensure_ascii=False)
            }

    return EventSourceResponse(event_generator())


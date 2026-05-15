import json
from typing import Optional
from fastapi import APIRouter
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse
from loguru import logger

from app.models.aiops import AIOpsRequest
from app.services.cli_service import cli_service

router = APIRouter()

DEFAULT_AIOPS_USER_INPUT = """
请检查当前系统 CPU、内存、磁盘使用情况，
如果存在异常进程，请分析原因，
并给出修复建议。
""".strip()

DEFAULT_CLI_USER_INPUT = """
帮我查看8004端口是否正在使用
""".strip()


class CLIConfirmRequest(BaseModel):
    """前端命令确认请求"""
    session_id: str = Field(..., description="会话 ID, 必须与 SSE 流的 session_id 一致")
    action: str = Field(
        ...,
        description="确认动作: execute(执行) / skip(跳过) / edit(编辑后执行) / auto(执行并切换到自动模式) / quit(终止任务)",
    )
    edited_command: Optional[str] = Field(
        default=None,
        description="action=edit 时, 用户编辑后的命令",
    )


@router.post("/cli")
async def cli_execute(request: AIOpsRequest):
    """
    CLI 命令执行接口（流式响应）

    功能：将自然语言转换为 shell 命令并执行

    请求体示例:
    {
        "user_input": "帮我查看8004端口是否正在使用",
        "session_id": "user123",
        "auto_mode": false
    }

    响应事件类型:
    - plan: 命令计划生成
    - execution: 命令执行结果
    - confirm_required: 等待前端确认 (非 auto_mode 时由 executor 推送)
    - replan: 重新规划
    - report: 最终报告
    - complete: 完成
    - error: 错误
    """
    session_id = request.session_id or "default"
    user_input = (request.user_input or "").strip() or DEFAULT_CLI_USER_INPUT
    auto_mode = getattr(request, 'auto_mode', False)

    logger.info(
        f"[会话 {session_id}] 收到 CLI 请求: {user_input}, auto_mode={auto_mode}"
    )

    async def event_generator():
        try:
            async for event in cli_service.execute(
                user_input=user_input,
                session_id=session_id,
                auto_mode=auto_mode,
            ):
                yield {
                    "event": "message",
                    "data": json.dumps(event, ensure_ascii=False),
                }
                if event.get("type") in ("complete", "error"):
                    break

            logger.info(f"[会话 {session_id}] CLI 执行完成")

        except Exception as e:
            logger.error(f"[会话 {session_id}] CLI 执行异常: {e}", exc_info=True)
            yield {
                "event": "message",
                "data": json.dumps({
                    "type": "error",
                    "stage": "exception",
                    "message": f"CLI执行异常: {str(e)}",
                }, ensure_ascii=False),
            }

    return EventSourceResponse(event_generator())


@router.post("/cli/confirm")
async def cli_confirm(request: CLIConfirmRequest):
    """
    确认 CLI 命令执行(交互式模式由前端调用)

    Body:
    - session_id: 会话 ID
    - action: execute / skip / edit / auto / quit
    - edited_command: action=edit 时必填
    """
    logger.info(
        f"收到命令确认: session_id={request.session_id}, action={request.action}, "
        f"edited={request.edited_command is not None}"
    )

    result = await cli_service.resolve_confirmation(
        session_id=request.session_id,
        action=request.action,
        edited_command=request.edited_command,
    )

    return result


@router.get("/cli/status")
async def cli_status():
    """获取 CLI 服务状态"""
    return cli_service.get_service_status()


@router.delete("/cli/session/{session_id}")
async def cli_cleanup(session_id: str):
    """清理 CLI 会话"""
    success = cli_service.cleanup_session(session_id)
    return {"success": success, "session_id": session_id}

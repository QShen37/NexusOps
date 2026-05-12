import json
from fastapi import APIRouter
from sse_starlette.sse import EventSourceResponse
from loguru import logger

from app.models.aiops import AIOpsRequest
from app.services.cli_service import cli_service  # 导入 CLI 服务

router = APIRouter()

DEFAULT_AIOPS_USER_INPUT = """
请检查当前系统 CPU、内存、磁盘使用情况，
如果存在异常进程，请分析原因，
并给出修复建议。
""".strip()

DEFAULT_CLI_USER_INPUT = """
帮我查看8004端口是否正在使用
""".strip()

@router.post("/cli")
async def cli_execute(request: AIOpsRequest):
    """
    CLI 命令执行接口（流式响应）

    功能：将自然语言转换为 shell 命令并执行

    请求体示例:
    {
        "user_input": "帮我查看8004端口是否正在使用",
        "session_id": "user123",
        "auto_mode": false  # 可选，是否自动执行
    }

    响应事件类型:
    - plan: 命令计划生成
    - execution: 命令执行结果
    - report: 最终报告
    - complete: 完成
    - error: 错误
    """
    session_id = request.session_id or "default"
    user_input = (request.user_input or "").strip() or DEFAULT_CLI_USER_INPUT
    auto_mode = getattr(request, 'auto_mode', False)  # 是否自动执行模式

    logger.info(f"[会话 {session_id}] 收到 CLI 请求: {user_input}, auto_mode={auto_mode}")

    async def event_generator():
        """CLI 事件生成器"""
        try:
            # 遍历 CLI 服务的执行事件
            async for event in cli_service.execute(
                    user_input=user_input,
                    session_id=session_id,
                    auto_mode=auto_mode
            ):
                # 发送事件
                yield {
                    "event": "message",
                    "data": json.dumps(event, ensure_ascii=False)
                }

                # 如果是完成或错误事件，结束流
                if event.get("type") in ["complete", "error"]:
                    break

            logger.info(f"[会话 {session_id}] CLI 执行完成")

        except Exception as e:
            logger.error(f"[会话 {session_id}] CLI 执行异常: {e}", exc_info=True)
            yield {
                "event": "message",
                "data": json.dumps({
                    "type": "error",
                    "stage": "exception",
                    "message": f"CLI执行异常: {str(e)}"
                }, ensure_ascii=False)
            }

    return EventSourceResponse(event_generator())


@router.post("/cli/confirm")
async def cli_confirm(session_id: str, confirmed: bool, edited_command: str = None):
    """
    确认 CLI 命令执行（用于交互式模式）

    参数:
    - session_id: 会话ID
    - confirmed: 是否确认执行
    - edited_command: 编辑后的命令（可选）
    """
    logger.info(f"收到命令确认: session_id={session_id}, confirmed={confirmed}")

    result = await cli_service.confirm_command(
        session_id=session_id,
        confirmed=confirmed,
        edited_command=edited_command
    )

    return result


@router.get("/cli/status")
async def cli_status():
    """
    获取 CLI 服务状态
    """
    return cli_service.get_service_status()


@router.delete("/cli/session/{session_id}")
async def cli_cleanup(session_id: str):
    """
    清理 CLI 会话
    """
    success = cli_service.cleanup_session(session_id)
    return {"success": success, "session_id": session_id}
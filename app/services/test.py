import asyncio

from rag_agent_service import rag_agent_service


async def main():

    session_id = "test_001"

    print("======== 非流式测试 ========")

    result = await rag_agent_service.query(
        question="现在几点了？",
        session_id=session_id
    )

    print(result)

    print("\n======== 流式测试 ========")

    async for chunk in rag_agent_service.query_stream(
            question="帮我介绍一下Python",
            session_id=session_id
    ):
        if chunk.get("type") == "content":
            print(chunk["data"], end="", flush=True)

    print("\n======== 历史记录 ========")

    history = rag_agent_service.get_session_history(session_id)

    for msg in history:
        print(msg)

    print("\n======== 清除会话 ========")

    success = rag_agent_service.clear_session(session_id)
    print(success)


if __name__ == "__main__":
    asyncio.run(main())
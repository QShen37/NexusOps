"""
AIOps Plan-Execute-Replan 完整流程测试
用于观察 LangGraph Agent 推理链路
"""

import asyncio
import json
from pprint import pprint

from app.services.aiops_service import aiops_service


def pretty_print(title, data):
    print("\n" + "=" * 100)
    print(f"🧠 {title}")
    print("=" * 100)
    print(json.dumps(data, ensure_ascii=False, indent=2))


async def test_plan_execute_replan():
    user_input = """
    请帮我检查系统的运行状况：

    1. 查看 CPU 使用情况（使用 top 或 mpstat 命令）
    2. 查看内存使用情况（使用 free 命令）
    3. 查看磁盘使用情况（使用 df 命令）
    4. 找出占用 CPU 或内存最高的前 5 个进程（使用 ps 命令）
    5. 检查系统负载（使用 uptime 命令）

    如果发现异常（如 CPU 使用率超过 80%、内存不足、磁盘使用率超过 90%、或者有异常的进程），请分析可能的原因并给出修复建议。
    """

    print("\n🚀 启动 AIOps Plan-Execute-Replan Agent")
    print(f"📝 Input: {user_input}")

    step_count = 0

    async for event in aiops_service.execute(
        user_input=user_input,
        session_id="debug-session-001"
    ):

        step_count += 1

        print("\n" + "-" * 100)
        print(f"📌 STEP {step_count}")
        print(f"📌 TYPE  : {event.get('type')}")
        print(f"📌 STAGE : {event.get('stage')}")
        print("-" * 100)

        # =========================
        # Planner 输出
        # =========================
        if event.get("type") == "plan":

            print("\n📍 [PLANNER OUTPUT]")

            plan = event.get("plan", [])

            print(f"📋 生成计划（{len(plan)} steps）:")

            for i, p in enumerate(plan):
                print(f"  {i+1}. {p}")

        # =========================
        # Executor 输出
        # =========================
        elif event.get("type") == "executor":

            print("\n⚙️ [EXECUTOR OUTPUT]")

            state = event.get("state", {})

            print("📊 当前状态：")

            pretty_print("EXECUTOR STATE", state)

            plan = state.get("plan", [])
            past = state.get("past_steps", [])

            print(f"\n🔄 已完成步骤: {len(past)}")
            print(f"⏳ 剩余步骤: {len(plan)}")

            if past:
                print("\n✔️ 最新执行步骤:")
                print(f"   {past[-1]}")

        # =========================
        # Replanner 输出
        # =========================
        elif event.get("type") == "replan":

            print("\n🔁 [REPLANNER OUTPUT]")

            plan = event.get("plan", [])

            print(f"📋 当前剩余计划: {len(plan)} steps")

            for i, p in enumerate(plan):
                print(f"  {i+1}. {p}")

        elif event.get("type") == "report":

            print("\n📄 [FINAL REPORT]")

            print(event.get("report"))

        # =========================
        # 完成
        # =========================
        elif event.get("type") == "complete":

            print("\n🎉 [COMPLETE]")

            print("最终输出：")
            print(event.get("response"))

        # =========================
        # error
        # =========================
        elif event.get("type") == "error":

            print("\n❌ [ERROR]")
            print(event.get("message"))

        else:
            print("\n📦 RAW EVENT:")
            pprint(event)

    print("\n" + "=" * 100)
    print("✅ 测试结束")
    print("=" * 100)


if __name__ == "__main__":
    asyncio.run(test_plan_execute_replan())
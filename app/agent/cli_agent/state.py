"""
CLI Plan-Execute-Replan 状态定义

支持:
- 自动 / 交互执行模式
- 通过 session_id 与前端进行命令确认交互
- 共享 Skill 上下文 (与 aiops agent 保持一致)
"""

from typing import List, TypedDict, Annotated, Optional, Dict, Any
import operator


class PlanExecuteState(TypedDict, total=False):
    """CLI Plan-Execute-Replan 状态

    Skill 相关字段由 planner 写入, executor / replanner 读取,
    用于在前端确认与 LLM 决策时共享同一份 Playbook / 允许工具列表 / 风险等级。
    """

    # 用户输入(任务描述)
    input: str

    # 执行计划(shell 命令列表)
    plan: List[str]

    # 已执行的步骤历史
    # 使用 operator.add 实现追加式更新(而非覆盖)
    past_steps: Annotated[List[tuple], operator.add]

    # 最终响应/报告
    response: str

    # ===== 执行控制 =====
    # 会话 ID, 用于和前端进行命令确认交互
    session_id: str
    # 自动执行模式(跳过用户确认)
    auto_mode: bool
    # 重新规划次数
    replan_count: int
    # 任务状态(running / terminated / completed 等)
    task_status: str
    # 执行记录(完整命令日志)
    execution_records: List[Dict[str, Any]]
    # 步骤说明 {step_index: description}
    step_descriptions: Dict[int, str]

    # ===== Skills 共享上下文 =====
    # 匹配到的 Skill 名称 (如 container_diagnosis)
    matched_skill: Optional[str]
    # planner 推荐的 Skill 名称 (可与 matched_skill 一致或不同)
    suggested_skill: Optional[str]
    # 该 Skill 的 Playbook 完整内容 (markdown)
    skill_playbook: Optional[str]
    # 该 Skill 的展示名 (如 "Docker 容器诊断")
    skill_display_name: Optional[str]
    # 该 Skill 的描述
    skill_description: Optional[str]
    # 该 Skill 推荐使用的工具白名单
    skill_allowed_tools: Optional[List[str]]
    # 该 Skill 的风险等级 (low / medium / high)
    skill_risk_level: Optional[str]

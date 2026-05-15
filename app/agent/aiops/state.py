"""
通用 Plan-Execute-Replan 状态定义
基于 LangGraph 官方教程实现
"""

from typing import List, TypedDict, Annotated, Optional
import operator


class PlanExecuteState(TypedDict, total=False):
    """Plan-Execute-Replan 状态

    skill 相关字段由 planner 节点写入, 由 executor / replanner 读取,
    使三个节点共享同一份 skill 上下文 (playbook / 允许工具 / 风险等级)
    """

    # 用户输入（任务描述）
    input: str

    # 执行计划（步骤列表）
    plan: List[str]

    # 已执行的步骤历史
    # 使用 operator.add 实现追加式更新（而非覆盖）
    past_steps: Annotated[List[tuple], operator.add]

    # 最终响应/报告
    response: str

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

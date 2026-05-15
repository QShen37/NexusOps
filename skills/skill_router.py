from typing import Dict, Any, Optional
from .skill_registry import SkillRegistry, SkillMetadata


class SkillRouter:
    """Skill 路由器 - 纯 Markdown 驱动"""

    def __init__(self, registry: SkillRegistry = None):
        self.registry = registry or SkillRegistry()

    async def route(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        路由查询到合适的 skill
        返回诊断结果（让 LLM 根据 markdown 执行）
        """
        if context is None:
            context = {}

        # 1. 匹配最合适的 skill
        matched_skill = self.registry.match_skill(query)

        if not matched_skill:
            return {
                "error": "未找到合适的诊断 skill",
                "query": query,
                "available_skills": [s.name for s in self.registry.get_all_skills()]
            }

        # 2. 构建诊断 prompt
        prompt = self._build_diagnosis_prompt(matched_skill, query, context)

        # 3. 返回诊断指令（实际使用时调用 LLM）
        return {
            "skill": matched_skill.name,
            "display_name": matched_skill.display_name,
            "description": matched_skill.description,
            "allowed_tools": matched_skill.allowed_tools,
            "prompt": prompt,
            "query": query,
            "context": context
        }

    def _build_diagnosis_prompt(self, skill: SkillMetadata, query: str, context: Dict) -> str:
        """构建让 LLM 执行的诊断 prompt"""

        tools_section = ""
        if skill.allowed_tools:
            tools_section = f"""
## 可用工具
你可以使用以下工具（调用时需要真实执行）：
{chr(10).join(f'- {tool}' for tool in skill.allowed_tools)}
"""

        context_section = ""
        if context:
            context_section = f"""
## 上下文信息
{context}
"""

        prompt = f"""你是一个专业的运维诊断专家，请根据以下 Playbook 诊断用户问题。

## Skill 信息
- 名称: {skill.display_name}
- 描述: {skill.description}
- 风险等级: {skill.risk_level}

## Playbook 内容
{skill.content}

{tools_section}
{context_section}
## 用户问题
{query}

## 输出格式要求
请按以下格式输出诊断结果：
1. **问题现象总结**: 简要描述
2. **诊断过程**: 按 Playbook 步骤执行，记录每一步的结果
3. **关键发现**: 列出异常指标
4. **根本原因**: 推断的问题原因
5. **解决方案/建议**: 具体可操作的步骤
6. **严重程度**: Critical/High/Medium/Low

## 注意事项
- 严格遵循 Playbook 的 Phase 流程
- 证据不足时诚实说明，不下武断结论
- 生产环境优先建议止损（回滚/重启）而非查根因
- 不自主执行危险操作，只给出建议
- 如果信息不足，明确说明需要补充什么信息

请开始诊断：
"""
        return prompt

    def get_skills_info(self) -> Dict:
        """获取所有 skills 信息"""
        return {
            "total": len(self.registry.skills),
            "skills": [
                {
                    "name": s.name,
                    "display_name": s.display_name,
                    "description": s.description,
                    "category": s.category,
                    "triggers": s.triggers[:5]  # 只显示前5个触发词
                }
                for s in self.registry.get_all_skills()
            ]
        }
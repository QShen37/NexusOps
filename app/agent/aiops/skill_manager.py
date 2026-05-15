"""
Skill Manager - 管理所有运维 Skills
从 SKILL.md 文件中加载和匹配 Skills
"""

import re
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from loguru import logger


@dataclass
class Skill:
    """Skill 数据结构"""
    name: str
    display_name: str
    description: str
    content: str
    category: str = "general"
    risk_level: str = "low"
    triggers: List[str] = field(default_factory=list)
    allowed_tools: List[str] = field(default_factory=list)
    file_path: str = ""


class SkillManager:
    """Skill 管理器"""

    _instance = None
    _skills: Dict[str, Skill] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, skills_dir: str = "skills"):
        if not self._skills:
            self.skills_dir = Path(skills_dir)
            self._load_skills()

    def _load_skills(self):
        """加载所有 SKILL.md 文件"""
        if not self.skills_dir.exists():
            logger.warning(f"Skills 目录不存在: {self.skills_dir}")
            return

        for md_file in self.skills_dir.glob("*/SKILL.md"):
            try:
                skill = self._parse_skill_md(md_file)
                if skill:
                    self._skills[skill.name] = skill
                    logger.info(f"✓ 加载 Skill: {skill.display_name} ({skill.name})")
            except Exception as e:
                logger.error(f"加载失败 {md_file}: {e}")

        logger.info(f"共加载 {len(self._skills)} 个 Skills")

    def _parse_skill_md(self, md_path: Path) -> Optional[Skill]:
        """解析 SKILL.md 文件"""
        content = md_path.read_text(encoding='utf-8')

        # 解析 YAML frontmatter
        if not content.startswith('---'):
            return None

        parts = content.split('---', 2)
        if len(parts) < 2:
            return None

        yaml_content = parts[1].strip()
        markdown_content = parts[2].strip() if len(parts) > 2 else ""

        # 解析 YAML（简单解析）
        config = self._parse_yaml_simple(yaml_content)

        # 处理 triggers
        triggers = []
        if 'triggers' in config:
            if isinstance(config['triggers'], list):
                triggers = config['triggers']
            elif isinstance(config['triggers'], str):
                triggers = [t.strip() for t in config['triggers'].split(',')]

        # 处理 allowed_tools
        allowed_tools = []
        if 'allowed_tools' in config:
            if isinstance(config['allowed_tools'], list):
                allowed_tools = config['allowed_tools']
            elif isinstance(config['allowed_tools'], str):
                allowed_tools = [t.strip() for t in config['allowed_tools'].split(',')]

        return Skill(
            name=config.get('name', md_path.parent.name),
            display_name=config.get('display_name', config.get('name', '')),
            description=config.get('description', ''),
            content=markdown_content,
            category=config.get('category', 'general'),
            risk_level=config.get('risk_level', 'low'),
            triggers=triggers,
            allowed_tools=allowed_tools,
            file_path=str(md_path)
        )

    def _parse_yaml_simple(self, yaml_content: str) -> Dict:
        """简单的 YAML 解析器"""
        config = {}
        lines = yaml_content.split('\n')
        i = 0

        while i < len(lines):
            line = lines[i].strip()
            if not line or line.startswith('#'):
                i += 1
                continue

            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip().strip('"')

                # 检查是否是列表开始
                if value == '' or value == '[]':
                    # 可能是多行列表
                    list_items = []
                    j = i + 1
                    while j < len(lines) and lines[j].strip().startswith('-'):
                        item = lines[j].strip().lstrip('-').strip().strip('"')
                        list_items.append(item)
                        j += 1
                    if list_items:
                        config[key] = list_items
                        i = j
                        continue
                    else:
                        config[key] = []
                elif value.startswith('-'):
                    # 单行列表
                    items = [v.strip().strip('"') for v in value.split('-')[1:]]
                    config[key] = items
                else:
                    config[key] = value
            i += 1

        return config

    def match_skill(self, query: str) -> Optional[Skill]:
        """
        匹配最合适的 Skill
        基于 triggers 关键词匹配
        """
        query_lower = query.lower()
        best_match = None
        best_score = 0

        for skill in self._skills.values():
            score = 0
            for trigger in skill.triggers:
                if trigger.lower() in query_lower:
                    # 精确匹配加分
                    if len(trigger) > 2:
                        score += 0.3
                    else:
                        score += 0.2

            # 如果分数相同，优先选择更具体的 skill（更长的描述）
            if score > best_score:
                best_score = score
                best_match = skill
            elif score == best_score and best_match:
                if len(skill.description) > len(best_match.description):
                    best_match = skill

        # 需要最低分数才匹配
        if best_score >= 0.2:
            logger.info(f"Skill 匹配: {best_match.name} (分数: {best_score})")
            return best_match

        # 没有匹配到具体 skill，返回通用 skill
        generic = self._skills.get('generic_oncall')
        if generic:
            logger.info("未匹配到特定 Skill，使用通用 Skill")
            return generic

        return None

    def get_skill(self, name: str) -> Optional[Skill]:
        """获取指定 Skill"""
        return self._skills.get(name)

    def get_all_skills(self) -> List[Skill]:
        """获取所有 Skills"""
        return list(self._skills.values())

    def get_skills_summary(self) -> str:
        """获取 Skills 摘要信息"""
        summary = []
        for skill in self._skills.values():
            summary.append(
                f"- {skill.display_name}: {skill.description[:50]}... "
                f"(触发词: {', '.join(skill.triggers[:3])})"
            )
        return "\n".join(summary)
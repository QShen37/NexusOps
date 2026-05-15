import re
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import yaml


@dataclass
class SkillMetadata:
    """从 SKILL.md 解析的元数据"""
    name: str
    display_name: str
    description: str
    category: str
    risk_level: str
    triggers: List[str] = field(default_factory=list)
    allowed_tools: List[str] = field(default_factory=list)
    content: str = ""  # markdown 正文
    file_path: str = ""


class SkillRegistry:
    """纯 Markdown 驱动的 Skill 注册器"""

    def __init__(self, skills_dir: str = "skills"):
        self.skills_dir = Path(skills_dir)
        self.skills: Dict[str, SkillMetadata] = {}
        self._discover_all()

    def _discover_all(self):
        """发现所有 SKILL.md 文件"""
        if not self.skills_dir.exists():
            return

        for md_file in self.skills_dir.glob("*/SKILL.md"):
            skill = self._parse_skill_md(md_file)
            if skill:
                self.skills[skill.name] = skill
                print(f"✓ 发现 Skill: {skill.name} - {skill.display_name}")

    def _parse_skill_md(self, md_path: Path) -> Optional[SkillMetadata]:
        """解析 SKILL.md 文件"""
        try:
            content = md_path.read_text(encoding='utf-8')

            # 解析 YAML frontmatter
            if not content.startswith('---'):
                print(f"⚠ {md_path} 缺少 YAML frontmatter")
                return None

            parts = content.split('---', 2)
            if len(parts) < 2:
                return None

            yaml_content = parts[1].strip()
            markdown_content = parts[2].strip() if len(parts) > 2 else ""

            # 解析 YAML
            config = yaml.safe_load(yaml_content)

            return SkillMetadata(
                name=config.get('name', md_path.parent.name),
                display_name=config.get('display_name', config.get('name', '')),
                description=config.get('description', ''),
                category=config.get('category', 'general'),
                risk_level=config.get('risk_level', 'low'),
                triggers=config.get('triggers', []),
                allowed_tools=config.get('allowed_tools', []),
                content=markdown_content,
                file_path=str(md_path)
            )
        except Exception as e:
            print(f"✗ 解析失败 {md_path}: {e}")
            return None

    def match_skill(self, query: str) -> Optional[SkillMetadata]:
        """基于 triggers 匹配最合适的 skill"""
        query_lower = query.lower()
        best_match = None
        best_score = 0

        for skill in self.skills.values():
            score = 0
            for trigger in skill.triggers:
                if trigger.lower() in query_lower:
                    # 精确匹配分数更高
                    if len(trigger) > 2:
                        score += 0.3
                    else:
                        score += 0.2

            # 如果分数相同，优先选择更具体的 skill
            if score > best_score:
                best_score = score
                best_match = skill

        # 需要最低分数才匹配
        if best_score >= 0.2:
            return best_match

        # 没有匹配到具体 skill，返回通用 skill
        return self.skills.get('generic_oncall')

    def get_all_skills(self) -> List[SkillMetadata]:
        """获取所有 skills"""
        return list(self.skills.values())

    def get_skill(self, name: str) -> Optional[SkillMetadata]:
        """获取指定 skill"""
        return self.skills.get(name)
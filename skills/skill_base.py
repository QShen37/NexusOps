from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from enum import Enum


class SkillCategory(Enum):
    DIAGNOSIS = "diagnosis"
    ONCALL = "oncall"
    MONITORING = "monitoring"
    NETWORK = "network"
    GENERAL = "general"


@dataclass
class SkillInfo:
    """Skill元信息"""
    name: str
    description: str
    version: str
    category: SkillCategory
    keywords: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'description': self.description,
            'version': self.version,
            'category': self.category.value,
            'keywords': self.keywords,
            'dependencies': self.dependencies
        }


class Skill(ABC):
    """Skill基类"""

    def __init__(self):
        self._info: Optional[SkillInfo] = None

    @property
    def info(self) -> SkillInfo:
        if self._info is None:
            raise ValueError("Skill info not set")
        return self._info

    @abstractmethod
    def can_handle(self, query: str, context: Dict[str, Any] = None) -> float:
        """
        判断该skill是否能处理当前查询
        返回0-1之间的分数，分数越高越匹配
        """
        pass

    @abstractmethod
    async def execute(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """执行skill"""
        pass

    def load_metadata_from_md(self, md_path: str) -> Dict[str, Any]:
        """从SKILL.md文件中解析元数据"""
        import re
        metadata = {}

        try:
            with open(md_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # 解析YAML frontmatter（如果存在）
            if content.startswith('---'):
                parts = content.split('---', 2)
                if len(parts) >= 2:
                    yaml_content = parts[1].strip()
                    for line in yaml_content.split('\n'):
                        if ':' in line:
                            key, value = line.split(':', 1)
                            metadata[key.strip()] = value.strip()
        except Exception as e:
            print(f"Error loading metadata from {md_path}: {e}")

        return metadata
"""
枢机 (Shuji) - 技能系统
实现可插拔的Skill架构
"""
from .skill_manager import SkillManager, Skill, SkillRegistry
from .builtin_skills import (
    FileSkill,
    BashSkill,
    BrowserSkill,
    SearchSkill,
    CodeSkill,
)

__all__ = [
    'SkillManager',
    'Skill',
    'SkillRegistry',
    'FileSkill',
    'BashSkill',
    'BrowserSkill',
    'SearchSkill',
    'CodeSkill',
]
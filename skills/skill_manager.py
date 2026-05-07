"""
枢机 (Shuji) - 技能管理器
实现Skill的注册、发现和执行，支持从Python文件动态加载
"""
import os
import json
import importlib.util
import inspect
from typing import Dict, List, Optional, Callable, Any, Type
from dataclasses import dataclass, field
import logging

from .builtin_skills import BaseSkill


logger = logging.getLogger(__name__)


@dataclass
class Skill:
    """技能定义"""
    name: str
    description: str
    version: str = "1.0.0"
    author: str = "unknown"
    tags: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    handler: Optional[Callable] = None
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "author": self.author,
            "tags": self.tags,
            "parameters": self.parameters,
        }


class SkillRegistry:
    """技能注册表"""
    
    def __init__(self):
        self.skills: Dict[str, Skill] = {}
    
    def register(self, skill: Skill):
        """注册技能"""
        self.skills[skill.name] = skill
    
    def unregister(self, name: str):
        """注销技能"""
        if name in self.skills:
            del self.skills[name]
    
    def get(self, name: str) -> Optional[Skill]:
        """获取技能"""
        return self.skills.get(name)
    
    def list_all(self) -> List[Skill]:
        """列出所有技能"""
        return list(self.skills.values())
    
    def search(self, query: str) -> List[Skill]:
        """搜索技能"""
        results = []
        query_lower = query.lower()
        
        for skill in self.skills.values():
            if (query_lower in skill.name.lower() or
                query_lower in skill.description.lower() or
                any(query_lower in tag.lower() for tag in skill.tags)):
                results.append(skill)
        
        return results


class SkillManager:
    """
    技能管理器
    """
    
    def __init__(self, skills_dir: str = "./skills"):
        self.skills_dir = skills_dir
        self.registry = SkillRegistry()
        self.loaded_skills: Dict[str, Any] = {}
        
        # 确保目录存在
        os.makedirs(skills_dir, exist_ok=True)
    
    def register_builtin_skills(self, **kwargs):
        """注册内置技能，允许传递配置参数"""
        from .builtin_skills import (
            FileSkill, BashSkill, BrowserSkill, SearchSkill, CodeSkill
        )
        
        # 文件技能
        file_skill_instance = FileSkill(allowed_base_dir=kwargs.get("file_base_dir"))
        file_skill = Skill(
            name="file",
            description="File operations including read, write, and edit",
            tags=["file", "io"],
            parameters={
                "action": {"type": "string", "enum": ["read", "write", "edit", "delete", "list"]},
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            handler=file_skill_instance.execute,
        )
        self.registry.register(file_skill)
        
        # Bash技能
        bash_skill_instance = BashSkill(
            allowed_commands=kwargs.get("bash_allowed_commands"),
            timeout=kwargs.get("bash_timeout", 30.0)
        )
        bash_skill = Skill(
            name="bash",
            description="Execute bash commands (with security restrictions)",
            tags=["shell", "command"],
            parameters={
                "command": {"type": "string"},
                "cwd": {"type": "string"},
                "timeout": {"type": "number"},
            },
            handler=bash_skill_instance.execute,
        )
        self.registry.register(bash_skill)
        
        # 浏览器技能
        browser_skill_instance = BrowserSkill(headless=kwargs.get("browser_headless", True))
        browser_skill = Skill(
            name="browser",
            description="Web browser automation",
            tags=["web", "browser", "automation"],
            parameters={
                "action": {"type": "string", "enum": ["navigate", "click", "type", "extract", "screenshot"]},
                "url": {"type": "string"},
                "selector": {"type": "string"},
                "text": {"type": "string"},
            },
            handler=browser_skill_instance.execute,
        )
        self.registry.register(browser_skill)
        
        # 搜索技能
        search_skill_instance = SearchSkill(api_key=kwargs.get("search_api_key"))
        search_skill = Skill(
            name="search",
            description="Web search",
            tags=["web", "search", "information"],
            parameters={
                "query": {"type": "string"},
                "num_results": {"type": "number"},
            },
            handler=search_skill_instance.execute,
        )
        self.registry.register(search_skill)
        
        # 代码技能
        code_skill_instance = CodeSkill(sandbox=kwargs.get("code_sandbox", True))
        code_skill = Skill(
            name="code",
            description="Code operations including analysis and safe execution",
            tags=["code", "programming"],
            parameters={
                "action": {"type": "string", "enum": ["analyze", "execute", "format"]},
                "language": {"type": "string"},
                "code": {"type": "string"},
            },
            handler=code_skill_instance.execute,
        )
        self.registry.register(code_skill)
    
    def load_skill_from_file(self, filepath: str) -> Optional[Skill]:
        """从Python文件加载技能（动态导入）"""
        try:
            # 获取模块名
            module_name = os.path.splitext(os.path.basename(filepath))[0]
            
            # 加载规范
            spec = importlib.util.spec_from_file_location(module_name, filepath)
            if spec is None or spec.loader is None:
                raise ImportError(f"Could not load spec from {filepath}")
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # 寻找继承自BaseSkill的类
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if issubclass(obj, BaseSkill) and obj != BaseSkill:
                    # 实例化
                    instance = obj()
                    # 查找execute方法
                    if hasattr(instance, 'execute') and callable(instance.execute):
                        # 创建Skill定义
                        # 可以从类文档或属性中获取元数据
                        skill = Skill(
                            name=module_name,
                            description=getattr(instance, '__doc__', f"Skill loaded from {filepath}"),
                            version="1.0.0",
                            author="unknown",
                            tags=[],
                            parameters={},  # 可以从instance获取schema
                            handler=instance.execute,
                        )
                        self.registry.register(skill)
                        logger.info(f"Loaded skill from {filepath}")
                        return skill
            return None
        
        except Exception as e:
            logger.error(f"Error loading skill from {filepath}: {e}")
            return None
    
    def load_skills_from_directory(self, directory: str):
        """从目录加载所有技能（Python文件）"""
        for filename in os.listdir(directory):
            if filename.endswith('.py') and not filename.startswith('__'):
                filepath = os.path.join(directory, filename)
                self.load_skill_from_file(filepath)
    
    async def execute(self, skill_name: str, **kwargs) -> Any:
        """执行技能"""
        skill = self.registry.get(skill_name)
        
        if skill is None:
            raise ValueError(f"Skill not found: {skill_name}")
        
        if skill.handler is None:
            raise ValueError(f"Skill has no handler: {skill_name}")
        
        return await skill.handler(**kwargs)
    
    def list_skills(self) -> List[Dict]:
        """列出所有技能"""
        return [skill.to_dict() for skill in self.registry.list_all()]
    
    def search_skills(self, query: str) -> List[Dict]:
        """搜索技能"""
        skills = self.registry.search(query)
        return [skill.to_dict() for skill in skills]
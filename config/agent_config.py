"""
枢机 (Shuji) - 智能体配置系统
实现SOUL.md配置格式和Agent身份管理
"""
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum
import json
import yaml


class AgentRole(Enum):
    """智能体角色类型"""
    ORCHESTRATOR = "orchestrator"  # 协调者
    RESEARCHER = "researcher"      # 研究员
    CODER = "coder"                # 程序员
    WRITER = "writer"              # 写手
    ANALYST = "analyst"            # 分析师
    GENERALIST = "generalist"      # 通才


class AgentStatus(Enum):
    """智能体状态"""
    IDLE = "idle"
    BUSY = "busy"
    PAUSED = "paused"
    ERROR = "error"
    OFFLINE = "offline"


@dataclass
class SoulConfig:
    """
    SOUL.md配置 - 定义智能体的灵魂/个性
    
    对应OpenClaw的SOUL.md格式:
    - Identity: 身份定义
    - Personality: 个性特征
    - Rules: 行为规则
    - Skills: 技能列表
    """
    
    # 身份 (Identity)
    name: str = "Assistant"
    role: str = "General AI Assistant"
    description: str = "A helpful AI assistant"
    version: str = "1.0.0"
    emoji: str = "🤖"
    
    # 个性 (Personality)
    traits: List[str] = field(default_factory=lambda: [
        "helpful", "friendly", "professional"
    ])
    communication_style: str = "clear and concise"
    tone: str = "professional but warm"
    creativity_level: float = 0.5  # 0.0-1.0
    formality_level: float = 0.5   # 0.0-1.0
    
    # 行为规则 (Rules)
    rules: List[str] = field(default_factory=lambda: [
        "Always be helpful and accurate",
        "Respect user privacy",
        "Ask for clarification when uncertain"
    ])
    prohibitions: List[str] = field(default_factory=lambda: [
        "Never share sensitive information",
        "Never execute destructive commands without confirmation"
    ])
    
    # 技能 (Skills)
    skills: List[str] = field(default_factory=lambda: [
        "conversation", "file_read", "file_write"
    ])
    
    # 工作流 (Workflow)
    workflow_steps: List[str] = field(default_factory=list)
    
    # 输出格式 (Output Format)
    output_format: str = "markdown"
    preferred_language: str = "zh"
    
    def to_markdown(self) -> str:
        """转换为SOUL.md格式"""
        md = f"""# {self.name}

## Identity
- Name: {self.name}
- Role: {self.role}
- Description: {self.description}
- Version: {self.version}
- Emoji: {self.emoji}

## Personality
"""
        for trait in self.traits:
            md += f"- {trait}\n"
        
        md += f"""
- Communication Style: {self.communication_style}
- Tone: {self.tone}
- Creativity Level: {self.creativity_level}
- Formality Level: {self.formality_level}

## Rules
"""
        for rule in self.rules:
            md += f"- {rule}\n"
        
        md += "\n## Prohibitions\n"
        for prohibition in self.prohibitions:
            md += f"- {prohibition}\n"
        
        md += "\n## Skills\n"
        for skill in self.skills:
            md += f"- {skill}\n"
        
        if self.workflow_steps:
            md += "\n## Workflow\n"
            for i, step in enumerate(self.workflow_steps, 1):
                md += f"{i}. {step}\n"
        
        md += f"""
## Output Format
- Format: {self.output_format}
- Preferred Language: {self.preferred_language}
"""
        return md
    
    @classmethod
    def from_markdown(cls, markdown_text: str) -> 'SoulConfig':
        """从SOUL.md格式解析"""
        config = cls()
        lines = markdown_text.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # 识别章节
            if line.startswith('# '):
                config.name = line[2:].strip()
            elif line.startswith('## '):
                current_section = line[3:].strip().lower()
            elif line.startswith('- ') or line.startswith('* '):
                content = line[2:].strip()
                
                if current_section == 'identity':
                    if content.startswith('Name:'):
                        config.name = content[5:].strip()
                    elif content.startswith('Role:'):
                        config.role = content[5:].strip()
                    elif content.startswith('Description:'):
                        config.description = content[12:].strip()
                    elif content.startswith('Version:'):
                        config.version = content[8:].strip()
                    elif content.startswith('Emoji:'):
                        config.emoji = content[6:].strip()
                
                elif current_section == 'personality':
                    if content.startswith('Communication Style:'):
                        config.communication_style = content[20:].strip()
                    elif content.startswith('Tone:'):
                        config.tone = content[5:].strip()
                    elif content.startswith('Creativity Level:'):
                        config.creativity_level = float(content[17:].strip())
                    elif content.startswith('Formality Level:'):
                        config.formality_level = float(content[16:].strip())
                    else:
                        config.traits.append(content)
                
                elif current_section == 'rules':
                    config.rules.append(content)
                
                elif current_section == 'prohibitions':
                    config.prohibitions.append(content)
                
                elif current_section == 'skills':
                    config.skills.append(content)
        
        return config
    
    def save_to_file(self, filepath: str):
        """保存到SOUL.md文件"""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(self.to_markdown())
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'SoulConfig':
        """从SOUL.md文件加载"""
        with open(filepath, 'r', encoding='utf-8') as f:
            return cls.from_markdown(f.read())


@dataclass
class AgentIdentity:
    """智能体身份信息"""
    agent_id: str
    name: str
    emoji: str = "🤖"
    display_name: Optional[str] = None
    avatar_url: Optional[str] = None
    color: str = "#6366f1"  # 默认靛蓝色
    
    def __post_init__(self):
        if self.display_name is None:
            self.display_name = f"{self.emoji} {self.name}"


@dataclass
class AgentConfig:
    """
    智能体完整配置
    
    整合SOUL配置、模型配置、工具配置等
    """
    
    # 基础身份
    identity: AgentIdentity = field(default_factory=lambda: AgentIdentity(
        agent_id="default",
        name="Assistant",
        emoji="🤖"
    ))
    
    # SOUL配置
    soul: SoulConfig = field(default_factory=SoulConfig)
    
    # 模型配置
    model_name: str = "deepseekv3mini"
    model_version: str = "1.0.0"
    temperature: float = 0.7
    max_tokens: int = 2048
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.0
    
    # 角色和状态
    role: AgentRole = AgentRole.GENERALIST
    status: AgentStatus = AgentStatus.IDLE
    
    # 工具配置
    enabled_tools: List[str] = field(default_factory=lambda: [
        "file_read", "file_write", "bash", "search"
    ])
    disabled_tools: List[str] = field(default_factory=list)
    
    # 记忆配置
    memory_enabled: bool = True
    memory_max_context: int = 10  # 最大上下文轮数
    persistent_memory: bool = True
    
    # 频道配置
    allowed_channels: List[str] = field(default_factory=lambda: ["cli", "websocket"])
    default_channel: str = "cli"
    
    # 安全配置
    sandbox_mode: str = "docker"  # docker, process, none
    require_approval_for: List[str] = field(default_factory=lambda: [
        "file_delete", "bash_destructive"
    ])
    
    # 工作区配置
    workspace_dir: str = "./workspace"
    memory_dir: str = "./memory"
    skills_dir: str = "./skills"
    
    # 高级配置
    max_concurrent_tasks: int = 5
    task_timeout: float = 60.0
    auto_save_interval: float = 300.0
    
    # 元数据
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'identity': {
                'agent_id': self.identity.agent_id,
                'name': self.identity.name,
                'emoji': self.identity.emoji,
                'display_name': self.identity.display_name,
                'color': self.identity.color,
            },
            'soul': self.soul.to_markdown(),
            'model_name': self.model_name,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'role': self.role.value,
            'enabled_tools': self.enabled_tools,
            'memory_enabled': self.memory_enabled,
            'allowed_channels': self.allowed_channels,
            'sandbox_mode': self.sandbox_mode,
        }
    
    def save(self, filepath: str):
        """保存配置到YAML文件"""
        with open(filepath, 'w', encoding='utf-8') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, allow_unicode=True)
    
    @classmethod
    def load(cls, filepath: str) -> 'AgentConfig':
        """从YAML文件加载配置"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        config = cls()
        
        if 'identity' in data:
            id_data = data['identity']
            config.identity = AgentIdentity(
                agent_id=id_data.get('agent_id', 'default'),
                name=id_data.get('name', 'Assistant'),
                emoji=id_data.get('emoji', '🤖'),
                display_name=id_data.get('display_name'),
                color=id_data.get('color', '#6366f1')
            )
        
        if 'soul' in data:
            if isinstance(data['soul'], str):
                config.soul = SoulConfig.from_markdown(data['soul'])
        
        config.model_name = data.get('model_name', 'deepseekv3mini')
        config.temperature = data.get('temperature', 0.7)
        config.max_tokens = data.get('max_tokens', 2048)
        config.enabled_tools = data.get('enabled_tools', [])
        config.memory_enabled = data.get('memory_enabled', True)
        config.allowed_channels = data.get('allowed_channels', ['cli'])
        
        return config


# 预设智能体配置
PRESET_AGENTS = {
    "orchestrator": AgentConfig(
        identity=AgentIdentity("orchestrator", "Orchestrator", "🎯"),
        soul=SoulConfig(
            name="Orchestrator",
            role="Multi-Agent Orchestrator",
            description="Coordinates multiple agents to accomplish complex tasks",
            traits=["strategic", "organized", "decisive"],
            skills=["orchestration", "task_delegation", "coordination"]
        ),
        role=AgentRole.ORCHESTRATOR,
        temperature=0.3,
    ),
    
    "researcher": AgentConfig(
        identity=AgentIdentity("researcher", "Researcher", "🔍"),
        soul=SoulConfig(
            name="Researcher",
            role="Research Analyst",
            description="Gathers and analyzes information from various sources",
            traits=["thorough", "analytical", "curious"],
            skills=["web_search", "data_analysis", "summarization"]
        ),
        role=AgentRole.RESEARCHER,
        temperature=0.5,
    ),
    
    "coder": AgentConfig(
        identity=AgentIdentity("coder", "CodeSmith", "⚡"),
        soul=SoulConfig(
            name="CodeSmith",
            role="Software Developer",
            description="Writes, reviews, and debugs code",
            traits=["precise", "logical", "efficient"],
            skills=["code_writing", "code_review", "debugging", "testing"]
        ),
        role=AgentRole.CODER,
        temperature=0.2,
    ),
    
    "writer": AgentConfig(
        identity=AgentIdentity("writer", "Writer", "✍️"),
        soul=SoulConfig(
            name="Writer",
            role="Content Creator",
            description="Creates engaging and well-structured content",
            traits=["creative", "articulate", "adaptable"],
            skills=["writing", "editing", "formatting", "seo"]
        ),
        role=AgentRole.WRITER,
        temperature=0.8,
    ),
}
"""
枢机 (Shuji) - 多智能体框架配置系统
基于 DeepSeekV3Mini 架构实现
"""
from .shuji_config import ShujiConfig, get_default_config, get_lightweight_config
from .agent_config import AgentConfig, SoulConfig, AgentIdentity
from .gateway_config import GatewayConfig, ChannelConfig, SecurityConfig

__all__ = [
    'ShujiConfig',
    'get_default_config',
    'get_lightweight_config',
    'AgentConfig',
    'SoulConfig',
    'AgentIdentity',
    'GatewayConfig',
    'ChannelConfig',
    'SecurityConfig',
]
"""
枢机 (Shuji) - 多智能体框架
基于DeepSeekV3Mini架构实现，功能与OpenClaw近乎一致

核心特性:
- 基于DeepSeekV3Mini的MLA + MoE + MTP架构
- Gateway网关系统 (WebSocket)
- MCP/ACP协议支持
- 三层记忆系统 (Markdown + Vector + Graph)
- 可插拔技能系统
- Hub-and-Spoke多智能体协调
"""

__version__ = "1.0.0"
__author__ = "Shuji Team"

from .config import ShujiConfig, get_default_config, get_lightweight_config
from .core import ShujiGateway, ShujiAgent, AgentOrchestrator
from .models import ShujiForCausalLM

__all__ = [
    '__version__',
    'ShujiConfig',
    'get_default_config',
    'get_lightweight_config',
    'ShujiGateway',
    'ShujiAgent',
    'AgentOrchestrator',
    'ShujiForCausalLM',
]
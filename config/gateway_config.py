"""
枢机 (Shuji) - 网关配置系统
定义Gateway、频道和安全配置
"""
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum


class ChannelType(Enum):
    """频道类型"""
    CLI = "cli"
    WEBSOCKET = "websocket"
    TELEGRAM = "telegram"
    SLACK = "slack"
    DISCORD = "discord"
    EMAIL = "email"
    WHATSAPP = "whatsapp"
    HTTP = "http"


class AuthMethod(Enum):
    """认证方式"""
    TOKEN = "token"
    PASSWORD = "password"
    OAUTH = "oauth"
    API_KEY = "api_key"
    NONE = "none"


@dataclass
class SecurityConfig:
    """安全配置"""
    
    # 认证配置
    auth_method: AuthMethod = AuthMethod.TOKEN
    token_secret: Optional[str] = None
    token_expiry: float = 3600.0  # 1小时
    
    # 访问控制
    allowed_origins: List[str] = field(default_factory=lambda: ["localhost", "127.0.0.1"])
    allowed_ips: List[str] = field(default_factory=lambda: ["127.0.0.1"])
    blocked_ips: List[str] = field(default_factory=list)
    
    # 速率限制
    rate_limit_enabled: bool = True
    rate_limit_requests: int = 100  # 每窗口请求数
    rate_limit_window: float = 60.0  # 窗口大小(秒)
    
    # 加密配置
    encryption_enabled: bool = True
    tls_cert_path: Optional[str] = None
    tls_key_path: Optional[str] = None
    
    # 审计日志
    audit_log_enabled: bool = True
    audit_log_path: str = "./logs/audit.log"
    
    # 沙箱配置
    sandbox_enabled: bool = True
    sandbox_type: str = "docker"  # docker, process, none
    sandbox_cpu_limit: float = 1.0  # CPU核心数
    sandbox_memory_limit: str = "512m"  # 内存限制
    sandbox_timeout: float = 30.0  # 沙箱超时


@dataclass
class ChannelConfig:
    """频道配置"""
    
    # 基础配置
    channel_type: ChannelType = ChannelType.CLI
    enabled: bool = True
    
    # 连接配置
    host: str = "127.0.0.1"
    port: int = 0  # 0表示使用默认端口
    path: str = "/"
    
    # 认证配置
    auth_required: bool = False
    auth_method: AuthMethod = AuthMethod.NONE
    api_key: Optional[str] = None
    
    # 消息配置
    max_message_size: int = 65536  # 64KB
    message_timeout: float = 30.0
    
    # 特定频道配置
    telegram_bot_token: Optional[str] = None
    slack_bot_token: Optional[str] = None
    discord_bot_token: Optional[str] = None
    email_smtp_server: Optional[str] = None
    email_imap_server: Optional[str] = None
    
    # 消息处理配置
    auto_reply: bool = True
    reply_template: str = "Received: {message}"
    
    # 过滤器配置
    allowed_users: List[str] = field(default_factory=list)
    blocked_users: List[str] = field(default_factory=list)
    allowed_channels: List[str] = field(default_factory=list)


@dataclass
class GatewayConfig:
    """
    网关配置
    
    对应OpenClaw Gateway的核心配置
    """
    
    # 基础配置
    host: str = "127.0.0.1"
    port: int = 18789
    protocol: str = "websocket"
    
    # 服务器配置
    max_connections: int = 1000
    max_agents: int = 100
    worker_threads: int = 4
    
    # 连接配置
    heartbeat_interval: float = 30.0
    connection_timeout: float = 300.0
    reconnect_attempts: int = 3
    reconnect_delay: float = 5.0
    
    # 协议配置
    protocol_version: str = "1.0.0"
    min_protocol_version: str = "1.0.0"
    max_protocol_version: str = "1.0.0"
    
    # 安全配置
    security: SecurityConfig = field(default_factory=SecurityConfig)
    
    # 频道配置
    channels: Dict[ChannelType, ChannelConfig] = field(default_factory=lambda: {
        ChannelType.CLI: ChannelConfig(
            channel_type=ChannelType.CLI,
            enabled=True,
        ),
        ChannelType.WEBSOCKET: ChannelConfig(
            channel_type=ChannelType.WEBSOCKET,
            enabled=True,
            host="127.0.0.1",
            port=18789,
        ),
    })
    
    # 日志配置
    log_level: str = "INFO"
    log_file: str = "./logs/gateway.log"
    log_max_size: int = 10485760  # 10MB
    log_backup_count: int = 5
    
    # 性能配置
    request_queue_size: int = 1000
    batch_size: int = 32
    batch_timeout: float = 0.1
    
    # 插件配置
    plugins_enabled: bool = True
    plugins_directory: str = "./plugins"
    
    # 健康检查
    health_check_enabled: bool = True
    health_check_interval: float = 60.0
    
    def get_channel_port(self, channel_type: ChannelType) -> int:
        """获取频道端口"""
        default_ports = {
            ChannelType.CLI: 0,  # CLI不需要端口
            ChannelType.WEBSOCKET: 18789,
            ChannelType.TELEGRAM: 0,  # 通过webhook
            ChannelType.SLACK: 3000,
            ChannelType.DISCORD: 0,  # 通过gateway
            ChannelType.HTTP: 8080,
        }
        
        if channel_type in self.channels:
            port = self.channels[channel_type].port
            if port > 0:
                return port
        
        return default_ports.get(channel_type, 0)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'host': self.host,
            'port': self.port,
            'protocol': self.protocol,
            'max_connections': self.max_connections,
            'max_agents': self.max_agents,
            'heartbeat_interval': self.heartbeat_interval,
            'connection_timeout': self.connection_timeout,
            'security': {
                'auth_method': self.security.auth_method.value,
                'rate_limit_enabled': self.security.rate_limit_enabled,
                'sandbox_enabled': self.security.sandbox_enabled,
            },
            'channels': {
                k.value: {
                    'enabled': v.enabled,
                    'host': v.host,
                    'port': v.port,
                }
                for k, v in self.channels.items()
            },
        }


# 默认频道端口映射
DEFAULT_CHANNEL_PORTS = {
    ChannelType.CLI: None,
    ChannelType.WEBSOCKET: 18789,
    ChannelType.HTTP: 8080,
    ChannelType.TELEGRAM: None,
    ChannelType.SLACK: 3000,
    ChannelType.DISCORD: None,
    ChannelType.EMAIL: None,
    ChannelType.WHATSAPP: None,
}
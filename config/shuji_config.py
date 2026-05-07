"""
枢机 (Shuji) - 主配置系统
整合DeepSeekV3Mini模型配置与多智能体框架配置
"""
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import json


@dataclass
class ShujiConfig:
    """
    枢机框架主配置类
    
    整合以下子系统配置:
    - 模型配置 (基于DeepSeekV3Mini)
    - 网关配置
    - 智能体配置
    - 记忆系统配置
    - 安全配置
    """
    
    # ==================== 模型配置 (DeepSeekV3Mini) ====================
    vocab_size: int = 32000
    num_layers: int = 24
    hidden_size: int = 1024
    num_attention_heads: int = 16
    qk_head_dim: int = 64
    v_head_dim: int = 64
    qk_nope_head_dim: int = 64
    qk_rope_head_dim: int = 32
    kv_lora_rank: int = 128
    q_lora_rank: int = 256
    max_position_embeddings: int = 4096
    rope_theta: float = 10000.0
    
    # DeepSeekMoE参数
    num_shared_experts: int = 1
    num_routed_experts: int = 32
    num_experts_per_tok: int = 4
    expert_intermediate_size: int = 256
    
    # 无辅助损失负载均衡
    aux_loss_free: bool = True
    bias_update_speed: float = 0.001
    seq_aux_loss_factor: float = 0.0001
    
    # MTP参数
    num_mtp_tokens: int = 1
    mtp_loss_weight: float = 0.3
    
    # 归一化和初始化
    rms_norm_eps: float = 1e-6
    initializer_range: float = 0.006
    dropout_rate: float = 0.0
    attention_dropout: float = 0.0
    use_gradient_checkpointing: bool = False
    
    # ==================== 网关配置 ====================
    gateway_host: str = "127.0.0.1"
    gateway_port: int = 18789
    gateway_protocol: str = "websocket"
    max_connections: int = 1000
    heartbeat_interval: float = 30.0
    connection_timeout: float = 300.0
    
    # ==================== 智能体配置 ====================
    max_agents: int = 100
    default_agent_model: str = "deepseekv3mini"
    agent_timeout: float = 60.0
    max_agent_memory_mb: int = 512
    
    # ==================== 记忆系统配置 ====================
    memory_backend: str = "sqlite"  # sqlite, chromadb, redis
    memory_max_entries: int = 10000
    vector_embedding_model: str = "all-MiniLM-L6-v2"
    knowledge_graph_enabled: bool = True
    
    # ==================== 安全配置 ====================
    sandbox_enabled: bool = True
    sandbox_type: str = "docker"  # docker, process, none
    max_file_size_mb: int = 100
    allowed_file_extensions: List[str] = field(default_factory=lambda: [
        ".txt", ".md", ".py", ".json", ".yaml", ".yml", ".csv"
    ])
    blocked_commands: List[str] = field(default_factory=lambda: [
        "rm -rf /", "mkfs", "dd", "> /dev/sda"
    ])
    
    # ==================== 技能系统配置 ====================
    skills_directory: str = "./skills"
    auto_discover_skills: bool = True
    skill_timeout: float = 30.0
    max_skill_memory_mb: int = 256
    
    # ==================== 频道配置 ====================
    channels_enabled: List[str] = field(default_factory=lambda: [
        "cli", "websocket"
    ])
    
    # ==================== 日志配置 ====================
    log_level: str = "INFO"
    log_file: Optional[str] = None
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    @property
    def head_dim(self) -> int:
        """总头维度 = 无RoPE维度 + 有RoPE维度"""
        return self.qk_nope_head_dim + self.qk_rope_head_dim
    
    @property
    def moe_intermediate_size(self) -> int:
        """Dense FFN的中间维度"""
        return self.hidden_size * 4
    
    @property
    def total_params(self) -> int:
        """估算总参数量"""
        # Embedding参数
        embedding_params = self.vocab_size * self.hidden_size
        
        # MLA参数
        mla_params = (
            self.hidden_size * self.q_lora_rank +
            self.q_lora_rank * self.num_attention_heads * self.head_dim +
            self.hidden_size * self.kv_lora_rank +
            self.kv_lora_rank * self.num_attention_heads * self.qk_nope_head_dim +
            self.kv_lora_rank * self.num_attention_heads * self.v_head_dim +
            self.hidden_size * self.num_attention_heads * self.qk_rope_head_dim +
            self.num_attention_heads * self.v_head_dim * self.hidden_size
        )
        
        # MoE参数
        moe_params = (
            2 * self.hidden_size * self.expert_intermediate_size +
            self.num_routed_experts * 2 * self.hidden_size * self.expert_intermediate_size +
            self.hidden_size * self.num_routed_experts
        )
        
        # 每层总参数
        layer_params = mla_params + moe_params + 2 * self.hidden_size
        all_layers_params = self.num_layers * layer_params
        
        # 最终输出层
        output_params = self.hidden_size * self.vocab_size
        
        # MTP模块参数
        mtp_params = self.num_mtp_tokens * (
            mla_params + moe_params + 2 * self.hidden_size + self.hidden_size * 2 * self.hidden_size
        )
        
        total = embedding_params + all_layers_params + output_params + mtp_params
        return total
    
    @property
    def activated_params(self) -> int:
        """估算每个token激活的参数量"""
        embedding_params = self.vocab_size * self.hidden_size
        
        mla_params = (
            self.hidden_size * self.q_lora_rank +
            self.q_lora_rank * self.num_attention_heads * self.head_dim +
            self.hidden_size * self.kv_lora_rank +
            self.kv_lora_rank * self.num_attention_heads * self.qk_nope_head_dim +
            self.kv_lora_rank * self.num_attention_heads * self.v_head_dim +
            self.hidden_size * self.num_attention_heads * self.qk_rope_head_dim +
            self.num_attention_heads * self.v_head_dim * self.hidden_size
        )
        
        moe_params = (
            2 * self.hidden_size * self.expert_intermediate_size +
            self.num_experts_per_tok * 2 * self.hidden_size * self.expert_intermediate_size +
            self.hidden_size * self.num_routed_experts
        )
        
        layer_params = mla_params + moe_params + 2 * self.hidden_size
        all_layers_params = self.num_layers * layer_params
        output_params = self.hidden_size * self.vocab_size
        
        return embedding_params + all_layers_params + output_params
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith('_')
        }
    
    def to_json(self, indent: int = 2) -> str:
        """转换为JSON字符串"""
        return json.dumps(self.to_dict(), indent=indent, default=str)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ShujiConfig':
        """从字典创建配置"""
        return cls(**config_dict)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'ShujiConfig':
        """从JSON字符串创建配置"""
        return cls.from_dict(json.loads(json_str))
    
    def save(self, filepath: str):
        """保存配置到文件"""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(self.to_json())
    
    @classmethod
    def load(cls, filepath: str) -> 'ShujiConfig':
        """从文件加载配置"""
        with open(filepath, 'r', encoding='utf-8') as f:
            return cls.from_json(f.read())


def get_default_config() -> ShujiConfig:
    """获取默认配置 (约1.2B参数)"""
    return ShujiConfig(
        vocab_size=32000,
        num_layers=24,
        hidden_size=1024,
        num_attention_heads=16,
        qk_head_dim=64,
        v_head_dim=64,
        qk_nope_head_dim=64,
        qk_rope_head_dim=32,
        kv_lora_rank=128,
        q_lora_rank=256,
        max_position_embeddings=4096,
        num_shared_experts=1,
        num_routed_experts=32,
        num_experts_per_tok=4,
        expert_intermediate_size=256,
        num_mtp_tokens=1,
        gateway_host="127.0.0.1",
        gateway_port=18789,
        max_agents=100,
        memory_backend="sqlite",
        sandbox_enabled=True,
    )


def get_lightweight_config() -> ShujiConfig:
    """获取轻量级配置 (约600M参数，用于快速测试)"""
    return ShujiConfig(
        vocab_size=32000,
        num_layers=16,
        hidden_size=768,
        num_attention_heads=12,
        qk_head_dim=64,
        v_head_dim=64,
        qk_nope_head_dim=64,
        qk_rope_head_dim=32,
        kv_lora_rank=96,
        q_lora_rank=192,
        max_position_embeddings=2048,
        num_shared_experts=1,
        num_routed_experts=16,
        num_experts_per_tok=2,
        expert_intermediate_size=192,
        num_mtp_tokens=1,
        gateway_host="127.0.0.1",
        gateway_port=18789,
        max_agents=50,
        memory_backend="sqlite",
        sandbox_enabled=True,
    )


def get_enterprise_config() -> ShujiConfig:
    """获取企业级配置"""
    return ShujiConfig(
        vocab_size=64000,
        num_layers=32,
        hidden_size=1536,
        num_attention_heads=24,
        qk_head_dim=64,
        v_head_dim=64,
        qk_nope_head_dim=64,
        qk_rope_head_dim=32,
        kv_lora_rank=192,
        q_lora_rank=384,
        max_position_embeddings=8192,
        num_shared_experts=2,
        num_routed_experts=64,
        num_experts_per_tok=6,
        expert_intermediate_size=384,
        num_mtp_tokens=2,
        gateway_host="0.0.0.0",
        gateway_port=18789,
        max_agents=500,
        memory_backend="redis",
        knowledge_graph_enabled=True,
        sandbox_enabled=True,
        channels_enabled=["cli", "websocket", "telegram", "slack", "discord"],
    )


# 预设配置
CONFIGS = {
    "default": get_default_config(),
    "lightweight": get_lightweight_config(),
    "enterprise": get_enterprise_config(),
}
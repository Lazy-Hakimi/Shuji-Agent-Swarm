"""
枢机 (Shuji) - 基础层组件
包含RMSNorm、RoPE、DecoderLayer等
"""
import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """RMSNorm归一化层"""
    
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class RotaryEmbedding(nn.Module):
    """旋转位置编码 (RoPE)"""
    
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 4096,
        base: float = 10000.0
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        # 预计算逆频率
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # 预计算缓存
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=self.inv_freq.device,
            dtype=torch.float32
        )
    
    def _set_cos_sin_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        
        # 计算频率
        freqs = torch.outer(t, self.inv_freq)
        # 重复频率以匹配维度
        emb = torch.cat([freqs, freqs], dim=-1)
        
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)
    
    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
        
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype)
        )


class DeepSeekV3DecoderLayer(nn.Module):
    """DeepSeek-V3解码器层"""
    
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        # 自注意力
        from .attention import MultiHeadLatentAttention
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = MultiHeadLatentAttention(config)
        
        # FFN/MoE
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        if layer_idx < 3:
            # 前3层使用标准FFN (SwiGLU)
            intermediate_size = config.moe_intermediate_size
            self.w1 = nn.Linear(config.hidden_size, intermediate_size, bias=False)
            self.w2 = nn.Linear(intermediate_size, config.hidden_size, bias=False)
            self.w3 = nn.Linear(config.hidden_size, intermediate_size, bias=False)
            self.use_moe = False
        else:
            # 后续层使用MoE
            from .moe import DeepSeekMoE
            self.mlp = DeepSeekMoE(config, layer_idx)
            self.use_moe = True
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        training_step: bool = True,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]], Optional[torch.Tensor]]:
        """解码器层前向传播"""
        residual = hidden_states
        
        # 自注意力
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states
        
        # FFN/MoE
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        
        if self.use_moe:
            hidden_states, aux_loss = self.mlp(hidden_states, training_step=training_step)
        else:
            # SwiGLU
            hidden_states = self.w2(F.silu(self.w1(hidden_states)) * self.w3(hidden_states))
            aux_loss = None
        
        hidden_states = residual + hidden_states
        
        return hidden_states, present_key_value, aux_loss
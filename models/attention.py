"""
枢机 (Shuji) - 多头潜在注意力 (MLA)
基于DeepSeekV3论文实现
"""
import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import RMSNorm, RotaryEmbedding


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """旋转张量的一半维度"""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """应用旋转位置编码"""
    # 确保cos和sin有正确的维度
    if cos.dim() == 2:
        cos = cos.unsqueeze(0).unsqueeze(2)
        sin = sin.unsqueeze(0).unsqueeze(2)
    elif cos.dim() == 3:
        cos = cos.unsqueeze(2)
        sin = sin.unsqueeze(2)
    
    # 确保维度匹配
    head_dim = q.shape[-1]
    cos_dim = cos.shape[-1]
    if head_dim != cos_dim:
        cos = cos[..., :head_dim]
        sin = sin[..., :head_dim]
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class MultiHeadLatentAttention(nn.Module):
    """
    多头潜在注意力 (Multi-head Latent Attention - MLA)
    
    核心创新:
    1. 低秩KV压缩 - 大幅减少推理时的KV缓存
    2. 低秩Query压缩 - 进一步减少内存
    3. 解耦的RoPE编码 - 分别处理位置信息和内容信息
    
    论文公式:
    - c_t^KV = W^DKV * h_t                    # KV压缩
    - k_t^C = W^UK * c_t^KV                   # Key升维
    - v_t^C = W^UV * c_t^KV                   # Value升维
    - k_t^R = RoPE(W^KR * h_t)                # 解耦的RoPE Key
    - q_t^C = W^UQ * RMSNorm(W^DQ * h_t)      # Query压缩和升维
    - q_t^R = RoPE(W^QR * RMSNorm(W^DQ * h_t)) # 解耦的RoPE Query
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.q_lora_rank = config.q_lora_rank
        
        self.head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.scaling = 1.0 / math.sqrt(self.head_dim)
        
        # ========== Query压缩和升维 ==========
        # W^DQ: 压缩Query到潜在空间
        self.q_down_proj = nn.Linear(self.hidden_size, self.q_lora_rank, bias=False)
        # W^UQ: 从潜在空间升维
        self.q_up_proj = nn.Linear(
            self.q_lora_rank,
            self.num_heads * self.head_dim,
            bias=False
        )
        # W^QR: 生成带RoPE的Query
        self.q_rope_proj = nn.Linear(
            self.q_lora_rank,
            self.num_heads * self.qk_rope_head_dim,
            bias=False
        )
        
        # ========== KV压缩和升维 ==========
        # W^DKV: 压缩KV到潜在空间
        self.kv_down_proj = nn.Linear(self.hidden_size, self.kv_lora_rank, bias=False)
        # W^UK: 从潜在空间升维Key
        self.k_up_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads * self.qk_nope_head_dim,
            bias=False
        )
        # W^UV: 从潜在空间升维Value
        self.v_up_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads * self.v_head_dim,
            bias=False
        )
        
        # ========== 解耦的Key RoPE ==========
        # W^KR: 生成带RoPE的Key (每个token只有一个，不是每个头)
        self.k_rope_proj = nn.Linear(
            self.hidden_size,
            self.qk_rope_head_dim,
            bias=False
        )
        
        # ========== 输出投影 ==========
        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=False
        )
        
        # ========== 归一化 ==========
        self.q_norm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
        self.kv_norm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        
        # ========== RoPE ==========
        self.rotary_emb = RotaryEmbedding(
            self.qk_rope_head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta
        )
        
        self.attention_dropout = config.attention_dropout
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        MLA前向传播
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, 1, seq_len, seq_len]
            position_ids: [batch_size, seq_len]
            past_key_value: 缓存的KV (用于推理)
            use_cache: 是否使用缓存
        
        Returns:
            attn_output: [batch_size, seq_len, hidden_size]
            present_key_value: 更新后的缓存
        """
        bsz, q_len, _ = hidden_states.size()
        
        # ========== Query压缩和升维 ==========
        # c_t^Q = W^DQ * h_t
        q_compressed = self.q_down_proj(hidden_states)
        q_compressed = self.q_norm(q_compressed)
        
        # 升维得到无RoPE的Query
        q_nope = self.q_up_proj(q_compressed)
        q_nope = q_nope.view(bsz, q_len, self.num_heads, self.head_dim)
        
        # 升维得到带RoPE的Query
        q_rope = self.q_rope_proj(q_compressed)
        q_rope = q_rope.view(bsz, q_len, self.num_heads, self.qk_rope_head_dim)
        
        # ========== KV压缩 ==========
        # c_t^KV = W^DKV * h_t
        kv_compressed = self.kv_down_proj(hidden_states)
        kv_compressed = self.kv_norm(kv_compressed)
        
        # 升维得到无RoPE的Key和Value
        k_nope = self.k_up_proj(kv_compressed)
        k_nope = k_nope.view(bsz, q_len, self.num_heads, self.qk_nope_head_dim)
        
        v = self.v_up_proj(kv_compressed)
        v = v.view(bsz, q_len, self.num_heads, self.v_head_dim)
        
        # ========== 解耦的Key RoPE ==========
        k_rope = self.k_rope_proj(hidden_states)
        k_rope = k_rope.unsqueeze(2).expand(-1, -1, self.num_heads, -1)
        
        # ========== 应用RoPE ==========
        cos, sin = self.rotary_emb(hidden_states, q_len)
        if position_ids is not None:
            max_pos = cos.size(0)
            position_ids = position_ids.clamp(0, max_pos - 1)
            cos = cos[position_ids]
            sin = sin[position_ids]
        
        q_rope, k_rope = apply_rotary_pos_emb(q_rope, k_rope, cos, sin)
        
        # ========== 拼接Query和Key ==========
        q = torch.cat([q_nope[..., :self.qk_nope_head_dim], q_rope], dim=-1)
        k = torch.cat([k_nope, k_rope], dim=-1)
        
        # ========== 处理缓存 (用于推理) ==========
        if past_key_value is not None:
            past_kv, past_k_rope = past_key_value
            kv_cache = kv_compressed
            k_rope_cache = k_rope[:, :, 0, :]
            
            # 合并缓存
            kv_cache = torch.cat([past_kv, kv_cache], dim=1)
            k_rope_cache = torch.cat([past_k_rope, k_rope_cache], dim=1)
            
            # 重新计算Key
            k_nope_cache = self.k_up_proj(kv_cache).view(
                kv_cache.size(0), -1, self.num_heads, self.qk_nope_head_dim
            )
            k_rope_cache = k_rope_cache.unsqueeze(2).expand(-1, -1, self.num_heads, -1)
            k = torch.cat([k_nope_cache, k_rope_cache], dim=-1)
            
            # 重新计算Value
            v_cache = self.v_up_proj(kv_cache).view(
                kv_cache.size(0), -1, self.num_heads, self.v_head_dim
            )
            v = v_cache
            
            if use_cache:
                present_key_value = (kv_cache, k_rope_cache[:, :, 0, :])
            else:
                present_key_value = None
        else:
            if use_cache:
                kv_cache = kv_compressed
                k_rope_cache = k_rope[:, :, 0, :]
                present_key_value = (kv_cache, k_rope_cache)
            else:
                present_key_value = None
        
        # ========== 注意力计算 ==========
        # 转置以进行注意力计算
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # 计算注意力分数
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        
        # 应用注意力掩码
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        # Softmax
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_weights = F.dropout(
            attn_weights,
            p=self.attention_dropout,
            training=self.training
        )
        
        # 应用注意力到Value
        attn_output = torch.matmul(attn_weights, v)
        
        # 合并头
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            bsz, q_len, self.num_heads * self.v_head_dim
        )
        
        # 输出投影
        attn_output = self.o_proj(attn_output)
        
        return attn_output, present_key_value
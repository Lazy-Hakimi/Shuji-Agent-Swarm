"""
枢机 (Shuji) - Multi-Token Prediction (MTP)模块
基于DeepSeekV3论文实现
"""
from typing import Optional
import torch
import torch.nn as nn

from .layers import RMSNorm


class MTPModule(nn.Module):
    """
    Multi-Token Prediction模块
    
    预测额外的未来token，提高训练效率和模型性能
    
    论文公式:
    h'^k_i = M_k[RMSNorm(h^{k-1}_i); RMSNorm(Emb(t_{i+k}))]
    h^k_{1:T-k} = TRM_k(h'^k_{1:T-k})
    P^k_{i+k+1} = OutHead(h^k_i)
    
    L_MTP = (λ/D) * Σ L^k_MTP
    """
    
    def __init__(self, config, depth: int):
        super().__init__()
        self.config = config
        self.depth = depth
        
        # 投影矩阵 M_k
        self.projection = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)
        
        # 归一化层
        self.prev_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.next_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # Transformer块
        from .layers import DeepSeekV3DecoderLayer
        self.layer = DeepSeekV3DecoderLayer(
            config,
            layer_idx=config.num_layers + depth
        )
    
    def forward(
        self,
        prev_hidden_states: torch.Tensor,
        next_token_emb: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        MTP模块前向传播
        
        Args:
            prev_hidden_states: 前一深度的表示 [bsz, seq_len, hidden_size]
            next_token_emb: 下一个token的embedding [bsz, seq_len, hidden_size]
            attention_mask: 注意力掩码
            position_ids: 位置ID
        
        Returns:
            hidden_states: 当前深度的表示
        """
        # 归一化输入
        prev_norm = self.prev_norm(prev_hidden_states)
        next_norm = self.next_norm(next_token_emb)
        
        # 拼接并投影
        combined = torch.cat([prev_norm, next_norm], dim=-1)
        hidden_states = self.projection(combined)
        
        # 通过Transformer块
        hidden_states, _, _ = self.layer(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=None,
            use_cache=False,
            training_step=True,
        )
        
        return hidden_states
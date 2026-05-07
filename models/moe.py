"""
枢机 (Shuji) - DeepSeekMoE with Auxiliary-Loss-Free Load Balancing
基于DeepSeekV3论文实现
"""
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    """
    单个专家网络 (SwiGLU)
    
    SwiGLU公式: Swish(xW1) ⊙ (xW3) W2
    """
    
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=False)  # gate
        self.w2 = nn.Linear(intermediate_size, hidden_size, bias=False)  # down
        self.w3 = nn.Linear(hidden_size, intermediate_size, bias=False)  # up
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: Swish(xW1) ⊙ (xW3) W2
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class DeepSeekMoE(nn.Module):
    """
    DeepSeekMoE with Auxiliary-Loss-Free Load Balancing
    
    核心创新:
    1. 共享专家 + 路由专家架构
    2. 无辅助损失负载均衡 - 通过偏置动态调整
    3. 序列级辅助损失 - 防止极端不平衡
    
    论文公式:
    h'_t = u_t + Σ FFN_i^(s)(u_t) + Σ g_i,t * FFN_i^(r)(u_t)
    
    无辅助损失负载均衡:
    s_i,t + b_i ∈ TopK({s_j,t + b_j}, K_r)
    
    偏置更新:
    if expert_overloaded: b_i -= γ
    if expert_underloaded: b_i += γ
    """
    
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_shared_experts = config.num_shared_experts
        self.num_routed_experts = config.num_routed_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.expert_intermediate_size = config.expert_intermediate_size
        self.aux_loss_free = config.aux_loss_free
        self.bias_update_speed = config.bias_update_speed
        self.seq_aux_loss_factor = config.seq_aux_loss_factor
        self.layer_idx = layer_idx
        
        # ========== 共享专家 ==========
        self.shared_experts = nn.ModuleList([
            Expert(self.hidden_size, self.expert_intermediate_size)
            for _ in range(self.num_shared_experts)
        ])
        
        # ========== 路由专家 ==========
        self.routed_experts = nn.ModuleList([
            Expert(self.hidden_size, self.expert_intermediate_size)
            for _ in range(self.num_routed_experts)
        ])
        
        # ========== 路由门控 ==========
        self.gate = nn.Linear(self.hidden_size, self.num_routed_experts, bias=False)
        
        # ========== 无辅助损失负载均衡的偏置项 ==========
        if self.aux_loss_free:
            self.register_buffer("expert_biases", torch.zeros(self.num_routed_experts))
        else:
            self.expert_biases = None
        
        # ========== 用于统计专家负载 ==========
        self.register_buffer("expert_loads", torch.zeros(self.num_routed_experts))
        self.register_buffer("num_calls", torch.tensor(0))
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        training_step: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        DeepSeekMoE前向传播
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            training_step: 是否为训练步骤(用于更新偏置)
        
        Returns:
            output: [batch_size, seq_len, hidden_size]
            aux_loss: 辅助损失(仅在训练时)
        """
        bsz, seq_len, _ = hidden_states.size()
        hidden_states_flat = hidden_states.view(-1, self.hidden_size)
        num_tokens = hidden_states_flat.size(0)
        
        # ========== 共享专家输出 ==========
        shared_output = sum(expert(hidden_states_flat) for expert in self.shared_experts)
        
        # ========== 计算路由分数 ==========
        router_logits = self.gate(hidden_states_flat)
        
        # Sigmoid激活
        affinity_scores = torch.sigmoid(router_logits)
        
        # ========== 应用偏置进行路由决策 ==========
        if self.aux_loss_free and self.training:
            routing_scores = affinity_scores + self.expert_biases.unsqueeze(0)
        else:
            routing_scores = affinity_scores
        
        # Top-K路由
        topk_scores, topk_indices = torch.topk(
            routing_scores,
            self.num_experts_per_tok,
            dim=-1
        )
        
        # 归一化门控值
        topk_affinity = torch.gather(affinity_scores, 1, topk_indices)
        gating_values = topk_affinity / (topk_affinity.sum(dim=-1, keepdim=True) + 1e-9)
        
        # ========== 计算专家输出 ==========
        routed_output = torch.zeros_like(hidden_states_flat)
        
        for i in range(self.num_experts_per_tok):
            expert_indices = topk_indices[:, i]
            expert_gates = gating_values[:, i:i+1]
            
            for expert_idx in range(self.num_routed_experts):
                mask = (expert_indices == expert_idx)
                if mask.any():
                    expert_input = hidden_states_flat[mask]
                    expert_output = self.routed_experts[expert_idx](expert_input)
                    routed_output[mask] += expert_gates[mask] * expert_output
        
        # ========== 合并输出 ==========
        output = shared_output + routed_output
        output = output.view(bsz, seq_len, self.hidden_size)
        
        # ========== 计算辅助损失 ==========
        aux_loss = None
        if self.training:
            # 序列级辅助损失
            if self.seq_aux_loss_factor > 0:
                aux_loss = self._compute_seq_aux_loss(
                    affinity_scores, topk_indices, seq_len
                )
            
            # 更新偏置
            if self.aux_loss_free and training_step:
                self._update_biases(topk_indices, num_tokens)
        
        return output, aux_loss
    
    def _compute_seq_aux_loss(
        self,
        affinity_scores: torch.Tensor,
        topk_indices: torch.Tensor,
        seq_len: int
    ) -> torch.Tensor:
        """计算序列级辅助损失"""
        num_tokens = affinity_scores.size(0)
        batch_size = num_tokens // seq_len
        
        # 计算每个token的路由概率
        router_prob = affinity_scores / (affinity_scores.sum(dim=-1, keepdim=True) + 1e-9)
        
        # 计算每个专家的负载
        expert_mask = torch.zeros_like(affinity_scores).scatter_(1, topk_indices, 1.0)
        
        # 按序列计算
        aux_loss = 0.0
        for b in range(batch_size):
            start = b * seq_len
            end = (b + 1) * seq_len
            
            seq_router_prob = router_prob[start:end]
            seq_expert_mask = expert_mask[start:end]
            
            # 每个专家的平均亲和度
            f_i = seq_expert_mask.float().mean(dim=0)
            # 每个专家的路由概率
            P_i = seq_router_prob.mean(dim=0)
            
            # 序列级平衡损失
            seq_aux_loss = self.num_routed_experts * (f_i * P_i).sum()
            aux_loss += seq_aux_loss
        
        aux_loss = aux_loss / batch_size * self.seq_aux_loss_factor
        return aux_loss
    
    def _update_biases(self, topk_indices: torch.Tensor, num_tokens: int):
        """更新专家偏置以实现负载均衡"""
        with torch.no_grad():
            # 计算每个专家的实际负载
            expert_counts = torch.bincount(
                topk_indices.flatten(),
                minlength=self.num_routed_experts
            ).float()
            
            # 期望负载 (均匀分布)
            expected_load = num_tokens * self.num_experts_per_tok / self.num_routed_experts
            
            # 更新偏置: 过载的减少偏置，欠载的增加偏置
            for i in range(self.num_routed_experts):
                if expert_counts[i] > expected_load:
                    self.expert_biases[i] -= self.bias_update_speed
                elif expert_counts[i] < expected_load:
                    self.expert_biases[i] += self.bias_update_speed
            
            # 记录负载统计
            self.expert_loads += expert_counts
            self.num_calls += 1
    
    def get_expert_load_stats(self) -> dict:
        """获取专家负载统计"""
        if self.num_calls.item() == 0:
            return {
                "avg_loads": [0.0] * self.num_routed_experts,
                "current_biases": self.expert_biases.tolist() if self.aux_loss_free else [],
            }
        
        avg_loads = (self.expert_loads / self.num_calls).tolist()
        return {
            "avg_loads": avg_loads,
            "current_biases": self.expert_biases.tolist() if self.aux_loss_free else [],
            "total_calls": self.num_calls.item(),
        }
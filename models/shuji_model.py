"""
枢机 (Shuji) - 主模型架构
整合DeepSeekV3Mini与多智能体能力
"""
from typing import Optional, List, Tuple, Dict
import torch
import torch.nn as nn

from .layers import RMSNorm, DeepSeekV3DecoderLayer
from .mtp import MTPModule


class ShujiPreTrainedModel(nn.Module):
    """枢机预训练模型基类"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_hidden_layers = config.num_layers
        
        # 词嵌入
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # 解码器层
        self.layers = nn.ModuleList([
            DeepSeekV3DecoderLayer(config, layer_idx)
            for layer_idx in range(config.num_layers)
        ])
        
        # 最终归一化
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # 输出头
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        std = self.config.initializer_range
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=std)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=std)
    
    def get_input_embeddings(self):
        return self.embed_tokens
    
    def set_input_embeddings(self, value):
        self.embed_tokens = value
    
    def get_output_embeddings(self):
        return self.lm_head
    
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings


class ShujiModel(ShujiPreTrainedModel):
    """枢机基础模型 (用于推理)"""
    
    def __init__(self, config):
        super().__init__(config)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """基础模型前向传播"""
        batch_size, seq_length = input_ids.shape
        
        # 创建位置ID
        if position_ids is None:
            if past_key_values is not None:
                past_length = past_key_values[0][0].size(1)
                position_ids = torch.arange(
                    past_length, seq_length + past_length,
                    dtype=torch.long, device=input_ids.device
                ).unsqueeze(0).expand(batch_size, -1)
            else:
                position_ids = torch.arange(
                    seq_length, dtype=torch.long, device=input_ids.device
                ).unsqueeze(0).expand(batch_size, -1)
        
        # 词嵌入
        inputs_embeds = self.embed_tokens(input_ids)
        
        # 创建因果注意力掩码
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length), dtype=torch.bool, device=input_ids.device
            )
        
        causal_mask = self._prepare_causal_attention_mask(
            attention_mask, input_ids.shape, inputs_embeds.dtype, past_key_values
        )
        
        # 通过解码器层
        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        next_cache = [] if use_cache else None
        
        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            
            past_key_value = past_key_values[idx] if past_key_values is not None else None
            
            # 梯度检查点
            if self.training and self.config.use_gradient_checkpointing:
                from torch.utils.checkpoint import checkpoint
                layer_outputs = checkpoint(
                    decoder_layer,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_value,
                    use_cache,
                    False,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states=hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                    training_step=False,
                )
            
            hidden_states = layer_outputs[0]
            
            if use_cache:
                next_cache.append(layer_outputs[1])
        
        # 最终归一化
        hidden_states = self.norm(hidden_states)
        
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        
        if return_dict:
            return {
                "last_hidden_state": hidden_states,
                "past_key_values": next_cache,
                "hidden_states": all_hidden_states,
            }
        return hidden_states
    
    def _prepare_causal_attention_mask(
        self,
        attention_mask: torch.Tensor,
        input_shape: Tuple[int, ...],
        dtype: torch.dtype,
        past_key_values: Optional[List] = None,
    ) -> torch.Tensor:
        """准备因果注意力掩码"""
        batch_size, seq_length = input_shape
        
        # 创建因果掩码
        causal_mask = torch.triu(
            torch.full((seq_length, seq_length), float('-inf'), device=attention_mask.device),
            diagonal=1
        )
        
        # 添加batch维度
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        
        # 结合padding掩码
        if attention_mask is not None:
            padding_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            padding_mask = (1.0 - padding_mask.float()) * float('-inf')
            causal_mask = causal_mask + padding_mask
        
        return causal_mask.to(dtype)


class ShujiForCausalLM(ShujiModel):
    """
    枢机因果语言模型
    
    支持:
    - 标准因果语言建模
    - Multi-Token Prediction (MTP)
    - KV缓存优化
    """
    
    def __init__(self, config):
        super().__init__(config)
        
        # MTP模块
        self.mtp_modules = nn.ModuleList([
            MTPModule(config, depth=i+1)
            for i in range(config.num_mtp_tokens)
        ])
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List] = None,
        use_cache: bool = False,
        labels: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        compute_mtp_loss: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        因果语言模型前向传播
        
        Args:
            input_ids: 输入token IDs
            attention_mask: 注意力掩码
            position_ids: 位置ID
            past_key_values: 缓存的KV
            use_cache: 是否使用缓存
            labels: 标签(用于训练)
            output_hidden_states: 是否输出隐藏状态
            return_dict: 是否返回字典
            compute_mtp_loss: 是否计算MTP损失
        
        Returns:
            包含loss、logits等的字典
        """
        batch_size, seq_length = input_ids.shape
        
        # 基础模型前向传播
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_hidden_states=True,
            return_dict=True,
        )
        
        hidden_states = outputs["last_hidden_state"]
        all_hidden_states = outputs["hidden_states"]
        
        # 主模型预测
        logits = self.lm_head(hidden_states)
        
        # 计算损失
        loss = None
        mtp_losses = []
        
        if labels is not None:
            # 主损失
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1)
            )
            
            # MTP损失
            if compute_mtp_loss and self.config.num_mtp_tokens > 0:
                for mtp_idx, mtp_module in enumerate(self.mtp_modules):
                    target_positions = mtp_idx + 2
                    if target_positions <= seq_length:
                        target_token_ids = labels[..., target_positions-1:].contiguous()
                        
                        if mtp_idx == 0:
                            prev_hidden = hidden_states
                        else:
                            prev_hidden = mtp_hidden
                        
                        # 获取下一个token的embedding
                        safe_token_ids = target_token_ids.clone()
                        safe_token_ids[safe_token_ids == -100] = 0
                        next_token_emb = self.embed_tokens(safe_token_ids)
                        
                        # 截断到相同长度
                        min_len = min(prev_hidden.size(1), next_token_emb.size(1))
                        prev_hidden_trunc = prev_hidden[:, :min_len, :]
                        next_token_emb_trunc = next_token_emb[:, :min_len, :]
                        
                        # MTP模块前向传播
                        if attention_mask is not None:
                            mtp_attention_mask = attention_mask[:, :min_len]
                        else:
                            mtp_attention_mask = None
                        
                        mtp_hidden = mtp_module(
                            prev_hidden_states=prev_hidden_trunc,
                            next_token_emb=next_token_emb_trunc,
                            attention_mask=mtp_attention_mask,
                            position_ids=position_ids[:, :min_len] if position_ids is not None else None,
                        )
                        
                        # MTP预测
                        mtp_logits = self.lm_head(mtp_hidden)
                        
                        # MTP损失
                        mtp_shift_logits = mtp_logits[..., :-1, :].contiguous()
                        mtp_shift_labels = target_token_ids[..., 1:].contiguous()
                        
                        if mtp_shift_logits.numel() > 0:
                            mtp_loss = loss_fct(
                                mtp_shift_logits.view(-1, self.config.vocab_size),
                                mtp_shift_labels.view(-1)
                            )
                            mtp_losses.append(mtp_loss)
                
                # 合并MTP损失
                if mtp_losses:
                    avg_mtp_loss = sum(mtp_losses) / len(mtp_losses)
                    loss = loss + self.config.mtp_loss_weight * avg_mtp_loss
        
        if return_dict:
            return {
                "loss": loss,
                "logits": logits,
                "past_key_values": outputs["past_key_values"],
                "hidden_states": all_hidden_states,
                "mtp_losses": mtp_losses,
            }
        return logits
    
    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[List] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """为生成准备输入"""
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
            "use_cache": True,
        }
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
        num_return_sequences: int = 1,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        自回归生成
        
        Args:
            input_ids: 输入token IDs
            max_length: 最大序列长度
            max_new_tokens: 最大新生成token数
            temperature: 采样温度
            top_p: nucleus sampling参数
            top_k: top-k采样参数
            do_sample: 是否采样
            num_return_sequences: 返回序列数
            pad_token_id: padding token ID
            eos_token_id: 结束token ID
        
        Returns:
            生成的token IDs
        """
        batch_size = input_ids.shape[0]
        input_length = input_ids.shape[1]
        device = input_ids.device
        
        if pad_token_id is None:
            pad_token_id = 0
        if eos_token_id is None:
            eos_token_id = 2
        
        # 复制输入以生成多个序列
        if num_return_sequences > 1:
            input_ids = input_ids.unsqueeze(1).expand(-1, num_return_sequences, -1)
            input_ids = input_ids.reshape(batch_size * num_return_sequences, -1)
            batch_size = batch_size * num_return_sequences
        
        # 确定最大生成长度
        if max_new_tokens is not None:
            max_length = input_length + max_new_tokens
        elif max_length is None:
            max_length = input_length + 100
        
        # 初始化
        past_key_values = None
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=device)
        
        for _ in range(max_length - input_length):
            # 准备输入
            model_inputs = self.prepare_inputs_for_generation(
                input_ids, past_key_values, None
            )
            
            # 前向传播
            outputs = self.forward(**model_inputs, compute_mtp_loss=False)
            
            # 获取下一个token的logits
            next_token_logits = outputs["logits"][:, -1, :]
            
            # 应用温度
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            
            # 采样
            if do_sample:
                # Top-k过滤
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Top-p过滤
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    for batch_idx in range(batch_size):
                        indices_to_remove = sorted_indices[batch_idx][sorted_indices_to_remove[batch_idx]]
                        next_token_logits[batch_idx, indices_to_remove] = float('-inf')
                
                # 采样
                probs = torch.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                # 贪婪解码
                next_tokens = torch.argmax(next_token_logits, dim=-1)
            
            # 更新未完成序列
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
            
            # 添加到序列
            input_ids = torch.cat([input_ids, next_tokens.unsqueeze(-1)], dim=-1)
            
            # 更新KV缓存
            past_key_values = outputs["past_key_values"]
            
            # 检查是否完成
            unfinished_sequences = unfinished_sequences.mul(next_tokens.ne(eos_token_id).long())
            
            if unfinished_sequences.max() == 0:
                break
        
        return input_ids
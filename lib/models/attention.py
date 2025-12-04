import numpy as np
import torch
from torch import nn
from lib.models.new_containers import Module
import torch.nn.functional as F
import os
import logging

class ScaledDotProductAttention(nn.Module):
    '''
    Scaled dot-product attention
    '''
    def __init__(self, d_model, d_k, d_v, h):
        super(ScaledDotProductAttention, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc_q.weight)
        nn.init.xavier_uniform_(self.fc_k.weight)
        nn.init.xavier_uniform_(self.fc_v.weight)
        nn.init.xavier_uniform_(self.fc_o.weight)
        nn.init.constant_(self.fc_q.bias, 0)
        nn.init.constant_(self.fc_k.bias, 0)
        nn.init.constant_(self.fc_v.bias, 0)
        nn.init.constant_(self.fc_o.bias, 0)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]
        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)
        att = torch.matmul(q, k) / np.sqrt(self.d_k)
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -float('inf'))
        att = torch.softmax(att, -1)
        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)
        out = self.fc_o(out)
        return out


class DynamicMemoryUpdater(nn.Module):

    def __init__(self, d_model, m, core_mem_dim, num_heads_updater=4, per_slot_proj_dim_factor=4,
                 top_k=4, router_noise_std=1.0):
        super().__init__()
        self.m = m
        self.d_model = d_model
        self.core_mem_dim = core_mem_dim
        self.num_heads_updater = num_heads_updater
        assert self.core_mem_dim % num_heads_updater == 0, "core_mem_dim must be divisible by num_heads_updater"
        self.head_dim = self.core_mem_dim // num_heads_updater

        self.top_k = top_k
        self.router_noise_std = router_noise_std
        self.gumbel_tau = 2.0
        self.query_proj_batch = nn.Linear(d_model, self.core_mem_dim)

        _per_slot_intermediate_dim = self.core_mem_dim * per_slot_proj_dim_factor
        self.per_slot_query_proj_net = nn.Sequential(
            nn.Linear(self.core_mem_dim, _per_slot_intermediate_dim),
            nn.ReLU(),
            nn.Linear(_per_slot_intermediate_dim, self.core_mem_dim)
        )

        self.memory_bias = nn.Parameter(torch.randn(num_heads_updater, m, 1))

        self.update_net_input_dim = self.core_mem_dim * 2
        self.update_net = nn.Sequential(
            nn.LayerNorm(self.update_net_input_dim),
            nn.Linear(self.update_net_input_dim, self.core_mem_dim * 2),
            nn.ReLU(),
            nn.Linear(self.core_mem_dim * 2, self.core_mem_dim)
        )
        self.norm_output = nn.LayerNorm(self.core_mem_dim)
        self.init_weights()

    def set_gumbel_tau(self, tau):
        self.gumbel_tau = tau
    def init_weights(self):
        nn.init.xavier_uniform_(self.query_proj_batch.weight)
        nn.init.constant_(self.query_proj_batch.bias, 0)
        for layer in self.per_slot_query_proj_net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
        for layer in self.update_net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
            elif isinstance(layer, nn.LayerNorm):
                nn.init.constant_(layer.weight, 1.0)
                nn.init.constant_(layer.bias, 0.0)
        nn.init.normal_(self.memory_bias, mean=0.0, std=0.01)
        nn.init.constant_(self.norm_output.weight, 1.0)
        nn.init.constant_(self.norm_output.bias, 0.0)

    def forward(self, batch_queries, global_memory_base):
        b_s, seq_len, d_model_in = batch_queries.shape
        _, m_slots, core_dim_mem = global_memory_base.shape
        assert m_slots == self.m, f"The number of memory slots does not match: {m_slots} vs {self.m}"
        assert core_dim_mem == self.core_mem_dim, f"Memory dimensions do not match: {core_dim_mem} vs {self.core_mem_dim}"

        flat_batch_queries = batch_queries.reshape(b_s * seq_len, d_model_in)
        projected_batch_info_k = self.query_proj_batch(flat_batch_queries)
        projected_batch_info_v = projected_batch_info_k

        reshaped_global_memory = global_memory_base.squeeze(0)
        projected_q_slots = self.per_slot_query_proj_net(reshaped_global_memory)
        q_mem_projected = projected_q_slots.unsqueeze(0)

        q_mem = q_mem_projected.view(1, self.m, self.num_heads_updater, self.head_dim).permute(0, 2, 1, 3).squeeze(0)
        k_batch = projected_batch_info_k.view(b_s * seq_len, self.num_heads_updater, self.head_dim).permute(1, 0, 2)
        v_batch = projected_batch_info_v.view(b_s * seq_len, self.num_heads_updater, self.head_dim).permute(1, 0, 2)

        attn_scores = torch.matmul(q_mem, k_batch.transpose(-1, -2)) / np.sqrt(self.head_dim)
        attn_scores_for_diversity = attn_scores + self.memory_bias * 5.0

        gate_logits = attn_scores_for_diversity

        if self.training and self.router_noise_std > 0:
            noise = torch.randn_like(gate_logits) * self.router_noise_std
            gate_logits = gate_logits + noise

        if self.training:
            gumbel_input_logits = gate_logits.permute(0, 2, 1)
            k_gating_decisions = []
            for _ in range(self.top_k):
                k_gating_decisions.append(F.gumbel_softmax(gumbel_input_logits, tau=self.gumbel_tau, hard=True, dim=-1))
            gating_decision_combined = torch.stack(k_gating_decisions, dim=0).sum(dim=0).clamp(max=1.0)
            gating_weights = gating_decision_combined.permute(0, 2, 1)
        else:
            top_k_logits, top_k_indices = torch.topk(gate_logits, self.top_k, dim=1)
            sparse_gates = torch.zeros_like(gate_logits)
            sparse_gates.scatter_(1, top_k_indices, 1)
            gating_weights = sparse_gates

        attended_batch_context_per_head = torch.einsum('hmn,hnd->hmd', gating_weights, v_batch)

        attended_batch_context_permuted = attended_batch_context_per_head.permute(1, 0, 2)
        attended_batch_context = attended_batch_context_permuted.contiguous().view(1, self.m, self.core_mem_dim)

        update_net_input = torch.cat((global_memory_base, attended_batch_context), dim=-1)
        dynamic_adjustment_proposal = self.update_net(update_net_input)
        dynamic_memory_component = self.norm_output(dynamic_adjustment_proposal)

        load_fraction = gating_weights.sum(dim=-1).mean(dim=0)
        return dynamic_memory_component, attn_scores_for_diversity, gate_logits, load_fraction

class ScaledDotProductAttentionMemory(nn.Module):

    def __init__(self, d_model, d_k, d_v, h, m,
                 core_mem_dim_k=None,
                 core_mem_dim_v=None,
                 num_heads_updater_k=4,
                 num_heads_updater_v=4,
                 top_k=4,
                 router_noise_std=1.0
                 ):
        super(ScaledDotProductAttentionMemory, self).__init__()

        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h
        self.m = m

        self.core_mem_dim_k = core_mem_dim_k if core_mem_dim_k is not None else d_k
        self.core_mem_dim_v = core_mem_dim_v if core_mem_dim_v is not None else d_v

        self.m_k_base = nn.Parameter(torch.randn(1, m, self.core_mem_dim_k) * (1. / np.sqrt(self.core_mem_dim_k)))
        self.m_v_base = nn.Parameter(torch.randn(1, m, self.core_mem_dim_v) * (1. / np.sqrt(self.core_mem_dim_v)))

        self.dynamic_updater_k = DynamicMemoryUpdater(d_model, m, self.core_mem_dim_k, num_heads_updater=num_heads_updater_k, top_k=top_k,
                                                      router_noise_std=router_noise_std)
        self.dynamic_updater_v = DynamicMemoryUpdater(d_model, m, self.core_mem_dim_v, num_heads_updater=num_heads_updater_v, top_k=top_k,
                                                      router_noise_std=router_noise_std)

        self.combined_mem_proj_k = nn.Linear(self.core_mem_dim_k, h * d_k)
        self.combined_mem_proj_v = nn.Linear(self.core_mem_dim_v, h * d_v)

        self.fusion_gate_k = nn.Linear(self.core_mem_dim_k * 2, 2)
        self.fusion_gate_v = nn.Linear(self.core_mem_dim_v * 2, 2)
        self.init_weights()

    def init_weights(self):
        for layer in [self.fc_q, self.fc_k, self.fc_v, self.fc_o, self.combined_mem_proj_k, self.combined_mem_proj_v]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0)
        for layer in [self.fusion_gate_k, self.fusion_gate_v]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0)
    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        dynamic_k_component, attn_weights_k_for_diversity, gate_logits_k, load_fraction_k = self.dynamic_updater_k(queries, self.m_k_base)
        dynamic_v_component, _, _, _ = self.dynamic_updater_v(queries, self.m_v_base)

        gates_info_dict = {}

        gate_input_k = torch.cat((self.m_k_base, dynamic_k_component), dim=-1)
        gate_logits_k = self.fusion_gate_k(gate_input_k)
        weights_k = torch.softmax(gate_logits_k, dim=-1)
        w_keep_k = weights_k[..., 0:1]
        w_new_k = weights_k[..., 1:2]
        combined_m_k = w_keep_k * self.m_k_base + w_new_k * dynamic_k_component

        gate_input_v = torch.cat((self.m_v_base, dynamic_v_component), dim=-1)
        gate_logits_v = self.fusion_gate_v(gate_input_v)
        weights_v = torch.softmax(gate_logits_v, dim=-1)
        w_keep_v = weights_v[..., 0:1]
        w_new_v = weights_v[..., 1:2]
        combined_m_v = w_keep_v * self.m_v_base + w_new_v * dynamic_v_component

        projected_m_k = self.combined_mem_proj_k(combined_m_k)
        projected_m_v = self.combined_mem_proj_v(combined_m_v)

        mem_k_ready = projected_m_k.expand(b_s, -1, -1).view(b_s, self.m, self.h, self.d_k).permute(0, 2, 3, 1)
        mem_v_ready = projected_m_v.expand(b_s, -1, -1).view(b_s, self.m, self.h, self.d_v).permute(0, 2, 1, 3)

        q_proj = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)
        k_input_proj = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)
        v_input_proj = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)

        k_all = torch.cat([k_input_proj, mem_k_ready], dim=3)
        v_all = torch.cat([v_input_proj, mem_v_ready], dim=2)

        att_scores = torch.matmul(q_proj, k_all) / np.sqrt(self.d_k)
        if attention_mask is not None:
            mem_mask_part = torch.zeros(b_s, self.h, nq, self.m, dtype=torch.bool, device=attention_mask.device)
            full_attention_mask = torch.cat([attention_mask, mem_mask_part], dim=3)
            att_scores = att_scores.masked_fill(full_attention_mask, -float('inf'))
        if attention_weights is not None:
            mem_weights_part = torch.ones(b_s, self.h, nq, self.m, device=attention_weights.device)
            full_attention_weights = torch.cat([attention_weights, mem_weights_part], dim=3)
            att_scores = att_scores * full_attention_weights
        att_probs = torch.softmax(att_scores, dim=-1)
        out = torch.matmul(att_probs, v_all).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)
        out = self.fc_o(out)

        m_k_base_clone_for_loss = self.m_k_base.clone()
        with torch.no_grad():
            gates_info_dict['w_keep_k_mean'] = w_keep_k.mean().item()
            gates_info_dict['w_new_k_mean'] = w_new_k.mean().item()
            gates_info_dict['w_keep_v_mean'] = w_keep_v.mean().item()
            gates_info_dict['w_new_v_mean'] = w_new_v.mean().item()

        return out, m_k_base_clone_for_loss, gates_info_dict, attn_weights_k_for_diversity, gate_logits_k, load_fraction_k

    def set_gumbel_tau(self, tau):
        if hasattr(self, 'dynamic_updater_k') and hasattr(self.dynamic_updater_k, 'set_gumbel_tau'):
            self.dynamic_updater_k.set_gumbel_tau(tau)
        if hasattr(self, 'dynamic_updater_v') and hasattr(self.dynamic_updater_v, 'set_gumbel_tau'):
            self.dynamic_updater_v.set_gumbel_tau(tau)

    def get_persistent_memory_parameters(self):
        return self.m_k_base, self.m_v_base


class MultiHeadAttention(Module):
    '''
    Multi-head attention layer with Dropout and Layer Normalization.
    '''
    def __init__(self, d_model, d_k, d_v, h, dropout=.1, identity_map_reordering=False, can_be_stateful=False,
                 attention_module=None, attention_module_kwargs=None):
        super(MultiHeadAttention, self).__init__()
        self.identity_map_reordering = identity_map_reordering
        if attention_module is not None:
            mem_kwargs = attention_module_kwargs or {}

            self.attention = attention_module(
                d_model=d_model, d_k=d_k, d_v=d_v, h=h,
                m=mem_kwargs.get('num_memory', 64),
                num_heads_updater_k=mem_kwargs.get('num_heads_updater_k', 4),
                num_heads_updater_v=mem_kwargs.get('num_heads_updater_v', 4),
                top_k=mem_kwargs.get('top_k', 4),
                router_noise_std=mem_kwargs.get('router_noise_std', 1.0)
            )
        else:
            self.attention = ScaledDotProductAttention(d_model=d_model, d_k=d_k, d_v=d_v, h=h)

        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        self.can_be_stateful = can_be_stateful
        if self.can_be_stateful:
            self.register_state('running_keys', torch.zeros((0, d_model)))
            self.register_state('running_values', torch.zeros((0, d_model)))

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        if self.can_be_stateful and getattr(self, '_is_stateful', False):
            self.running_keys = torch.cat([self.running_keys, keys], 1)
            keys_to_use = self.running_keys
            self.running_values = torch.cat([self.running_values, values], 1)
            values_to_use = self.running_values
        else:
            keys_to_use = keys
            values_to_use = values

        memory_info_for_loss = None
        gates_info_collected = {}
        updater_attn_weights_k_collected = None
        gate_logits_k_collected = None
        load_fraction_k_collected = None

        if self.identity_map_reordering:
            q_norm = self.layer_norm(queries)
            k_norm = self.layer_norm(keys_to_use)
            v_norm = self.layer_norm(values_to_use)
            attention_output_tuple = self.attention(q_norm, k_norm, v_norm, attention_mask, attention_weights)
        else:
            attention_output_tuple = self.attention(queries, keys_to_use, values_to_use, attention_mask, attention_weights)

        if isinstance(self.attention, ScaledDotProductAttentionMemory):
            if isinstance(attention_output_tuple, tuple) and len(attention_output_tuple) == 6:
                attn_out, memory_info_for_loss, gates_info_collected, \
                    updater_attn_weights_k_collected, gate_logits_k_collected, load_fraction_k_collected = attention_output_tuple
            else:
                attn_out = attention_output_tuple[0] if isinstance(attention_output_tuple, tuple) else attention_output_tuple
        elif isinstance(attention_output_tuple, tuple):
            attn_out = attention_output_tuple[0]
        else:
            attn_out = attention_output_tuple

        if self.identity_map_reordering:
            out = queries + self.dropout(torch.relu(attn_out))
        else:
            out = self.layer_norm(queries + self.dropout(attn_out))

        return out, memory_info_for_loss, gates_info_collected, updater_attn_weights_k_collected, gate_logits_k_collected, load_fraction_k_collected
    def set_gumbel_tau(self, tau):
        if hasattr(self, 'attention') and hasattr(self.attention, 'set_gumbel_tau'):
            self.attention.set_gumbel_tau(tau)


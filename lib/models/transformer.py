# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
"""
DETR Transformer class.
Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
import matplotlib.pyplot as plt
from lib.models.attention import MultiHeadAttention, ScaledDotProductAttentionMemory,ScaledDotProductAttention
from lib.config import config
import logging
import os
logger = logging.getLogger(__name__)

class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_memory=0,num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False,
                 top_k=4, router_noise_std=1.0
                 ):
        super().__init__()

        attention_module = ScaledDotProductAttentionMemory if num_memory > 0 else ScaledDotProductAttention
        attention_module_kwargs = {}
        if num_memory > 0:
            attention_module_kwargs = {
                'num_memory': num_memory,
                'top_k': top_k,
                'router_noise_std': router_noise_std
            }

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                            dropout, activation, normalize_before,
                                            attention_module=attention_module,
                                            attention_module_kwargs=attention_module_kwargs)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(d_model, encoder_layer, num_encoder_layers, encoder_norm,)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(
            decoder_layer=decoder_layer, num_layers=num_decoder_layers,
            norm=decoder_norm, return_intermediate=return_intermediate_dec)

        self.d_model = d_model
        self.nhead = nhead
        self.query_proj = nn.Sequential(
            nn.Linear(1280, 156),
            nn.ReLU()
        )
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def set_gumbel_tau(self, tau):
        if hasattr(self, 'encoder') and hasattr(self.encoder, 'set_gumbel_tau'):
            self.encoder.set_gumbel_tau(tau)
    def forward(self, src, mask, query_embed, pos_embed):
        # flatten NxCxHxW to HWxNxC
        bs, hw, c = src.shape
        src = src.permute(1, 0, 2)  # [c, bs, hw]
        pos_embed = pos_embed.permute(1, 0, 2)

        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)

        if mask is not None:
            mask = mask.flatten(1)

            # --- 接收新的返回值 ---
        memory, last_global_persistent_m_k, all_encoder_gates_info, \
        avg_updater_attn_weights_k, gate_logits_k, load_fractions_k = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)

        tgt = self.query_proj(memory.permute(1, 2, 0)).permute(2, 0, 1)
        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                              pos=pos_embed, query_pos=query_embed)

        output = {
            'hs': hs.transpose(1, 2),
            'memory_k': last_global_persistent_m_k,
            'encoder_gates_info': all_encoder_gates_info,
            'updater_attn_weights_k': avg_updater_attn_weights_k,
            'gate_logits_k': gate_logits_k, 
            'load_fractions_k': load_fractions_k,
        }

        return output
class TransformerEncoder(nn.Module):

    def __init__(self, d_model, encoder_layer, num_layers, norm=None,):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.MLP = nn.Sequential(
            nn.Linear(num_layers * d_model, num_layers * d_model),
            nn.LeakyReLU(),
            nn.Linear(num_layers * d_model, d_model),
            nn.LeakyReLU()
        )

    def set_gumbel_tau(self, tau):
        for layer in self.layers:
            if hasattr(layer, 'set_gumbel_tau'):
                layer.set_gumbel_tau(tau)
    def forward(self, src, mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None, pos: Optional[Tensor] = None):
        output = src
        outputs = []
        all_gates_info = []
        last_seen_global_m_k = None

        all_updater_attn_weights_k = []
        all_gate_logits_k = []
        all_load_fractions_k = []

        for i, layer in enumerate(self.layers):
            output, memory_k, layer_gates_info, updater_attn_weights_k, \
                gate_logits_k, load_fraction_k = layer(
                output, src_mask=mask, src_key_padding_mask=src_key_padding_mask, pos=pos
            )
            outputs.append(output)

            if memory_k is not None:
                last_seen_global_m_k = memory_k
            if layer_gates_info:
                all_gates_info.append({f"layer_{i}": layer_gates_info})

            if updater_attn_weights_k is not None:
                all_updater_attn_weights_k.append(updater_attn_weights_k)
            if gate_logits_k is not None:
                all_gate_logits_k.append(gate_logits_k)
            if load_fraction_k is not None:
                all_load_fractions_k.append(load_fraction_k)

        outputs = self.MLP(torch.cat(outputs, -1))
        output = 0.2 * outputs + output
        if self.norm is not None:
            output = self.norm(output)

        final_updater_attn_weights_k = torch.stack(all_updater_attn_weights_k, dim=0) \
            if all_updater_attn_weights_k else None
        final_gate_logits_k = torch.stack(all_gate_logits_k, dim=0) \
            if all_gate_logits_k else None
        final_load_fractions_k = torch.stack(all_load_fractions_k, dim=0) \
            if all_load_fractions_k else None

        return output, last_seen_global_m_k, all_gates_info, final_updater_attn_weights_k, final_gate_logits_k, final_load_fractions_k


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,attention_module=ScaledDotProductAttentionMemory,
                 attention_module_kwargs=None):
        super().__init__()

        self.self_attn = MultiHeadAttention(
            d_model=d_model, d_k=d_model, d_v=d_model,
            h=nhead, dropout=dropout,
            attention_module=attention_module,
            attention_module_kwargs=attention_module_kwargs
        )
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def set_gumbel_tau(self, tau):
        if hasattr(self, 'self_attn') and hasattr(self.self_attn, 'set_gumbel_tau'):
            self.self_attn.set_gumbel_tau(tau)
    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2, global_m_k, layer_gates_info, updater_attn_weights_k, \
            gate_logits_k, load_fraction_k = self.self_attn(
            q.permute(1, 0, 2), k.permute(1, 0, 2), values=src.permute(1, 0, 2),
            attention_mask=src_mask
        )
        src2=src2.permute(1, 0, 2)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, global_m_k, layer_gates_info, updater_attn_weights_k, gate_logits_k, load_fraction_k

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(**kwargs):
    return Transformer(
        d_model=kwargs['hidden_dim'],
        dropout=kwargs['dropout'],
        nhead=kwargs['nheads'],
        dim_feedforward=kwargs['dim_feedforward'],
        num_encoder_layers=kwargs['enc_layers'],
        num_decoder_layers=kwargs['dec_layers'],
        normalize_before=kwargs['pre_norm'],
        return_intermediate_dec=True,
        num_memory=config.MODEL.EXTRA.NUM_MEMORY,
        top_k=config.MODEL.EXTRA.TOP_K,
        router_noise_std=config.MODEL.EXTRA.ROUTER_NOISE_STD
    )


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from lib.models.transformer import build_transformer
from lib.models.backbone import build_backbone

import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k)
                                    for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class PoseTransformer(nn.Module):

    def __init__(self, cfg, backbone, transformer, **kwargs):
        super(PoseTransformer, self).__init__()
        extra = cfg.MODEL.EXTRA
        self.num_queries = 256
        self.transformer = transformer
        self.backbone = backbone
        hidden_dim = transformer.d_model
        self.class_embed_wflw = nn.Linear(hidden_dim, 68 + 57)
        self.kpt_embed = MLP(hidden_dim, hidden_dim, 2, 3)
        self.query_embed = nn.Embedding(156, hidden_dim)
        self.aux_loss = extra.AUX_LOSS

        self.num_feature_levels = extra.NUM_FEATURE_LEVELS
        if self.num_feature_levels > 1:
            num_backbone_outs = len(backbone.num_channels)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(self.num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
            prior_prob = 0.01
            bias_value = -math.log((1 - prior_prob) / prior_prob)
            self.class_embed.bias.data = torch.ones(68 + 57) * bias_value
            nn.init.constant_(self.kpt_embed.layers[-1].weight.data, 0)
            nn.init.constant_(self.kpt_embed.layers[-1].bias.data, 0)
            for proj in self.input_proj:
                nn.init.xavier_uniform_(proj[0].weight, gain=1)
                nn.init.constant_(proj[0].bias, 0)
            num_pred = transformer.decoder.num_layers
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.kpt_embed = nn.ModuleList([self.kpt_embed for _ in range(num_pred)])
        else:
            self.input_proj = nn.ModuleList([nn.Sequential(
                nn.Conv2d(self.backbone.num_channels[ii], hidden_dim, kernel_size=1), nn.GroupNorm(32, hidden_dim)) for
                ii in range(len(self.backbone.num_channels))])

    def set_gumbel_tau(self, tau):
        if hasattr(self, 'transformer') and hasattr(self.transformer, 'set_gumbel_tau'):
            self.transformer.set_gumbel_tau(tau)

    def forward(self, x):
        src, poses = self.backbone(x)
        srcs = []
        for l, feat in enumerate(src):
            projected = self.input_proj[l](feat)
            srcs.append(projected)

        src_flatten = []
        pos_flatten = []
        for lvl, (src_feat, pos_feat) in enumerate(zip(srcs, poses)):
            src_flat = src_feat.flatten(2).transpose(1, 2)
            pos_flat = pos_feat.flatten(2).transpose(1, 2)
            src_flatten.append(src_flat)
            pos_flatten.append(pos_flat)
        src_flatten = torch.cat(src_flatten, 1)
        pos_flatten = torch.cat(pos_flatten, 1)
        transformer_output = self.transformer(src_flatten, None, self.query_embed.weight, pos_flatten)

        hs = transformer_output['hs']
        memory_k = transformer_output['memory_k']
        encoder_gates_info = transformer_output['encoder_gates_info']
        gate_logits_k = transformer_output.get('gate_logits_k')
        load_fractions_k = transformer_output.get('load_fractions_k')
        updater_attn_weights_k = transformer_output.get('updater_attn_weights_k')


        outputs_class = self.class_embed_wflw(hs)
        outputs_coord = self.kpt_embed(hs).sigmoid()

        out_wflw = {
            'pred_logits': outputs_class[-1],
            'pred_coords': outputs_coord[-1],
            'memory_k': memory_k,
            'encoder_gates_info': encoder_gates_info,
            'updater_attn_weights_k': updater_attn_weights_k,
            'gate_logits_k': gate_logits_k,
            'load_fractions_k': load_fractions_k,
        }

        if self.aux_loss:
            out_wflw['aux_outputs'] = self._set_aux_loss(
                outputs_class,
                outputs_coord)
        return out_wflw


    @torch.jit.unused
    def _set_aux_loss(self,
                      outputs_class,
                      outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_coords': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


def get_face_alignment_net(cfg, is_train, **kwargs):
    extra = cfg.MODEL.EXTRA

    transformer_kwargs = {
        'hidden_dim': extra.HIDDEN_DIM,
        'dropout': extra.DROPOUT,
        'nheads': extra.NHEADS,
        'dim_feedforward': extra.DIM_FEEDFORWARD,
        'enc_layers': extra.ENC_LAYERS,
        'dec_layers': extra.DEC_LAYERS,
        'pre_norm': extra.PRE_NORM,

    }
    transformer = build_transformer(**transformer_kwargs)
    pretrained = is_train and cfg.MODEL.INIT_WEIGHTS
    backbone = build_backbone(cfg, pretrained)
    model = PoseTransformer(cfg, backbone, transformer, **kwargs)

    return model
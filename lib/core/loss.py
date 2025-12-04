from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
logger = logging.getLogger(__name__)
import numpy as np

class MoETopKLoss(nn.Module):
    def __init__(self, num_experts):
        super().__init__()
        self.num_experts = num_experts

    def forward(self, gate_logits, load_fractions):
        if gate_logits is None or load_fractions is None:
            return {'loss_moce_stab': torch.tensor(0.0), 'loss_moce_load': torch.tensor(0.0)}

        # --- 1. Load Balancing Loss ---
        mean_load = load_fractions.mean()
        variance_load = load_fractions.var()
        loss_load = variance_load / (mean_load ** 2)

        # --- 2. Logit Stabilization Loss ---
        log_sum_exp_per_token = torch.logsumexp(gate_logits, dim=2)
        loss_stab = (log_sum_exp_per_token ** 2).mean()

        return {
            'loss_moce_stab': loss_stab,
            'loss_moce_load': loss_load
        }


class SetCriterion(nn.Module):

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses,config):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)
        if 'moce' in self.losses:
            num_memory_slots = config.MODEL.EXTRA.NUM_MEMORY
            logger.info(f"Initializing MoCE loss with {num_memory_slots} experts.")
            self.moce_loss = MoETopKLoss(num_experts=num_memory_slots)

    @torch.no_grad()
    def accuracy(self, output, target, topk=(1,)):
        if target.numel() == 0:
            return [torch.zeros([], device=output.device)]
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

    def loss_moce(self, outputs, targets, indices, num_joints, landmark_index, **kwargs):
        gate_logits = outputs.get('gate_logits_k')
        load_fractions = outputs.get('load_fractions_k')

        if gate_logits is None or load_fractions is None:
            return {'loss_moce_stab': torch.tensor(0.0), 'loss_moce_load': torch.tensor(0.0)}

        loss_dict = self.moce_loss(gate_logits, load_fractions)
        return loss_dict

    def loss_labels(self, outputs, targets, indices, num_joints, landmark_index, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits'][..., landmark_index] #[bs, 256, 128]

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        target_classes_o = src_idx[1].to(src_logits.device)

        target_classes = torch.full(src_logits.shape[:2], src_logits.shape[2]-1,
                                    dtype=torch.int64, device=src_logits.device)
        # default to no-kpt class, for matched ones, set to 0, ..., 16
        target_classes[tgt_idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes)
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - self.accuracy(src_logits[tgt_idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_joints, landmark_index):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        tgt_lengths = pred_logits.new_ones(pred_logits.shape[0]) * (pred_logits.shape[2]-1)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_kpts(self, outputs, targets, indices, num_joints, landmark_index, weights):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        # loss = SmoothL1Loss()

        assert 'pred_coords' in outputs
        # match gt --> pred
        src_idx = self._get_src_permutation_idx(indices)  # always (0, 1, 2, .., 16)
        tgt_idx = self._get_tgt_permutation_idx(indices)  # must be in range(0, 100)

        target_kpts = targets[src_idx]
        weights = weights[src_idx]
        src_kpts = outputs['pred_coords'][tgt_idx]
        # src_kpts = targets
        # weights = weights
        # target_kpts = outputs['pred_coords']

        loss_bbox = F.l1_loss(src_kpts, target_kpts, reduction='none') * weights

        losses = {'loss_kpts': loss_bbox.mean() * num_joints}

        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_joints, landmark_index, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'kpts': self.loss_kpts,
            'moce': self.loss_moce,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_joints, landmark_index, **kwargs)

    def forward(self, outputs, targets, target_weights, config):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices, num_joints, landmark_index = self.matcher(outputs_without_aux, targets, config)
        # indices = 2*[torch.Tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67])]

        idx = self._get_tgt_permutation_idx(indices)
        src_kpts = outputs['pred_coords'][idx].view(-1, num_joints, 2)
        pred = src_kpts * target_weights

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            if loss == 'moce':
                losses.update(self.get_loss(loss, outputs, targets, None, num_joints, None))
            elif loss == 'kpts':
                losses.update(self.get_loss(loss, outputs, targets, indices, num_joints, landmark_index, weights=target_weights))
            else:
                # labels, cardinality
                losses.update(self.get_loss(loss, outputs, targets, indices, num_joints, landmark_index))

        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices, num_joints, landmark_index = self.matcher(aux_outputs, targets, config)
                for loss in self.losses:
                    kwargs = {}
                    if loss in ['moce']:
                        continue
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    elif loss == 'kpts':
                        kwargs = {'weights': target_weights}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_joints, landmark_index, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        interpretability_metrics = {}

        if 'updater_attn_weights_k' in outputs:
            attention_weights = outputs['updater_attn_weights_k']
            interpretability_metrics['train_attention_diversity'] = \
                self._calculate_training_attention_diversity(attention_weights)
            interpretability_metrics['train_attention_stability'] = \
                self._calculate_attention_stability(attention_weights)

        if 'load_fractions_k' in outputs:
            load_fractions = outputs['load_fractions_k']
            interpretability_metrics['train_expert_utilization'] = \
                self._calculate_training_utilization(load_fractions)
            interpretability_metrics['train_load_imbalance'] = \
                self._calculate_load_imbalance(load_fractions)

        if 'encoder_gates_info' in outputs:
            gates_info = outputs['encoder_gates_info']
            interpretability_metrics['train_gate_update_ratio'] = \
                self._calculate_gate_update_behavior(gates_info)

        outputs['train_interpretability_metrics'] = interpretability_metrics
        return losses, pred.detach().cpu()

    def _calculate_training_attention_diversity(self, attention_weights):
        if attention_weights is None:
            return 0.0

        num_heads = attention_weights.shape[1]
        diversities = []

        for i in range(num_heads):
            for j in range(i + 1, num_heads):
                p = F.softmax(attention_weights[:, i].flatten(), dim=0)
                q = F.softmax(attention_weights[:, j].flatten(), dim=0)
                m = 0.5 * (p + q)
                js_div = 0.5 * (F.kl_div(p.log(), m, reduction='sum') +
                                F.kl_div(q.log(), m, reduction='sum'))
                diversities.append(js_div.item())

        return np.mean(diversities) if diversities else 0.0

    def _calculate_attention_stability(self, attention_weights):
        if attention_weights is None:
            return 0.0

        variance = attention_weights.var(dim=-1).mean().item()
        stability = 1.0 / (1.0 + variance)
        return stability

    def _calculate_training_utilization(self, load_fractions):
        if load_fractions is None:
            return 0.0

        utilization = (load_fractions > 1e-6).float().mean().item()
        return utilization * 100

    def _calculate_load_imbalance(self, load_fractions):
        if load_fractions is None:
            return 0.0
        sorted_loads = torch.sort(load_fractions.flatten())[0]
        n = sorted_loads.shape[0]
        device = sorted_loads.device
        index = torch.arange(1, n + 1, dtype=torch.float32, device=device)
        gini = (torch.sum((2 * index - n - 1) * sorted_loads) /
                (n * torch.sum(sorted_loads) + 1e-8))
        return gini.item()

    def _calculate_gate_update_behavior(self, gates_info):
        if not gates_info:
            return 0.0

        update_ratios = []
        for layer_info in gates_info:
            for layer_name, gate_stats in layer_info.items():
                if 'w_new_k_mean' in gate_stats:
                    update_ratios.append(gate_stats['w_new_k_mean'])

        return np.mean(update_ratios) if update_ratios else 0.0
# ------------------------------------------------------------------------------
# Modified from HRNet-Human-Pose-Estimation
# (https://github.com/HRNet/HRNet-Human-Pose-Estimation)
# Copyright (c) Microsoft
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging
import os
import random
import torch.nn.functional as F
import numpy as np
import torch
from torch.utils import data

from lib.core.evaluate import get_transformer_coords, compute_nme, compute_nme_io
from lib.core.inference import get_final_preds_match
from torchvision import transforms
from PIL import Image
from ..utils.vis import save_debug_images
from itertools import cycle
import datetime
from ..models.attention import ScaledDotProductAttentionMemory

logger = logging.getLogger(__name__)
def calculate_gradient_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5


def calculate_memory_utilization(outputs):
    if 'load_fractions_k' not in outputs:
        return 0.0

    load_fractions = outputs['load_fractions_k']
    if load_fractions is None:
        return 0.0

    utilization = (load_fractions > 1e-6).float().mean().item()
    return utilization * 100


def calculate_expert_usage(outputs, threshold=0.01):
    if 'load_fractions_k' not in outputs:
        return 0.0, 0.0

    load_fractions = outputs['load_fractions_k']
    if load_fractions is None:
        return 0.0, 0.0

    active_experts = (load_fractions > threshold).float().mean().item()
    load_distribution = F.softmax(load_fractions.flatten(), dim=0)
    usage_entropy = -torch.sum(load_distribution * torch.log(load_distribution + 1e-8)).item()

    return active_experts * 100, usage_entropy


def analyze_gate_behavior(outputs):
    if 'encoder_gates_info' not in outputs:
        return {}

    gates_info = outputs['encoder_gates_info']
    if not gates_info:
        return {}

    analysis = {}
    for layer_info in gates_info:
        for layer_name, gate_stats in layer_info.items():
            if 'w_keep_k_mean' in gate_stats and 'w_new_k_mean' in gate_stats:
                keep_ratio = gate_stats['w_keep_k_mean']
                new_ratio = gate_stats['w_new_k_mean']
                analysis[layer_name] = {
                    'memory_preservation_ratio': keep_ratio,
                    'memory_update_ratio': new_ratio,
                    'update_aggressiveness': new_ratio / (keep_ratio + 1e-8)
                }

    return analysis

def train(config, train_loaders, model, criterion, optimizer, epoch, start_time, writer_dict, metrics_collector=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model.train()

    total_epochs = config.TRAIN.END_EPOCH

    anneal_cfg_moce = config.LOSS.ANNEAL_CONFIG.MOCE_LOAD_LOSS
    start_w_moce = anneal_cfg_moce.START_WEIGHT
    end_w_moce = anneal_cfg_moce.END_WEIGHT
    anneal_epochs_moce = total_epochs * anneal_cfg_moce.ANNEAL_EPOCHS_RATIO

    if epoch < anneal_epochs_moce:
        progress = epoch / anneal_epochs_moce
        current_w_moce = start_w_moce - (start_w_moce - end_w_moce) * progress
    else:
        current_w_moce = end_w_moce

    for k in list(criterion.weight_dict.keys()):
        if 'loss_moce_load' in k:
            criterion.weight_dict[k] = current_w_moce

    logger.info(f"Epoch [{epoch}] Loss Weight Annealing: "
                # f"loss_memory set to {current_w_mem:.4f}, "
                f"loss_moce_load set to {current_w_moce:.4f}")

    if config.LOSS.GUMBEL_TAU_ANNEAL.ENABLED:
        tau_anneal_cfg = config.LOSS.GUMBEL_TAU_ANNEAL
        total_epochs = config.TRAIN.END_EPOCH
        start_tau = tau_anneal_cfg.START_TAU
        end_tau = tau_anneal_cfg.END_TAU
        anneal_epochs_tau = total_epochs * tau_anneal_cfg.ANNEAL_EPOCHS_RATIO

        if epoch < anneal_epochs_tau:
            progress = epoch / anneal_epochs_tau
            current_tau = start_tau - (start_tau - end_tau) * progress
        else:
            current_tau = end_tau

        if hasattr(model.module, 'set_gumbel_tau'):
            model.module.set_gumbel_tau(current_tau)
            logger.info(f"Epoch [{epoch}] Gumbel-Softmax Tau Annealing: tau set to {current_tau:.4f}")

    nme_count = 0
    nme_batch_sum = 0

    end = time.time()
    data_iter = [iter(train_loaders["AFLW"]), iter(train_loaders["WFLW"]),iter(train_loaders["300W"]), iter(train_loaders["COFW"])]

    n=config.TRAIN.LARGEST_NUM
    k=len(train_loaders.keys())
    b=config.TRAIN.BATCH_SIZE_PER_GPU
    iter_num=n//b
    for i in range(iter_num):
        tmp_loss = []
        for ii in range(len(train_loaders.keys())):
            input, target, target_weight, meta = next(data_iter[ii])
            data_time.update(time.time() - end)

            outputs = model(input)

            target = target.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True)

            loss_dict, pred = criterion(outputs, target, target_weight, config)
            pred *= config.MODEL.IMAGE_SIZE[0]
            weight_dict = criterion.weight_dict
            loss = sum(loss_dict[k] * weight_dict[k]
                           for k in loss_dict.keys() if k in weight_dict)

            preds = get_transformer_coords(pred, meta, config.MODEL.IMAGE_SIZE)

            nme_batch = compute_nme(preds, meta)
            nme_batch_sum = nme_batch_sum + np.sum(nme_batch)
            nme_count = nme_count + preds.size(0)

            # optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item(), input.size(0))
            batch_time.update(time.time() - end)

            if (i+1) % config.PRINT_FREQ == 0:
                if ii==3:
                    et = (time.time() - start_time)
                    et = str(datetime.timedelta(seconds=et))[:-7]
                    tmp_loss.append(loss.item())


                    moce_load_val = loss_dict.get('loss_moce_load', torch.tensor(0.0)).item()
                    moce_stab_val = loss_dict.get('loss_moce_stab', torch.tensor(0.0)).item()
                    msg = '[{0}]\t' \
                          'Epoch: [{1}][{2}/{3}]\t' \
                          'AFLW_L: {AFLW_L:.5f} WFLW_L: {WFLW_L:.5f}  300W_L: {W300_L:.5f} COFW_L: {COFW_L:.5f}  \t' \
                          'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                          'MoCE(Load/Cplx): {moce_l:.4f}/{moce_c:.4f}  ' \
                        .format(et,
                                epoch, i + 1, iter_num, AFLW_L=tmp_loss[0], WFLW_L=tmp_loss[1],
                                W300_L=tmp_loss[2], COFW_L=tmp_loss[3], loss=losses,
                                moce_l=moce_load_val, moce_c=moce_stab_val
                                )
                    logger.info(msg)

                    grad_norm = calculate_gradient_norm(model)
                    memory_util = calculate_memory_utilization(outputs)
                    expert_usage, usage_entropy = calculate_expert_usage(outputs)
                    gate_analysis = analyze_gate_behavior(outputs)

                    if writer_dict:
                        writer = writer_dict['writer']
                        global_steps = writer_dict['train_global_steps']
                        writer.add_scalar('train_loss/total', losses.val, global_steps)
                        ce_loss_val = loss_dict.get('loss_ce', torch.tensor(0.0)).item() * weight_dict.get('loss_ce', 0.0)
                        kpts_loss_val = loss_dict.get('loss_kpts', torch.tensor(0.0)).item() * weight_dict.get('loss_kpts', 0.0)
                        writer.add_scalar('train_loss_detail/ce_weighted', ce_loss_val, global_steps)
                        writer.add_scalar('train_loss_detail/kpts_weighted', kpts_loss_val, global_steps)
                        writer.add_scalar('MoCE_Loss/Load', moce_load_val, global_steps)
                        writer.add_scalar('MoCE_Loss/Complexity', moce_stab_val, global_steps)
                        writer.add_scalar('metrics/gradient_norm', grad_norm, global_steps)
                        writer.add_scalar('metrics/memory_utilization', memory_util, global_steps)
                        writer.add_scalar('metrics/expert_usage', expert_usage, global_steps)
                        writer.add_scalar('metrics/usage_entropy', usage_entropy, global_steps)
                        for layer_name, gate_metrics in gate_analysis.items():
                            writer.add_scalar(f'gates/{layer_name}/preservation_ratio',
                                              gate_metrics['memory_preservation_ratio'], global_steps)
                            writer.add_scalar(f'gates/{layer_name}/update_aggressiveness',
                                              gate_metrics['update_aggressiveness'], global_steps)

                        writer_dict['train_global_steps'] = global_steps + 1
                    if metrics_collector:
                        metrics_collector.epoch_metrics['memory_utilization'].append(memory_util)
                        metrics_collector.epoch_metrics['expert_usage'].append(expert_usage)
                        metrics_collector.epoch_metrics['gate_entropy'].append(usage_entropy)

                else:
                    tmp_loss.append(loss.item())
    et = (time.time() - start_time)
    et = str(datetime.timedelta(seconds=et))[:-7]
    nme = nme_batch_sum / nme_count
    msg = '{} Train Epoch {} time:{:.4f} loss:{:.4f} nme:{:.4f} nme_count:{}' \
        .format(et ,epoch, batch_time.avg, losses.avg, nme, nme_count)
    logger.info(msg)
    return losses.avg


def validate(config, dataset_name, val_loader, model, criterion, epoch, start_time, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    num_classes = config.TEST.NUM_POINTS

    model.eval()

    nme_count = 0
    nme_batch_sum = 0
    nme_batch_ip = 0
    nme_batch_io = 0
    count_failure_008 = 0
    count_failure_010 = 0
    end = time.time()
    image_size = 512

    predictions = torch.zeros((len(val_loader.dataset), num_classes, 2))
    criterion.eval()
    utilization_accumulator = []
    load_balance_accumulator = []
    specialization_accumulator = []
    expert_usage_counts = []
    attention_diversity_scores = []
    memory_effectiveness_scores = []
    with torch.no_grad():
        for i, (input, target, target_weight, meta) in enumerate(val_loader):
            # measure data time
            data_time.update(time.time() - end)
            num_images = input.size(0)
            outputs = model(input)  # [-1]
            load_fractions = outputs.get('load_fractions_k')
            gate_logits = outputs.get('gate_logits_k')

            if load_fractions is not None:
                used_experts = (load_fractions > 1e-6).float()
                batch_utilization = used_experts.mean()
                utilization_accumulator.append(batch_utilization.item())

                load_std = load_fractions.std(dim=1)
                load_mean = load_fractions.mean(dim=1)
                cv = load_std / (load_mean + 1e-8)
                batch_load_balance = 1.0 / (1.0 + cv.mean())  #
                load_balance_accumulator.append(batch_load_balance.item())

                expert_usage = (load_fractions > 1e-6).sum(dim=0)
                expert_usage_counts.append(expert_usage.float())

            if gate_logits is not None:
                gate_probs = F.softmax(gate_logits, dim=2)
                gate_entropy = -torch.sum(gate_probs * torch.log(gate_probs + 1e-8), dim=2)
                max_entropy = torch.log(torch.tensor(gate_logits.size(2)).float())
                specialization_score = 1.0 - (gate_entropy.mean() / max_entropy)
                specialization_accumulator.append(specialization_score.item())

            if 'updater_attn_weights_k' in outputs and outputs['updater_attn_weights_k'] is not None:
                attn_weights = outputs['updater_attn_weights_k']

                head_diversity = 1 - F.cosine_similarity(
                    attn_weights.mean(dim=-1).unsqueeze(1),  # [L, H, M] -> [L, 1, H, M]
                    attn_weights.mean(dim=-1).unsqueeze(2),  # [L, H, M] -> [L, H, 1, M]
                    dim=-1
                ).mean().item()
                attention_diversity_scores.append(head_diversity)

                memory_effectiveness = attn_weights.max(dim=-1)[0].mean().item()
                memory_effectiveness_scores.append(memory_effectiveness)

            target = target.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True)

            # output = outputs[dt_name]

            loss_dict, pred_ = criterion(outputs, target, target_weight, config)
            pred_ *= image_size
            weight_dict = criterion.weight_dict
            loss = sum(loss_dict[k] * weight_dict[k]
                       for k in loss_dict.keys() if k in weight_dict)

            # preds = get_transformer_coords(pred_, meta, [256,256])
            # preds_loss0 = get_transformer_coords(meta['tpts'],meta,[256,256])
            num_joints = target.shape[-2]
            preds, _, pred = get_final_preds_match(config, outputs, num_joints, meta['center'], meta['scale'], meta['rotate'])
            # del outputs
            if config.TEST.SHUFFLE:
                input_flipped = torch.flip(input, [3, ]).clone()
                outputs_flipped = model(input_flipped)  # [-1]
                preds_flipped, _, _ = get_final_preds_match(config, outputs_flipped, num_joints, meta['center'], meta['scale'], meta['rotate'], True)
                # preds_flipped = get_transformer_coords(outputs_flipped['pred_coords'].detach().cpu()*image_size,
                #                                        meta, [256, 256])
                preds_mean = (preds + preds_flipped) / 2
                # del outputs_flipped

            # NME
            nme_temp = compute_nme(preds, meta)
            # nme_temp_loss0 = compute_nme(preds_loss0, meta)
            if config.TEST.SHUFFLE:
                nme_temp_ip = compute_nme(preds_mean, meta)
                nme_temp_io = compute_nme_io(preds_mean, meta)
                nme_batch_ip += np.sum(nme_temp_ip)
                nme_batch_io += np.sum(nme_temp_io)

            # Failure Rate under different threshold
            failure_008 = (nme_temp > 0.08).sum()
            failure_010 = (nme_temp > 0.10).sum()
            count_failure_008 += failure_008
            count_failure_010 += failure_010

            nme_batch_sum += np.sum(nme_temp)
            # nme_batch_loss0 += np.sum(nme_temp_loss0)
            nme_count = nme_count + preds.shape[0]

            losses.update(loss.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            prefix = '{}_{}'.format(os.path.join(config.OUTPUT_DIR, 'validate'), i)
            # save_debug_images(config, input, meta, target, pred, output, prefix)

    nme = nme_batch_sum / nme_count
    nme_batch_ip = nme_batch_ip / nme_count
    nme_batch_io = nme_batch_io / nme_count
    failure_008_rate = count_failure_008 / nme_count
    failure_010_rate = count_failure_010 / nme_count

    if utilization_accumulator:
        avg_utilization = np.mean(utilization_accumulator) * 100  # 转换为百分比
        logger.info(f"===== Average Memory Utilization on {dataset_name}: {avg_utilization:.2f}% =====")
    else:
        avg_utilization = 0.0

    if load_balance_accumulator:
        avg_load_balance = np.mean(load_balance_accumulator) * 100
        logger.info(f"===== Load Balance Score on {dataset_name}: {avg_load_balance:.2f}% =====")
    else:
        avg_load_balance = 0.0

    if specialization_accumulator:
        avg_specialization = np.mean(specialization_accumulator) * 100
        logger.info(f"===== Specialization Score on {dataset_name}: {avg_specialization:.2f}% =====")
    else:
        avg_specialization = 0.0

    if expert_usage_counts:
        expert_usage_tensor = torch.stack(expert_usage_counts)
        avg_expert_usage = expert_usage_tensor.mean(dim=0)
        usage_std = avg_expert_usage.std()
        usage_mean = avg_expert_usage.mean()
        usage_imbalance = (usage_std / (usage_mean + 1e-8)).item()

        logger.info(f"===== Expert Usage Imbalance on {dataset_name}: {usage_imbalance:.4f} =====")
        logger.info(f"===== Expert Usage Distribution: mean={usage_mean:.2f}, std={usage_std:.2f} =====")

    else:
        usage_imbalance = 0.0

    avg_attention_diversity = np.mean(attention_diversity_scores) if attention_diversity_scores else 0.0
    avg_memory_effectiveness = np.mean(memory_effectiveness_scores) if memory_effectiveness_scores else 0.0

    et = (time.time() - start_time)
    et = str(datetime.timedelta(seconds=et))[:-7]

    msg = '[{}] Test Epoch {} time:{:.4f} loss:{:.4f} nme:{:.4f} [008]:{:.4f} ' \
          '[010]: {:.4f} nme_ip: {:.4f} nme_io: {:.4f} | Util: {:.2f}% | Balance: {:.2f}% | Special: {:.2f}% | Imbalance: {:.4f}' \
        .format(et, epoch, batch_time.avg, losses.avg, nme,
                failure_008_rate, failure_010_rate, nme_batch_ip,
                nme_batch_io, avg_utilization, avg_load_balance, avg_specialization, usage_imbalance)
    logger.info(msg)

    if writer_dict:
        writer = writer_dict['writer']
        global_steps = writer_dict['valid_global_steps']
        writer.add_scalar('valid_loss', losses.avg, global_steps)
        writer.add_scalar('valid_nme', nme, global_steps)
        writer.add_scalar(f'MoCE_Stats/{dataset_name}_Utilization', avg_utilization, global_steps)
        writer.add_scalar(f'MoCE_Stats/{dataset_name}_LoadBalance', avg_load_balance, global_steps)
        writer.add_scalar(f'MoCE_Stats/{dataset_name}_Specialization', avg_specialization, global_steps)
        writer.add_scalar(f'MoCE_Stats/{dataset_name}_UsageImbalance', usage_imbalance, global_steps)
        writer.add_scalar(f'interpretability/{dataset_name}/attention_diversity',
                          avg_attention_diversity, global_steps)
        writer.add_scalar(f'interpretability/{dataset_name}/memory_effectiveness',
                          avg_memory_effectiveness, global_steps)
        writer_dict['valid_global_steps'] = global_steps + 1

    return nme_batch_io, predictions


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0
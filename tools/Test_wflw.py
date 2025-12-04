from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import time

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import lib
from lib.config import config as cfg
from lib.config import update_config
from lib.models.matcher import build_matcher
from lib.core.loss import SetCriterion
from lib.core.function import validate
from lib.utils.utils import create_logger

from lib.dataset import get_dataset
from lib.dataset import WFLW
from torch.utils.data import DataLoader
import lib.models


def parse_args():
    parser = argparse.ArgumentParser(description='Train Face Alignment')
    parser.add_argument('--cfg', default='/persist_data/home/keyingwu/face/AM-Former-wo-memory-loss/wflw.yaml', help='experiment configuration filename', type=str)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    update_config(cfg, args)

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, 'valid')

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    # cudnn related setting
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    model = eval('lib.models.' + cfg.MODEL.NAME + '.get_face_alignment_net')(
        cfg, is_train=False
    )
    model_state_file = '/persist_data/home/keyingwu/face/exp/top_k/4/outputs/output/pose_transformer/wflw/WFLW_checkpoint.pth'
    state = torch.load(model_state_file)

    last_epoch = state.get('epoch', 0)
    best_nme = state.get('best_nme', 100)

    logger.info(f'Loaded model from epoch: {last_epoch}, best_nme: {best_nme}')

    if 'best_state_dict' in state.keys():
        state_dict = state['best_state_dict']
        logger.info('=> Using best_state_dict')
    elif 'state_dict' in state.keys():
        state_dict = state['state_dict']
        logger.info('=> Using state_dict')
    else:
        state_dict = state
        logger.info('=> Using full state as state_dict')

    if hasattr(state_dict, 'module'):
        model.load_state_dict(state_dict.module.state_dict())
    else:
        model.load_state_dict(state_dict)

    matcher = build_matcher(cfg.MODEL.LANDMARKS)
    weight_dict = {
        'loss_ce': config.LOSS.CE_LOSS_COEF,
        'loss_kpts': config.LOSS.KPT_LOSS_COEF,
        "loss_moce_load": config.LOSS.ANNEAL_CONFIG.MOCE_LOAD_LOSS.START_WEIGHT,
        "loss_moce_stab": config.LOSS.MOCE_STAB_LOSS_COEF,
    }
    criterion = SetCriterion(cfg.MODEL.LANDMARKS, matcher, weight_dict, cfg.MODEL.EXTRA.EOS_COEF, ['labels', 'kpts', 'moce'], cfg).cuda()

    model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()

    gpus = list(cfg.GPUS)

    start_time = time.time()

    logger.info("=> [Step 1/7] Evaluating on WFLW Full Testset...")
    full_wflw_dataset = WFLW(cfg, is_train=False, subset=None)
    full_wflw_loader = DataLoader(
        dataset=full_wflw_dataset,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU * len(gpus),
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )

    nme, predictions = validate(cfg, 'WFLW_Full', full_wflw_loader, model, criterion,
                                epoch=last_epoch, start_time=start_time, writer_dict=None)
    logger.info(f"WFLW Full Testset NME: {nme:.4f}")
    logger.info("-" * 60)

    wflw_subsets = ['Pose', 'Expression', 'Illumination', 'Makeup', 'Occlusion', 'Blur']
    for i, subset_name in enumerate(wflw_subsets):
        logger.info(f"=> [Step {i + 2}/7] Evaluating on WFLW Subset: {subset_name}")

        subset_dataset = WFLW(cfg, is_train=False, subset=subset_name)

        if len(subset_dataset) == 0:
            logger.warning(f"Skipping subset '{subset_name}' as it contains 0 samples after filtering.")
            continue

        subset_loader = DataLoader(
            dataset=subset_dataset,
            batch_size=cfg.TEST.BATCH_SIZE_PER_GPU * len(gpus),
            shuffle=False,
            num_workers=cfg.WORKERS,
            pin_memory=cfg.PIN_MEMORY
        )

        nme, predictions = validate(cfg, f'WFLW_{subset_name}', subset_loader, model, criterion,
                                    epoch=last_epoch, start_time=start_time, writer_dict=None)
        logger.info(f"WFLW {subset_name} Subset NME: {nme:.4f}")
        logger.info("-" * 60)

    logger.info("=> WFLW Subset Evaluation Finished.")
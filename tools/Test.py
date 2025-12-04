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
from collections import OrderedDict
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import lib
from lib.config import config as cfg
from lib.config import update_config
from lib.models.matcher import build_matcher
from lib.core.loss import SetCriterion
from lib.core.function import validate
from lib.utils.utils import create_logger, model_key_helper

from lib.dataset import get_dataset
from lib.dataset import COFW, WFLW, Face300W, AFLW
from torch.utils.data import DataLoader
import lib.models

def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Train Face Alignment Model'
    )
    parser.add_argument(
        '--cfg',
        required=True, 
        help='experiment configuration filename',
        type=str
    )
    parser.add_argument(
        '--model_state_file',
        required=True,
        help='path to model checkpoint',
        type=str
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    update_config(cfg, args)

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, 'valid')

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    # cudnn related setting
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    model = eval('lib.models.'+cfg.MODEL.NAME+'.get_face_alignment_net')(
        cfg, is_train=False
    )
    logger.info(f'=> loading model from {model_state_file}')

    # Load the checkpoint onto the CPU first
    state = torch.load(model_state_file, map_location='cpu')

    # Safely access epoch and best_nme with default values
    last_epoch = state.get('epoch', -1)
    best_nme = state.get('best_nme', 'N/A')
    logger.info(f"=> Loaded checkpoint from epoch {last_epoch} with best NME: {best_nme}")

    # Correctly identify the weights dictionary
    if 'best_state_dict' in state:
        weights = state['best_state_dict']['state_dict']
    elif 'state_dict' in state:
        weights = state['state_dict']
    else:
        weights = state

    # Create a new state dict and remove the "module." prefix if it exists
    new_state_dict = OrderedDict()
    for k, v in weights.items():
        name = k[7:] if k.startswith('module.') else k  # remove `module.`
        new_state_dict[name] = v

    # Load the cleaned weights into the base model
    model.load_state_dict(new_state_dict)
    # define loss function (criterion) and optimizer
    matcher = build_matcher(cfg.MODEL.LANDMARKS)
    weight_dict = {
        'loss_ce': config.LOSS.CE_LOSS_COEF,
        'loss_kpts': config.LOSS.KPT_LOSS_COEF,
        "loss_moce_load": config.LOSS.ANNEAL_CONFIG.MOCE_LOAD_LOSS.START_WEIGHT,
        "loss_moce_stab": config.LOSS.MOCE_STAB_LOSS_COEF,
    }
    criterion = SetCriterion(cfg.MODEL.LANDMARKS, matcher, weight_dict, cfg.MODEL.EXTRA.EOS_COEF, ['labels', 'kpts','moce'],cfg).cuda()

    model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()

    gpus = list(cfg.GPUS)

    val_loader_300w = DataLoader(
        dataset=Face300W(cfg, is_train=False),
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU * len(gpus), shuffle=False, num_workers=cfg.WORKERS, pin_memory=cfg.PIN_MEMORY
    )
    val_loader_aflw = DataLoader(
        dataset=AFLW(cfg, is_train=False),
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU * len(gpus), shuffle=False, num_workers=cfg.WORKERS, pin_memory=cfg.PIN_MEMORY
    )
    val_loader_cofw = DataLoader(
        dataset=COFW(cfg, is_train=False),
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU * len(gpus), shuffle=False, num_workers=cfg.WORKERS, pin_memory=cfg.PIN_MEMORY
    )

    all_val_loaders = {
        '300W': val_loader_300w,
        'AFLW': val_loader_aflw,
        'COFW': val_loader_cofw,
    }

    start_time = time.time()

    logger.info("\n========== Starting Evaluation ==========")
    for dataset_name, val_loader in all_val_loaders.items():
        logger.info(f"--- Evaluating on {dataset_name} ---")
        validate(cfg, dataset_name, val_loader, model, criterion, epoch=last_epoch,
                 start_time=start_time, writer_dict=None)

    logger.info("========== Evaluation Finished ==========")

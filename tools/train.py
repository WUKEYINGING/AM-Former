# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Created by Han Liming
# ------------------------------------------------------------------------------
"""Training script for face alignment model."""

import os
import sys
import json
import time
import argparse
from typing import Dict, Any, Tuple

import pprint
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import lib
from lib.config import config, update_config
from lib.dataset import COFW, WFLW, Face300W, AFLW
from lib.core import function
from lib.utils import utils
from lib.models.matcher import build_matcher
from lib.core.loss import SetCriterion
from lib.utils.metrics import TrainingMetrics


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
    return parser.parse_args()


def setup_dataloaders(config: Any, gpus: list) -> Tuple[Dict, Dict]:
    """Setup train and validation dataloaders.

    Args:
        config: Configuration object
        gpus: List of GPU indices

    Returns:
        Tuple of (train_loaders, val_loaders)
    """
    batch_size = config.TRAIN.BATCH_SIZE_PER_GPU * len(gpus)
    test_batch_size = config.TEST.BATCH_SIZE_PER_GPU * len(gpus)

    # Train dataloaders
    train_datasets = {
        'COFW': COFW(config, is_train=True),
        'WFLW': WFLW(config, is_train=True),
        '300W': Face300W(config, is_train=True),
        'AFLW': AFLW(config, is_train=True)
    }

    train_loaders = {
        name: DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=config.TRAIN.SHUFFLE,
            num_workers=config.WORKERS,
            pin_memory=config.PIN_MEMORY
        )
        for name, dataset in train_datasets.items()
    }

    # Validation dataloaders
    val_datasets = {
        '300W': Face300W(config, is_train=False),
        'WFLW': WFLW(config, is_train=False),
        'AFLW': AFLW(config, is_train=False),
        'COFW': COFW(config, is_train=False)
    }

    val_loaders = {
        name: DataLoader(
            dataset=dataset,
            batch_size=test_batch_size,
            shuffle=False,
            num_workers=config.WORKERS,
            pin_memory=config.PIN_MEMORY
        )
        for name, dataset in val_datasets.items()
    }

    return train_loaders, val_loaders


def save_checkpoint(model: nn.Module, optimizer: optim.Optimizer,
                    epoch: int, best_nmes: Dict, final_output_dir: str,
                    dataset_name: str, nme: float, predictions: torch.Tensor,
                    logger: Any) -> None:
    """Save model checkpoint.

    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        best_nmes: Best NME scores
        final_output_dir: Output directory
        dataset_name: Name of dataset
        nme: Current NME score
        predictions: Model predictions
        logger: Logger object
    """
    base_name = f'{dataset_name}_checkpoint'
    final_filename = f'{base_name}.pth'

    logger.info(f'==> Saving checkpoint: {final_filename}')

    utils.save_checkpoint(
        states={
            "state_dict": model.module.state_dict(),
            "epoch": epoch + 1,
            "best_nmes": best_nmes,
            "optimizer": optimizer.state_dict(),
        },
        predictions=predictions,
        output_dir=final_output_dir,
        filename=final_filename
    )


def main() -> None:
    """Main training function."""
    # Parse arguments
    args = parse_args()
    update_config(config, args)

    # Setup logger
    logger, final_output_dir, tb_log_dir = utils.create_logger(
        config, args.cfg, 'train'
    )

    logger.info("Arguments:\n%s", pprint.pformat(args))
    logger.info("Configuration:\n%s", pprint.pformat(config))

    # CUDA settings
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    # Build model
    model = eval(f'lib.models.{config.MODEL.NAME}.get_face_alignment_net')(
        config, is_train=True
    )

    # Setup TensorBoard writer
    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    # Setup loss weights
    weight_dict = {
        'loss_ce': config.LOSS.CE_LOSS_COEF,
        'loss_kpts': config.LOSS.KPT_LOSS_COEF,
        "loss_moce_load": config.LOSS.ANNEAL_CONFIG.MOCE_LOAD_LOSS.START_WEIGHT,
        "loss_moce_stab": config.LOSS.MOCE_STAB_LOSS_COEF,
    }

    if config.MODEL.EXTRA.AUX_LOSS:
        for i in range(config.MODEL.EXTRA.DEC_LAYERS - 1):
            weight_dict.update(
                {f'{k}_{i}': v for k, v in weight_dict.items()}
            )

    # Setup criterion
    matcher = build_matcher(config.MODEL.LANDMARKS)
    criterion = SetCriterion(
        config.MODEL.LANDMARKS,
        matcher,
        weight_dict,
        config.MODEL.EXTRA.EOS_COEF,
        ['labels', 'kpts', 'moce'],
        config
    ).cuda()

    # DataParallel
    gpus = list(config.GPUS)
    model = nn.DataParallel(model, device_ids=gpus).cuda()

    # Optimizer and scheduler
    optimizer = utils.get_optimizer(config, model)
    last_epoch = config.TRAIN.BEGIN_EPOCH

    if isinstance(config.TRAIN.LR_STEP, list):
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, config.TRAIN.LR_STEP, config.TRAIN.LR_FACTOR, last_epoch - 1
        )
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            config.TRAIN.LR_STEP,
            config.TRAIN.LR_FACTOR,
            last_epoch - 1
        )

    # Setup dataloaders
    train_loaders, val_loaders = setup_dataloaders(config, gpus)

    # Calculate batches per epoch
    batches_per_epoch = sum(len(loader) for loader in train_loaders.values())
    total_training_steps = config.TRAIN.END_EPOCH * batches_per_epoch

    logger.info(f"Total batches per epoch: {batches_per_epoch}")
    logger.info(f"Estimated total training steps: {total_training_steps}")

    # Initialize best NME scores
    best_nmes = {dataset: 100.0 for dataset in train_loaders.keys()}

    # Training loop
    start_time = time.time()
    metrics_collector = TrainingMetrics()

    for epoch in range(last_epoch, config.TRAIN.END_EPOCH):
        current_lr = lr_scheduler.get_last_lr()[0]
        logger.info(f'Epoch [{epoch}] Learning rate: {current_lr}')

        # Train
        train_loss = function.train(
            config, train_loaders, model, criterion,
            optimizer, epoch, start_time, writer_dict, metrics_collector
        )
        lr_scheduler.step()

        # Validate
        if epoch % 1 == 0:  # Consider making this configurable
            for dataset_name, val_loader in val_loaders.items():
                nme, predictions = function.validate(
                    config, dataset_name, val_loader, model,
                    criterion, epoch, start_time, writer_dict
                )

                # Update best NME
                if nme < best_nmes[dataset_name]:
                    best_nmes[dataset_name] = nme
                    logger.info(f'!!! New best NME for {dataset_name}: {nme:.4f} !!!')

                    save_checkpoint(
                        model, optimizer, epoch, best_nmes,
                        final_output_dir, dataset_name, nme,
                        predictions, logger
                    )
                else:
                    logger.info(f'--- NME on {dataset_name}: {nme:.4f}, Best: {best_nmes[dataset_name]:.4f} ---')

            logger.info(f"===== Epoch {epoch} validation finished. Best NMEs: {best_nmes} =====")

        # Update metrics
        current_lr = lr_scheduler.get_last_lr()[0]
        metrics_collector.update_epoch(epoch, train_loss, best_nmes, current_lr)

    # Save final model and report
    final_model_state_file = os.path.join(final_output_dir, 'final_state.pth')
    logger.info(f'Saving final model state to {final_model_state_file}')

    total_training_time = time.time() - start_time
    final_report = metrics_collector.generate_final_report(
        model, total_training_time
    )

    report_path = os.path.join(final_output_dir, 'training_report.json')
    with open(report_path, 'w') as f:
        json.dump(final_report, f, indent=2)

    logger.info('=' * 50)
    logger.info(f'Final Best NMEs: {best_nmes}')
    logger.info('=' * 50)

    torch.save(model.module.state_dict(), final_model_state_file)
    writer_dict['writer'].close()


if __name__ == '__main__':
    main()
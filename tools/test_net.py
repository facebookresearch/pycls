#!/usr/bin/env python3

"""Test a trained classification model."""

import argparse
import numpy as np
import os
import sys
import torch

from pycls.config import assert_cfg
from pycls.config import cfg
from pycls.datasets import loader
from pycls.models import model_builder
from pycls.utils.meters import TestMeter

import pycls.models.losses as losses
import pycls.utils.checkpoint as cu
import pycls.utils.distributed as du
import pycls.utils.logging as lu
import pycls.utils.metrics as mu
import pycls.utils.multiprocessing as mpu

logger = lu.get_logger(__name__)


def parse_args():
    """Parses the arguments."""
    parser = argparse.ArgumentParser(
        description='Test a trained classification model'
    )
    parser.add_argument(
        '--cfg',
        dest='cfg_file',
        help='Config file',
        required=True,
        type=str
    )
    parser.add_argument(
        'opts',
        help='See pycls/core/config.py for all options',
        default=None,
        nargs=argparse.REMAINDER
    )
    if len(sys.argv) == 1:
        parser.logger.info_help()
        sys.exit(1)
    return parser.parse_args()


def log_model_info(model):
    """Logs model info"""
    logger.info('Model:\n{}'.format(model))
    logger.info('Params: {:,}'.format(mu.params_count(model)))
    logger.info('Flops: {:,}'.format(mu.flops_count(model)))


@torch.no_grad()
def eval_epoch(test_loader, model, test_meter, cur_epoch):
    """Evaluates the model on the test set."""

    # Enable eval mode
    model.eval()
    test_meter.iter_tic()

    for cur_iter, (inputs, labels) in enumerate(test_loader):
        # Transfer the data to the current GPU device
        inputs, labels = inputs.cuda(), labels.cuda(non_blocking=True)
        # Compute the predictions
        preds = model(inputs)
        # Compute the errors
        top1_err, top5_err = mu.topk_errors(preds, labels, [1, 5])
        # Combine the errors across the GPUs
        if cfg.NUM_GPUS > 1:
            top1_err, top5_err = du.scaled_all_reduce([top1_err, top5_err])
        # Copy the errors from GPU to CPU (sync point)
        top1_err, top5_err = top1_err.item(), top5_err.item()
        test_meter.iter_toc()
        # Update and log stats
        test_meter.update_stats(
            top1_err, top5_err, inputs.size(0) * cfg.NUM_GPUS
        )
        test_meter.log_iter_stats(cur_epoch, cur_iter)
        test_meter.iter_tic()

    # Log epoch stats
    test_meter.log_epoch_stats(cur_epoch)
    test_meter.reset()


def eval_model():
    """Evaluates the model."""

    # Build the model (before the loaders to ease debugging)
    model = model_builder.build_model()
    log_model_info(model)

    # Load a checkpoint if applicable
    if cfg.TRAIN.AUTO_RESUME and cu.has_checkpoint():
        last_checkpoint = cu.get_checkpoint_last()
        checkpoint_epoch = cu.load_checkpoint(last_checkpoint, model)
        logger.info('Loaded checkpoint from: {}'.format(last_checkpoint))
    elif cfg.TRAIN.START_CHECKPOINT:
        cu.load_checkpoint(cfg.TRAIN.START_CHECKPOINT, model)
        logger.info('Loaded checkpoint from: ' + cfg.TRAIN.START_CHECKPOINT)
    else:
        logger.info('No checkpoints are loaded. Evaluating a randomly initialized network.')
        
    # Create data loaders
    test_loader = loader.construct_test_loader()

    # Create meters
    test_meter = TestMeter(len(test_loader))

    # Evaluate the model
    eval_epoch(test_loader, model, test_meter, 0)


def single_proc_eval():
    """Performs single process evaluation."""

    # Setup logging
    lu.setup_logging()
    # Show the config
    logger.info('Config:\n{}'.format(cfg))

    # Fix the RNG seeds (see RNG comment in core/config.py for discussion)
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    # Configure the CUDNN backend
    torch.backends.cudnn.benchmark = cfg.CUDNN.BENCHMARK

    # evaluate the model
    eval_model()


def main():
    # Parse cmd line args
    args = parse_args()

    # Load config options
    cfg.merge_from_file(args.cfg_file)
    cfg.merge_from_list(args.opts)
    assert_cfg()
    cfg.freeze()

    # Perform evaluation
    if cfg.NUM_GPUS > 1:
        mpu.multi_proc_run(num_proc=cfg.NUM_GPUS, fun=single_proc_eval)
    else:
        single_proc_eval()


if __name__ == '__main__':
    main()


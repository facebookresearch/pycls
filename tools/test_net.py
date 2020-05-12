#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Test a trained classification model."""

import numpy as np
import pycls.core.losses as losses
import pycls.core.model_builder as model_builder
import pycls.datasets.loader as loader
import pycls.utils.benchmark as bu
import pycls.utils.checkpoint as cu
import pycls.utils.distributed as du
import pycls.utils.logging as lu
import pycls.utils.metrics as mu
import pycls.utils.multiprocessing as mpu
import pycls.utils.net as nu
import torch
from pycls.core.config import assert_and_infer_cfg, cfg, load_cfg_fom_args
from pycls.utils.meters import TestMeter


logger = lu.get_logger(__name__)


def log_model_info(model):
    """Logs model info"""
    logger.info("Model:\n{}".format(model))
    logger.info("Params: {:,}".format(mu.params_count(model)))
    logger.info("Flops: {:,}".format(mu.flops_count(model)))
    logger.info("Acts: {:,}".format(mu.acts_count(model)))


@torch.no_grad()
def test_epoch(test_loader, model, test_meter, cur_epoch):
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
        test_meter.update_stats(top1_err, top5_err, inputs.size(0) * cfg.NUM_GPUS)
        test_meter.log_iter_stats(cur_epoch, cur_iter)
        test_meter.iter_tic()

    # Log epoch stats
    test_meter.log_epoch_stats(cur_epoch)
    test_meter.reset()


def test_model():
    """Evaluates the model."""

    # Setup logging
    lu.setup_logging()
    # Show the config
    logger.info("Config:\n{}".format(cfg))

    # Fix the RNG seeds (see RNG comment in core/config.py for discussion)
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    # Configure the CUDNN backend
    torch.backends.cudnn.benchmark = cfg.CUDNN.BENCHMARK

    # Build the model (before the loaders to speed up debugging)
    model = model_builder.build_model()
    log_model_info(model)

    # Compute precise time
    if cfg.PREC_TIME.ENABLED:
        logger.info("Computing precise time...")
        loss_fun = losses.get_loss_fun()
        bu.compute_precise_time(model, loss_fun)
        nu.reset_bn_stats(model)

    # Load model weights
    cu.load_checkpoint(cfg.TEST.WEIGHTS, model)
    logger.info("Loaded model weights from: {}".format(cfg.TEST.WEIGHTS))

    # Create data loaders
    test_loader = loader.construct_test_loader()

    # Create meters
    test_meter = TestMeter(len(test_loader))

    # Evaluate the model
    test_epoch(test_loader, model, test_meter, 0)


def main():
    # Load config options
    load_cfg_fom_args("Test a trained classification model.")
    assert_and_infer_cfg()
    cfg.freeze()

    # Perform evaluation
    if cfg.NUM_GPUS > 1:
        mpu.multi_proc_run(num_proc=cfg.NUM_GPUS, fun=test_model)
    else:
        test_model()


if __name__ == "__main__":
    main()

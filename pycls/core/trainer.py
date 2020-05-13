#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Tools for training and testing a model."""

import numpy as np
import pycls.core.builders as builders
import pycls.core.checkpoint as checkpoint
import pycls.core.distributed as dist
import pycls.core.logging as logging
import pycls.core.meters as meters
import pycls.core.net as net
import pycls.core.optimizer as optim
import pycls.datasets.loader as loader
import torch
from pycls.core.config import cfg


logger = logging.get_logger(__name__)


def is_eval_epoch(cur_epoch):
    """Determines if the model should be evaluated at the current epoch."""
    return (cur_epoch + 1) % cfg.TRAIN.EVAL_PERIOD == 0 or (
        cur_epoch + 1
    ) == cfg.OPTIM.MAX_EPOCH


def train_epoch(train_loader, model, loss_fun, optimizer, train_meter, cur_epoch):
    """Performs one epoch of training."""

    # Shuffle the data
    loader.shuffle(train_loader, cur_epoch)
    # Update the learning rate
    lr = optim.get_epoch_lr(cur_epoch)
    optim.set_lr(optimizer, lr)
    # Enable training mode
    model.train()
    train_meter.iter_tic()

    for cur_iter, (inputs, labels) in enumerate(train_loader):
        # Transfer the data to the current GPU device
        inputs, labels = inputs.cuda(), labels.cuda(non_blocking=True)
        # Perform the forward pass
        preds = model(inputs)
        # Compute the loss
        loss = loss_fun(preds, labels)
        # Perform the backward pass
        optimizer.zero_grad()
        loss.backward()
        # Update the parameters
        optimizer.step()
        # Compute the errors
        top1_err, top5_err = meters.topk_errors(preds, labels, [1, 5])
        # Combine the stats across the GPUs
        if cfg.NUM_GPUS > 1:
            loss, top1_err, top5_err = dist.scaled_all_reduce(
                [loss, top1_err, top5_err]
            )
        # Copy the stats from GPU to CPU (sync point)
        loss, top1_err, top5_err = loss.item(), top1_err.item(), top5_err.item()
        train_meter.iter_toc()
        # Update and log stats
        train_meter.update_stats(
            top1_err, top5_err, loss, lr, inputs.size(0) * cfg.NUM_GPUS
        )
        train_meter.log_iter_stats(cur_epoch, cur_iter)
        train_meter.iter_tic()

    # Log epoch stats
    train_meter.log_epoch_stats(cur_epoch)
    train_meter.reset()


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
        top1_err, top5_err = meters.topk_errors(preds, labels, [1, 5])
        # Combine the errors across the GPUs
        if cfg.NUM_GPUS > 1:
            top1_err, top5_err = dist.scaled_all_reduce([top1_err, top5_err])
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


def train_model():
    """Trains the model."""

    # Setup logging
    logging.setup_logging()
    # Show the config
    logger.info("Config:\n{}".format(cfg))

    # Fix the RNG seeds (see RNG comment in core/config.py for discussion)
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    # Configure the CUDNN backend
    torch.backends.cudnn.benchmark = cfg.CUDNN.BENCHMARK

    # Build the model (before the loaders to speed up debugging)
    model = builders.build_model()
    logger.info("Model:\n{}".format(model))
    logger.info(logging.dump_json_stats(net.complexity(model)))

    # Define the loss function
    loss_fun = builders.build_loss_fun()
    # Construct the optimizer
    optimizer = optim.construct_optimizer(model)

    # Load checkpoint or initial weights
    start_epoch = 0
    if cfg.TRAIN.AUTO_RESUME and checkpoint.has_checkpoint():
        last_checkpoint = checkpoint.get_last_checkpoint()
        checkpoint_epoch = checkpoint.load_checkpoint(last_checkpoint, model, optimizer)
        logger.info("Loaded checkpoint from: {}".format(last_checkpoint))
        start_epoch = checkpoint_epoch + 1
    elif cfg.TRAIN.WEIGHTS:
        checkpoint.load_checkpoint(cfg.TRAIN.WEIGHTS, model)
        logger.info("Loaded initial weights from: {}".format(cfg.TRAIN.WEIGHTS))

    # Compute precise time
    if start_epoch == 0 and cfg.PREC_TIME.ENABLED:
        logger.info("Computing precise time...")
        prec_time = net.compute_precise_time(model, loss_fun)
        logger.info(logging.dump_json_stats(prec_time))
        net.reset_bn_stats(model)

    # Create data loaders
    train_loader = loader.construct_train_loader()
    test_loader = loader.construct_test_loader()

    # Create meters
    train_meter = meters.TrainMeter(len(train_loader))
    test_meter = meters.TestMeter(len(test_loader))

    # Perform the training loop
    logger.info("Start epoch: {}".format(start_epoch + 1))

    for cur_epoch in range(start_epoch, cfg.OPTIM.MAX_EPOCH):
        # Train for one epoch
        train_epoch(train_loader, model, loss_fun, optimizer, train_meter, cur_epoch)
        # Compute precise BN stats
        if cfg.BN.USE_PRECISE_STATS:
            net.compute_precise_bn_stats(model, train_loader)
        # Save a checkpoint
        if checkpoint.is_checkpoint_epoch(cur_epoch):
            checkpoint_file = checkpoint.save_checkpoint(model, optimizer, cur_epoch)
            logger.info("Wrote checkpoint to: {}".format(checkpoint_file))
        # Evaluate the model
        if is_eval_epoch(cur_epoch):
            test_epoch(test_loader, model, test_meter, cur_epoch)


def test_model():
    """Evaluates the model."""

    # Setup logging
    logging.setup_logging()
    # Show the config
    logger.info("Config:\n{}".format(cfg))

    # Fix the RNG seeds (see RNG comment in core/config.py for discussion)
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    # Configure the CUDNN backend
    torch.backends.cudnn.benchmark = cfg.CUDNN.BENCHMARK

    # Build the model (before the loaders to speed up debugging)
    model = builders.build_model()
    logger.info("Model:\n{}".format(model))
    logger.info(logging.dump_json_stats(net.complexity(model)))

    # Compute precise time
    if cfg.PREC_TIME.ENABLED:
        logger.info("Computing precise time...")
        loss_fun = builders.build_loss_fun()
        prec_time = net.compute_precise_time(model, loss_fun)
        logger.info(logging.dump_json_stats(prec_time))
        net.reset_bn_stats(model)

    # Load model weights
    checkpoint.load_checkpoint(cfg.TEST.WEIGHTS, model)
    logger.info("Loaded model weights from: {}".format(cfg.TEST.WEIGHTS))

    # Create data loaders
    test_loader = loader.construct_test_loader()

    # Create meters
    test_meter = meters.TestMeter(len(test_loader))

    # Evaluate the model
    test_epoch(test_loader, model, test_meter, 0)

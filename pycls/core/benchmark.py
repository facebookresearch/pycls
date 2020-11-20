#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Benchmarking functions."""

import pycls.core.logging as logging
import pycls.core.net as net
import pycls.datasets.loader as loader
import torch
import torch.cuda.amp as amp
from pycls.core.config import cfg
from pycls.core.timer import Timer


logger = logging.get_logger(__name__)


@torch.no_grad()
def compute_time_eval(model):
    """Computes precise model forward test time using dummy data."""
    # Use eval mode
    model.eval()
    # Generate a dummy mini-batch and copy data to GPU
    im_size, batch_size = cfg.TRAIN.IM_SIZE, int(cfg.TEST.BATCH_SIZE / cfg.NUM_GPUS)
    inputs = torch.zeros(batch_size, 3, im_size, im_size).cuda(non_blocking=False)
    # Compute precise forward pass time
    timer = Timer()
    total_iter = cfg.PREC_TIME.NUM_ITER + cfg.PREC_TIME.WARMUP_ITER
    for cur_iter in range(total_iter):
        # Reset the timers after the warmup phase
        if cur_iter == cfg.PREC_TIME.WARMUP_ITER:
            timer.reset()
        # Forward
        timer.tic()
        model(inputs)
        torch.cuda.synchronize()
        timer.toc()
    return timer.average_time


def compute_time_train(model, loss_fun):
    """Computes precise model forward + backward time using dummy data."""
    # Use train mode
    model.train()
    # Generate a dummy mini-batch and copy data to GPU
    im_size, batch_size = cfg.TRAIN.IM_SIZE, int(cfg.TRAIN.BATCH_SIZE / cfg.NUM_GPUS)
    inputs = torch.rand(batch_size, 3, im_size, im_size).cuda(non_blocking=False)
    labels = torch.zeros(batch_size, dtype=torch.int64).cuda(non_blocking=False)
    labels_one_hot = net.smooth_one_hot_labels(labels)
    # Cache BatchNorm2D running stats
    bns = [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    bn_stats = [[bn.running_mean.clone(), bn.running_var.clone()] for bn in bns]
    # Create a GradScaler for mixed precision training
    scaler = amp.GradScaler(enabled=cfg.TRAIN.MIXED_PRECISION)
    # Compute precise forward backward pass time
    fw_timer, bw_timer = Timer(), Timer()
    total_iter = cfg.PREC_TIME.NUM_ITER + cfg.PREC_TIME.WARMUP_ITER
    for cur_iter in range(total_iter):
        # Reset the timers after the warmup phase
        if cur_iter == cfg.PREC_TIME.WARMUP_ITER:
            fw_timer.reset()
            bw_timer.reset()
        # Forward
        fw_timer.tic()
        with amp.autocast(enabled=cfg.TRAIN.MIXED_PRECISION):
            preds = model(inputs)
            loss = loss_fun(preds, labels_one_hot)
        torch.cuda.synchronize()
        fw_timer.toc()
        # Backward
        bw_timer.tic()
        scaler.scale(loss).backward()
        torch.cuda.synchronize()
        bw_timer.toc()
    # Restore BatchNorm2D running stats
    for bn, (mean, var) in zip(bns, bn_stats):
        bn.running_mean, bn.running_var = mean, var
    return fw_timer.average_time, bw_timer.average_time


def compute_time_loader(data_loader):
    """Computes loader time."""
    timer = Timer()
    loader.shuffle(data_loader, 0)
    data_loader_iterator = iter(data_loader)
    total_iter = cfg.PREC_TIME.NUM_ITER + cfg.PREC_TIME.WARMUP_ITER
    total_iter = min(total_iter, len(data_loader))
    for cur_iter in range(total_iter):
        if cur_iter == cfg.PREC_TIME.WARMUP_ITER:
            timer.reset()
        timer.tic()
        next(data_loader_iterator)
        timer.toc()
    return timer.average_time


def compute_time_model(model, loss_fun):
    """Times model."""
    logger.info("Computing model timings only...")
    # Compute timings
    test_fw_time = compute_time_eval(model)
    train_fw_time, train_bw_time = compute_time_train(model, loss_fun)
    train_fw_bw_time = train_fw_time + train_bw_time
    # Output iter timing
    iter_times = {
        "test_fw_time": test_fw_time,
        "train_fw_time": train_fw_time,
        "train_bw_time": train_bw_time,
        "train_fw_bw_time": train_fw_bw_time,
    }
    logger.info(logging.dump_log_data(iter_times, "iter_times"))


def compute_time_full(model, loss_fun, train_loader, test_loader):
    """Times model and data loader."""
    logger.info("Computing model and loader timings...")
    # Compute timings
    test_fw_time = compute_time_eval(model)
    train_fw_time, train_bw_time = compute_time_train(model, loss_fun)
    train_fw_bw_time = train_fw_time + train_bw_time
    train_loader_time = compute_time_loader(train_loader)
    # Output iter timing
    iter_times = {
        "test_fw_time": test_fw_time,
        "train_fw_time": train_fw_time,
        "train_bw_time": train_bw_time,
        "train_fw_bw_time": train_fw_bw_time,
        "train_loader_time": train_loader_time,
    }
    logger.info(logging.dump_log_data(iter_times, "iter_times"))
    # Output epoch timing
    epoch_times = {
        "test_fw_time": test_fw_time * len(test_loader),
        "train_fw_time": train_fw_time * len(train_loader),
        "train_bw_time": train_bw_time * len(train_loader),
        "train_fw_bw_time": train_fw_bw_time * len(train_loader),
        "train_loader_time": train_loader_time * len(train_loader),
    }
    logger.info(logging.dump_log_data(epoch_times, "epoch_times"))
    # Compute data loader overhead (assuming DATA_LOADER.NUM_WORKERS>1)
    overhead = max(0, train_loader_time - train_fw_bw_time) / train_fw_bw_time
    logger.info("Overhead of data loader is {:.2f}%".format(overhead * 100))

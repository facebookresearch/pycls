#!/usr/bin/env python3

"""Functions for benchmarking networks."""

import torch

from pycls.config import cfg
from pycls.utils.timer import Timer

import pycls.utils.logging as lu


@torch.no_grad()
def compute_fw_test_time(model, inputs):
    """Computes forward test time (no grad, eval mode)."""
    # Use eval mode
    model.eval()
    # Warm up the caches
    for _cur_iter in range(cfg.PREC_TIME.WARMUP_ITER):
        _preds = model(inputs)
    # Make sure warmup kernels completed
    torch.cuda.synchronize()
    # Compute precise forward pass time
    timer = Timer()
    for _cur_iter in range(cfg.PREC_TIME.NUM_ITER):
        timer.tic()
        _preds = model(inputs)
        torch.cuda.synchronize()
        timer.toc()
    # Make sure forward kernels completed
    torch.cuda.synchronize()
    return timer.average_time


def compute_fw_bw_time(model, loss_fun, inputs, labels):
    """Computes forward backward time."""
    # Use train mode
    model.train()
    # Warm up the caches
    for _cur_iter in range(cfg.PREC_TIME.WARMUP_ITER):
        preds = model(inputs)
        loss = loss_fun(preds, labels)
        loss.backward()
    # Make sure warmup kernels completed
    torch.cuda.synchronize()
    # Compute precise forward backward pass time
    fw_timer = Timer()
    bw_timer = Timer()
    for _cur_iter in range(cfg.PREC_TIME.NUM_ITER):
        # Forward
        fw_timer.tic()
        preds = model(inputs)
        loss = loss_fun(preds, labels)
        torch.cuda.synchronize()
        fw_timer.toc()
        # Backward
        bw_timer.tic()
        loss.backward()
        torch.cuda.synchronize()
        bw_timer.toc()
    # Make sure forward backward kernels completed
    torch.cuda.synchronize()
    return fw_timer.average_time, bw_timer.average_time


def compute_precise_time(model, loss_fun):
    """Computes precise time."""
    assert cfg.TRAIN.DATASET in ['cifar10', 'imagenet'], \
        'Precise time for {} is not supported'.format(cfg.TRAIN.DATASET)
    # Generate a dummy mini-batch
    im_size = 32 if cfg.TRAIN.DATASET == 'cifar10' else 224
    inputs = torch.rand(cfg.PREC_TIME.BATCH_SIZE, 3, im_size, im_size)
    labels = torch.zeros(cfg.PREC_TIME.BATCH_SIZE, dtype=torch.int64)
    # Copy the data to the GPU
    inputs = inputs.cuda(non_blocking=False)
    labels = labels.cuda(non_blocking=False)
    # Compute precise time
    fw_test_time = compute_fw_test_time(model, inputs)
    fw_time, bw_time = compute_fw_bw_time(model, loss_fun, inputs, labels)
    # Log precise time
    lu.log_json_stats({
        'prec_test_fw_time': fw_test_time,
        'prec_train_fw_time': fw_time,
        'prec_train_bw_time': bw_time,
        'prec_train_fw_bw_time': fw_time + bw_time
    })

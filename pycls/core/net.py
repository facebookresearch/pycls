#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Functions for manipulating networks."""

import itertools

import numpy as np
import pycls.core.distributed as dist
import torch
from pycls.core.config import cfg


def unwrap_model(model):
    """Remove the DistributedDataParallel wrapper if present."""
    wrapped = isinstance(model, torch.nn.parallel.distributed.DistributedDataParallel)
    return model.module if wrapped else model


@torch.no_grad()
def compute_precise_bn_stats(model, loader):
    """Computes precise BN stats on training data."""
    # Compute the number of minibatches to use
    num_iter = int(cfg.BN.NUM_SAMPLES_PRECISE / loader.batch_size / cfg.NUM_GPUS)
    num_iter = min(num_iter, len(loader))
    # Retrieve the BN layers
    bns = [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    # Initialize BN stats storage for computing mean(mean(batch)) and mean(var(batch))
    running_means = [torch.zeros_like(bn.running_mean) for bn in bns]
    running_vars = [torch.zeros_like(bn.running_var) for bn in bns]
    # Remember momentum values
    momentums = [bn.momentum for bn in bns]
    # Set momentum to 1.0 to compute BN stats that only reflect the current batch
    for bn in bns:
        bn.momentum = 1.0
    # Average the BN stats for each BN layer over the batches
    for inputs, _labels in itertools.islice(loader, num_iter):
        model(inputs.cuda())
        for i, bn in enumerate(bns):
            running_means[i] += bn.running_mean / num_iter
            running_vars[i] += bn.running_var / num_iter
    # Sync BN stats across GPUs (no reduction if 1 GPU used)
    running_means = dist.scaled_all_reduce(running_means)
    running_vars = dist.scaled_all_reduce(running_vars)
    # Set BN stats and restore original momentum values
    for i, bn in enumerate(bns):
        bn.running_mean = running_means[i]
        bn.running_var = running_vars[i]
        bn.momentum = momentums[i]


def complexity(model):
    """Compute model complexity (model can be model instance or model class)."""
    size = cfg.TRAIN.IM_SIZE
    cx = {"h": size, "w": size, "flops": 0, "params": 0, "acts": 0}
    cx = unwrap_model(model).complexity(cx)
    return {"flops": cx["flops"], "params": cx["params"], "acts": cx["acts"]}


def smooth_one_hot_labels(labels):
    """Convert each label to a one-hot vector."""
    n_classes, label_smooth = cfg.MODEL.NUM_CLASSES, cfg.TRAIN.LABEL_SMOOTHING
    err_str = "Invalid input to one_hot_vector()"
    assert labels.ndim == 1 and labels.max() < n_classes, err_str
    shape = (labels.shape[0], n_classes)
    neg_val = label_smooth / n_classes
    pos_val = 1.0 - label_smooth + neg_val
    labels_one_hot = torch.full(shape, neg_val, dtype=torch.float, device=labels.device)
    labels_one_hot.scatter_(1, labels.long().view(-1, 1), pos_val)
    return labels_one_hot


class SoftCrossEntropyLoss(torch.nn.Module):
    """SoftCrossEntropyLoss (useful for label smoothing and mixup).
    Identical to torch.nn.CrossEntropyLoss if used with one-hot labels."""

    def __init__(self):
        super(SoftCrossEntropyLoss, self).__init__()

    def forward(self, x, y):
        loss = -y * torch.nn.functional.log_softmax(x, -1)
        return torch.sum(loss) / x.shape[0]


def mixup(inputs, labels):
    """
    Apply mixup or cutmix to minibatch depending MIXUP_ALPHA and CUTMIX_ALPHA.
    IF MIXUP_ALPHA > 0, applies mixup (https://arxiv.org/abs/1710.09412).
    IF CUTMIX_ALPHA > 0, applies cutmix (https://arxiv.org/abs/1905.04899).
    If both MIXUP_ALPHA > 0 and CUTMIX_ALPHA > 0, 50-50 chance of which is applied.
    """
    assert labels.shape[1] == cfg.MODEL.NUM_CLASSES, "mixup labels must be one-hot"
    mixup_alpha, cutmix_alpha = cfg.TRAIN.MIXUP_ALPHA, cfg.TRAIN.CUTMIX_ALPHA
    mixup_alpha = mixup_alpha if (cutmix_alpha == 0 or np.random.rand() < 0.5) else 0
    if mixup_alpha > 0:
        m = np.random.beta(mixup_alpha, mixup_alpha)
        permutation = torch.randperm(labels.shape[0])
        inputs = m * inputs + (1.0 - m) * inputs[permutation, :]
        labels = m * labels + (1.0 - m) * labels[permutation, :]
    elif cutmix_alpha > 0:
        m = np.random.beta(cutmix_alpha, cutmix_alpha)
        permutation = torch.randperm(labels.shape[0])
        h, w = inputs.shape[2], inputs.shape[3]
        w_b, h_b = np.int(w * np.sqrt(1.0 - m)), np.int(h * np.sqrt(1.0 - m))
        x_c, y_c = np.random.randint(w), np.random.randint(h)
        x_0, y_0 = np.clip(x_c - w_b // 2, 0, w), np.clip(y_c - h_b // 2, 0, h)
        x_1, y_1 = np.clip(x_c + w_b // 2, 0, w), np.clip(y_c + h_b // 2, 0, h)
        m = 1.0 - ((x_1 - x_0) * (y_1 - y_0) / (h * w))
        inputs[:, :, y_0:y_1, x_0:x_1] = inputs[permutation, :, y_0:y_1, x_0:x_1]
        labels = m * labels + (1.0 - m) * labels[permutation, :]
    return inputs, labels, labels.argmax(1)


def update_model_ema(model, model_ema, cur_epoch, cur_iter):
    """Update exponential moving average (ema) of model weights."""
    update_period = cfg.OPTIM.EMA_UPDATE_PERIOD
    if update_period == 0 or cur_iter % update_period != 0:
        return
    # Adjust alpha to be fairly independent of other parameters
    adjust = cfg.TRAIN.BATCH_SIZE / cfg.OPTIM.MAX_EPOCH * update_period
    alpha = min(1.0, cfg.OPTIM.EMA_ALPHA * adjust)
    # During warmup simply copy over weights instead of using ema
    alpha = 1.0 if cur_epoch < cfg.OPTIM.WARMUP_EPOCHS else alpha
    # Take ema of all parameters (not just named parameters)
    params = unwrap_model(model).state_dict()
    for name, param in unwrap_model(model_ema).state_dict().items():
        param.copy_(param * (1.0 - alpha) + params[name] * alpha)

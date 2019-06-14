#!/usr/bin/env python3

"""Learning rate policies."""

import numpy as np

from pycls.core.config import cfg


def lr_func_steps(cur_epoch):
    """
    For cfg.OPTIM.LR_POLICY = 'steps_with_relative_lrs'

    Change the learning rate by scaling the base learning rate by
    the factors specified for each step.

    Example:
        cfg.OPTIM.MAX_EPOCH: 90
        cfg.OPTIM.STEPS:     [0,    60,    80]
        cfg.OPTIM.LR_MULS:   [1, 0.1, 0.01]
        cfg.OPTIM.BASE_LR:   0.02
        for cur_epoch in [0, 59]   use 0.02
                      in [60, 79]  use 0.002
                      in [80, inf] use 0.0002
    """
    ind = [i for i, s in enumerate(cfg.OPTIM.STEPS) if cur_epoch >= s][-1]
    return cfg.OPTIM.LR_MULS[ind] * cfg.OPTIM.BASE_LR


def lr_func_exp(cur_epoch):
    """For cfg.OPTIM.LR_POLICY = 'exp'"""
    return cfg.OPTIM.BASE_LR * cfg.OPTIM.GAMMA ** cur_epoch


def lr_func_cos(cur_epoch):
    """For cfg.OPTIM.LR_POLICY = 'cos'"""
    base_lr, max_epoch = cfg.OPTIM.BASE_LR, cfg.OPTIM.MAX_EPOCH
    return 0.5 * base_lr * (1.0 + np.cos(np.pi * cur_epoch / max_epoch))


def get_lr_func():
    """Retrieves the specified lr policy function"""
    policy = 'lr_func_' + cfg.OPTIM.LR_POLICY
    if policy not in globals():
        raise NotImplementedError(
            'Unknown LR policy: {}'.format(cfg.OPTIM.LR_POLICY)
        )
    return globals()[policy]


def get_epoch_lr(cur_epoch):
    """Retrieves the lr for the given epoch according to the policy."""
    lr = get_lr_func()(cur_epoch)
    if cur_epoch < cfg.OPTIM.WARMUP_EPOCHS:
        # Linear warmup
        alpha = cur_epoch / cfg.OPTIM.WARMUP_EPOCHS
        warmup_factor = cfg.OPTIM.WARMUP_FACTOR * (1.0 - alpha) + alpha
        lr *= warmup_factor
    return lr

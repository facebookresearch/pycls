#!/usr/bin/env python3

"""Learning rate policies."""

import numpy as np

from pycls.core.config import cfg


def get_step_index(cur_epoch):
    """Retrieves the lr step index for the given epoch."""
    steps = cfg.OPTIM.STEPS + [cfg.OPTIM.MAX_EPOCH]
    for ind, step in enumerate(steps):
        if cur_epoch < step:
            break
    return ind - 1


def lr_func_steps_with_lrs(cur_epoch):
    """
    For cfg.OPTIM.LR_POLICY = 'steps_with_lrs'

    Change the learning rate to specified values at specified epochs.

    Example:
        cfg.OPTIM.MAX_EPOCH: 90
        cfg.OPTIM.STEPS:     [0,    60,    80]
        cfg.OPTIM.LRS:       [0.02, 0.002, 0.0002]
        for cur_epoch in [0, 59]   use 0.02
                      in [60, 79]  use 0.002
                      in [80, inf] use 0.0002
    """
    ind = get_step_index(cur_epoch)
    return cfg.OPTIM.LRS[ind]


def lr_func_steps_with_relative_lrs(cur_epoch):
    """
    For cfg.OPTIM.LR_POLICY = 'steps_with_relative_lrs'

    Change the learning rate by scaling the base learning rate by
    the factors specified for each step.

    Example:
        cfg.OPTIM.MAX_EPOCH: 90
        cfg.OPTIM.STEPS:     [0,    60,    80]
        cfg.OPTIM.LRS:       [1, 0.1, 0.01]
        cfg.OPTIM.BASE_LR:   0.02
        for cur_epoch in [0, 59]   use 0.02
                      in [60, 79]  use 0.002
                      in [80, inf] use 0.0002
    """
    ind = get_step_index(cur_epoch)
    return cfg.OPTIM.LRS[ind] * cfg.OPTIM.BASE_LR


def lr_func_steps_with_decay(cur_epoch):
    """
    For cfg.OPTIM.LR_POLICY = 'steps_with_decay'

    Change the learning rate specified epochs based on the formula
    lr = base_lr * gamma ** lr_epoch_count.

    Example:
        cfg.OPTIM.MAX_EPOCH: 90
        cfg.OPTIM.STEPS:     [0,    60,    80]
        cfg.OPTIM.GAMMA:     0.1
        cfg.OPTIM.BASE_LR:   0.02
        for cur_iter in [0, 59]   use 0.02 = 0.02 * 0.1 ** 0
                     in [60, 79]  use 0.002 = 0.02 * 0.1 ** 1
                     in [80, inf] use 0.0002 = 0.02 * 0.1 ** 2
    """
    ind = get_step_index(cur_epoch)
    return cfg.OPTIM.BASE_LR * cfg.OPTIM.GAMMA ** ind


def lr_func_exp(cur_epoch):
    """For cfg.OPTIM.LR_POLICY = 'exp'"""
    return (
        cfg.OPTIM.BASE_LR *
        cfg.OPTIM.GAMMA ** (cur_epoch // cfg.OPTIM.STEP_SIZE)
    )


def lr_func_cos(cur_epoch):
    """For cfg.OPTIM.LR_POLICY = 'cos'"""
    return (
        0.5 * cfg.OPTIM.BASE_LR * (
            1.0 + np.cos(
                np.pi *
                (cur_epoch // cfg.OPTIM.STEP_SIZE * cfg.OPTIM.STEP_SIZE) /
                cfg.OPTIM.MAX_EPOCH
            )
        )
    )


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

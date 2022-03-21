#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Optimizer."""

import matplotlib.pyplot as plt
import numpy as np
import torch
from pycls.core.config import cfg


def construct_optimizer(model):
    """Constructs the optimizer.

    Note that the momentum update in PyTorch differs from the one in Caffe2.
    In particular,

        Caffe2:
            V := mu * V + lr * g
            p := p - V

        PyTorch:
            V := mu * V + g
            p := p - lr * V

    where V is the velocity, mu is the momentum factor, lr is the learning rate,
    g is the gradient and p are the parameters.

    Since V is defined independently of the learning rate in PyTorch,
    when the learning rate is changed there is no need to perform the
    momentum correction by scaling V (unlike in the Caffe2 case).
    """
    # Split parameters into types and get weight decay for each type
    optim, wd, params = cfg.OPTIM, cfg.OPTIM.WEIGHT_DECAY, [[], [], [], []]
    for n, p in model.named_parameters():
        ks = [k for (k, x) in enumerate(["bn", "ln", "bias", ""]) if x in n]
        params[ks[0]].append(p)
    wds = [
        cfg.BN.CUSTOM_WEIGHT_DECAY if cfg.BN.USE_CUSTOM_WEIGHT_DECAY else wd,
        cfg.LN.CUSTOM_WEIGHT_DECAY if cfg.LN.USE_CUSTOM_WEIGHT_DECAY else wd,
        optim.BIAS_CUSTOM_WEIGHT_DECAY if optim.BIAS_USE_CUSTOM_WEIGHT_DECAY else wd,
        wd,
    ]
    param_wds = [{"params": p, "weight_decay": w} for (p, w) in zip(params, wds) if p]
    # Set up optimizer
    if optim.OPTIMIZER == "sgd":
        if cfg.OPTIM.MTA:
            optimizer_fn = torch.optim._multi_tensor.SGD
        else:
            optimizer_fn = torch.optim.SGD
        return optimizer_fn(
            param_wds,
            lr=optim.BASE_LR,
            momentum=optim.MOMENTUM,
            weight_decay=wd,
            dampening=optim.DAMPENING,
            nesterov=optim.NESTEROV,
        )
    elif optim.OPTIMIZER == "adam":
        if cfg.OPTIM.MTA:
            optimizer_fn = torch.optim._multi_tensor.Adam
        else:
            optimizer_fn = torch.optim.Adam
        return optimizer_fn(
            param_wds,
            lr=optim.BASE_LR,
            betas=(optim.BETA1, optim.BETA2),
            weight_decay=wd,
        )
    elif optim.OPTIMIZER == "adamw":
        if cfg.OPTIM.MTA:
            optimizer_fn = torch.optim._multi_tensor.AdamW
        else:
            optimizer_fn = torch.optim.AdamW
        return optimizer_fn(
            param_wds,
            lr=optim.BASE_LR,
            betas=(optim.BETA1, optim.BETA2),
            weight_decay=wd,
        )
    else:
        raise NotImplementedError


def lr_fun_steps(cur_epoch):
    """Steps schedule (cfg.OPTIM.LR_POLICY = 'steps')."""
    ind = [i for i, s in enumerate(cfg.OPTIM.STEPS) if cur_epoch >= s][-1]
    return cfg.OPTIM.LR_MULT ** ind


def lr_fun_exp(cur_epoch):
    """Exponential schedule (cfg.OPTIM.LR_POLICY = 'exp')."""
    return cfg.OPTIM.MIN_LR ** (cur_epoch / cfg.OPTIM.MAX_EPOCH)


def lr_fun_cos(cur_epoch):
    """Cosine schedule (cfg.OPTIM.LR_POLICY = 'cos')."""
    lr = 0.5 * (1.0 + np.cos(np.pi * cur_epoch / cfg.OPTIM.MAX_EPOCH))
    return (1.0 - cfg.OPTIM.MIN_LR) * lr + cfg.OPTIM.MIN_LR


def lr_fun_lin(cur_epoch):
    """Linear schedule (cfg.OPTIM.LR_POLICY = 'lin')."""
    lr = 1.0 - cur_epoch / cfg.OPTIM.MAX_EPOCH
    return (1.0 - cfg.OPTIM.MIN_LR) * lr + cfg.OPTIM.MIN_LR


def get_lr_fun():
    """Retrieves the specified lr policy function"""
    lr_fun = "lr_fun_" + cfg.OPTIM.LR_POLICY
    assert lr_fun in globals(), "Unknown LR policy: " + cfg.OPTIM.LR_POLICY
    err_str = "exp lr policy requires OPTIM.MIN_LR to be greater than 0."
    assert cfg.OPTIM.LR_POLICY != "exp" or cfg.OPTIM.MIN_LR > 0, err_str
    return globals()[lr_fun]


def get_epoch_lr(cur_epoch):
    """Retrieves the lr for the given epoch according to the policy."""
    # Get lr and scale by by BASE_LR
    lr = get_lr_fun()(cur_epoch) * cfg.OPTIM.BASE_LR
    # Linear warmup
    if cur_epoch < cfg.OPTIM.WARMUP_EPOCHS:
        alpha = cur_epoch / cfg.OPTIM.WARMUP_EPOCHS
        warmup_factor = cfg.OPTIM.WARMUP_FACTOR * (1.0 - alpha) + alpha
        lr *= warmup_factor
    return lr


def set_lr(optimizer, new_lr):
    """Sets the optimizer lr to the specified value."""
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr


def plot_lr_fun():
    """Visualizes lr function."""
    epochs = list(range(cfg.OPTIM.MAX_EPOCH))
    lrs = [get_epoch_lr(epoch) for epoch in epochs]
    plt.plot(epochs, lrs, ".-")
    plt.title("lr_policy: {}".format(cfg.OPTIM.LR_POLICY))
    plt.xlabel("epochs")
    plt.ylabel("learning rate")
    plt.ylim(bottom=0)
    plt.show()

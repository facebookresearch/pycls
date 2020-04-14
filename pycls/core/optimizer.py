#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Optimizer."""

import pycls.utils.lr_policy as lr_policy
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
    # Batchnorm parameters.
    bn_params = []
    # Non-batchnorm parameters.
    non_bn_parameters = []
    for name, p in model.named_parameters():
        if "bn" in name:
            bn_params.append(p)
        else:
            non_bn_parameters.append(p)
    # Apply different weight decay to Batchnorm and non-batchnorm parameters.
    bn_weight_decay = (
        cfg.BN.CUSTOM_WEIGHT_DECAY
        if cfg.BN.USE_CUSTOM_WEIGHT_DECAY
        else cfg.OPTIM.WEIGHT_DECAY
    )
    optim_params = [
        {"params": bn_params, "weight_decay": bn_weight_decay},
        {"params": non_bn_parameters, "weight_decay": cfg.OPTIM.WEIGHT_DECAY},
    ]
    # Check all parameters will be passed into optimizer.
    assert len(list(model.parameters())) == len(non_bn_parameters) + len(
        bn_params
    ), "parameter size does not match: {} + {} != {}".format(
        len(non_bn_parameters), len(bn_params), len(list(model.parameters()))
    )
    return torch.optim.SGD(
        optim_params,
        lr=cfg.OPTIM.BASE_LR,
        momentum=cfg.OPTIM.MOMENTUM,
        weight_decay=cfg.OPTIM.WEIGHT_DECAY,
        dampening=cfg.OPTIM.DAMPENING,
        nesterov=cfg.OPTIM.NESTEROV,
    )


def get_epoch_lr(cur_epoch):
    """Retrieves the lr for the given epoch (as specified by the lr policy)."""
    return lr_policy.get_epoch_lr(cur_epoch)


def set_lr(optimizer, new_lr):
    """Sets the optimizer lr to the specified value."""
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr

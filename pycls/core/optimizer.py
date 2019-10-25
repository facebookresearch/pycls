#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Optimizer."""

import torch

from pycls.core.config import cfg

import pycls.utils.lr_policy as lr_policy


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
    return torch.optim.SGD(
        model.parameters(),
        lr=cfg.OPTIM.BASE_LR,
        momentum=cfg.OPTIM.MOMENTUM,
        weight_decay=cfg.OPTIM.WEIGHT_DECAY,
        dampening=cfg.OPTIM.DAMPENING,
        nesterov=cfg.OPTIM.NESTEROV
    )


def get_epoch_lr(cur_epoch):
    """Retrieves the lr for the given epoch (as specified by the lr policy)."""
    return lr_policy.get_epoch_lr(cur_epoch)


def set_lr(optimizer, new_lr):
    """Sets the optimizer lr to the specified value."""
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

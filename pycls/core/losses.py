#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Loss functions."""

import torch.nn as nn
from pycls.core.config import cfg


# Supported loss functions
_loss_funs = {"cross_entropy": nn.CrossEntropyLoss}


def get_loss_fun():
    """Retrieves the loss function."""
    assert (
        cfg.MODEL.LOSS_FUN in _loss_funs.keys()
    ), "Loss function '{}' not supported".format(cfg.TRAIN.LOSS)
    return _loss_funs[cfg.MODEL.LOSS_FUN]().cuda()


def register_loss_fun(name, ctor):
    """Registers a loss function dynamically."""
    _loss_funs[name] = ctor

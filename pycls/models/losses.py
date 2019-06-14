#!/usr/bin/env python3

"""Loss functions."""

import torch.nn as nn

from pycls.core.config import cfg

# Supported losses
_LOSS_FUNS = {
    'cross_entropy': nn.CrossEntropyLoss,
}


def get_loss_fun():
    """Retrieves the loss function."""
    assert cfg.MODEL.LOSS_FUN in _LOSS_FUNS.keys(), \
        'Loss function \'{}\' not supported'.format(cfg.TRAIN.LOSS)
    return _LOSS_FUNS[cfg.MODEL.LOSS_FUN]().cuda()

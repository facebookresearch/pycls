#!/usr/bin/env python3

"""Functions for manipulating networks."""

# TODO(ilijar): Test precise bn computation for NUM_GPUS > 1

import itertools
import math
import torch
import torch.nn as nn

from pycls.config import cfg


def init_weights(model):
    """Performs ResNet style weight initialization."""
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            # Note that there is no bias due to BN
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(mean=0.0, std=math.sqrt(2.0 / fan_out))
        elif isinstance(m, nn.BatchNorm2d):
            zero_init_gamma = (
                hasattr(m, 'final_bn') and m.final_bn
                and cfg.BN.ZERO_INIT_FINAL_GAMMA
            )
            m.weight.data.fill_(0.0 if zero_init_gamma else 1.0)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(mean=0.0, std=0.01)
            m.bias.data.zero_()


@torch.no_grad()
def compute_precise_bn_stats(model, loader):
    """Computes precise BN stats on training data."""
    # Compute the number of minibatches to use
    num_iter = min(cfg.BN.NUM_SAMPLES_PRECISE // loader.batch_size, len(loader))
    # Retrieve the BN layers
    bns = [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    # Initialize stats storage
    mus = [torch.zeros_like(bn.running_mean) for bn in bns]
    sqs = [torch.zeros_like(bn.running_var) for bn in bns]
    # Remember momentum values
    moms = [bn.momentum for bn in bns]
    # Disable momentum
    for bn in bns:
        bn.momentum = 1.0
    # Accumulate the stats across the data samples
    for inputs, _labels in itertools.islice(loader, num_iter):
        model(inputs.cuda())
        # Accumulate the stats for each BN layer
        for i, bn in enumerate(bns):
            m, v = bn.running_mean, bn.running_var
            sqs[i] += (v + m * m) / num_iter
            mus[i] += m / num_iter
    # Set the stats and restore momentum values
    for i, bn in enumerate(bns):
        bn.running_var = sqs[i] - mus[i] * mus[i]
        bn.running_mean = mus[i]
        bn.momentum = moms[i]


def get_flat_weights(model):
    """Gets all model weights as a single flat vector."""
    return torch.cat([p.data.view(-1, 1) for p in model.parameters()], 0)


def set_flat_weights(model, flat_weights):
    """Sets all model weights from a single flat vector."""
    k = 0
    for p in model.parameters():
        n = p.data.numel()
        p.data.copy_(flat_weights[k:(k + n)].view_as(p.data))
        k += n
    assert k == flat_weights.numel()

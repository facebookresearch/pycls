#!/usr/bin/env python3

"""Functions for manipulating networks."""

import itertools

import torch

from pycls.core.config import cfg


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
        p.data.copy_(flat_weights[k:k+n].view_as(p.data))
        k += n
    assert k == flat_weights.numel()

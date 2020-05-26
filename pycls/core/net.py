#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Functions for manipulating networks."""

import itertools
import math

import torch
import torch.nn as nn
from pycls.core.config import cfg


def init_weights(m):
    """Performs ResNet-style weight initialization."""
    if isinstance(m, nn.Conv2d):
        # Note that there is no bias due to BN
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(mean=0.0, std=math.sqrt(2.0 / fan_out))
    elif isinstance(m, nn.BatchNorm2d):
        zero_init_gamma = cfg.BN.ZERO_INIT_FINAL_GAMMA
        zero_init_gamma = hasattr(m, "final_bn") and m.final_bn and zero_init_gamma
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


def reset_bn_stats(model):
    """Resets running BN stats."""
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.reset_running_stats()


def complexity_conv2d(cx, w_in, w_out, k, stride, padding, groups=1, bias=False):
    """Accumulates complexity of Conv2D into cx = (h, w, flops, params, acts)."""
    h, w, flops, params, acts = cx["h"], cx["w"], cx["flops"], cx["params"], cx["acts"]
    h = (h + 2 * padding - k) // stride + 1
    w = (w + 2 * padding - k) // stride + 1
    flops += k * k * w_in * w_out * h * w // groups
    params += k * k * w_in * w_out // groups
    flops += w_out if bias else 0
    params += w_out if bias else 0
    acts += w_out * h * w
    return {"h": h, "w": w, "flops": flops, "params": params, "acts": acts}


def complexity_batchnorm2d(cx, w_in):
    """Accumulates complexity of BatchNorm2D into cx = (h, w, flops, params, acts)."""
    h, w, flops, params, acts = cx["h"], cx["w"], cx["flops"], cx["params"], cx["acts"]
    params += 2 * w_in
    return {"h": h, "w": w, "flops": flops, "params": params, "acts": acts}


def complexity_maxpool2d(cx, k, stride, padding):
    """Accumulates complexity of MaxPool2d into cx = (h, w, flops, params, acts)."""
    h, w, flops, params, acts = cx["h"], cx["w"], cx["flops"], cx["params"], cx["acts"]
    h = (h + 2 * padding - k) // stride + 1
    w = (w + 2 * padding - k) // stride + 1
    return {"h": h, "w": w, "flops": flops, "params": params, "acts": acts}


def complexity(model):
    """Compute model complexity (model can be model instance or model class)."""
    size = cfg.TRAIN.IM_SIZE
    cx = {"h": size, "w": size, "flops": 0, "params": 0, "acts": 0}
    cx = model.complexity(cx)
    return {"flops": cx["flops"], "params": cx["params"], "acts": cx["acts"]}


def drop_connect(x, drop_ratio):
    """Drop connect (adapted from DARTS)."""
    keep_ratio = 1.0 - drop_ratio
    mask = torch.empty([x.shape[0], 1, 1, 1], dtype=x.dtype, device=x.device)
    mask.bernoulli_(keep_ratio)
    x.div_(keep_ratio)
    x.mul_(mask)
    return x


def get_flat_weights(model):
    """Gets all model weights as a single flat vector."""
    return torch.cat([p.data.view(-1, 1) for p in model.parameters()], 0)


def set_flat_weights(model, flat_weights):
    """Sets all model weights from a single flat vector."""
    k = 0
    for p in model.parameters():
        n = p.data.numel()
        p.data.copy_(flat_weights[k : (k + n)].view_as(p.data))
        k += n
    assert k == flat_weights.numel()

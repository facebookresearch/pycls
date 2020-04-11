#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Functions for computing metrics."""

import numpy as np
import torch
import torch.nn as nn
from pycls.core.config import cfg


# Number of bytes in a megabyte
_B_IN_MB = 1024 * 1024


def topks_correct(preds, labels, ks):
    """Computes the number of top-k correct predictions for each k."""
    assert preds.size(0) == labels.size(
        0
    ), "Batch dim of predictions and labels must match"
    # Find the top max_k predictions for each sample
    _top_max_k_vals, top_max_k_inds = torch.topk(
        preds, max(ks), dim=1, largest=True, sorted=True
    )
    # (batch_size, max_k) -> (max_k, batch_size)
    top_max_k_inds = top_max_k_inds.t()
    # (batch_size, ) -> (max_k, batch_size)
    rep_max_k_labels = labels.view(1, -1).expand_as(top_max_k_inds)
    # (i, j) = 1 if top i-th prediction for the j-th sample is correct
    top_max_k_correct = top_max_k_inds.eq(rep_max_k_labels)
    # Compute the number of topk correct predictions for each k
    topks_correct = [top_max_k_correct[:k, :].view(-1).float().sum() for k in ks]
    return topks_correct


def topk_errors(preds, labels, ks):
    """Computes the top-k error for each k."""
    num_topks_correct = topks_correct(preds, labels, ks)
    return [(1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct]


def topk_accuracies(preds, labels, ks):
    """Computes the top-k accuracy for each k."""
    num_topks_correct = topks_correct(preds, labels, ks)
    return [(x / preds.size(0)) * 100.0 for x in num_topks_correct]


def params_count(model):
    """Computes the number of parameters."""
    return np.sum([p.numel() for p in model.parameters()]).item()


def flops_count(model):
    """Computes the number of flops statically."""
    h, w = cfg.TRAIN.IM_SIZE, cfg.TRAIN.IM_SIZE
    count = 0
    for n, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            if "se." in n:
                count += m.in_channels * m.out_channels + m.bias.numel()
                continue
            h_out = (h + 2 * m.padding[0] - m.kernel_size[0]) // m.stride[0] + 1
            w_out = (w + 2 * m.padding[1] - m.kernel_size[1]) // m.stride[1] + 1
            count += np.prod([m.weight.numel(), h_out, w_out])
            if ".proj" not in n:
                h, w = h_out, w_out
        elif isinstance(m, nn.MaxPool2d):
            h = (h + 2 * m.padding - m.kernel_size) // m.stride + 1
            w = (w + 2 * m.padding - m.kernel_size) // m.stride + 1
        elif isinstance(m, nn.Linear):
            count += m.in_features * m.out_features + m.bias.numel()
    return count.item()


def acts_count(model):
    """Computes the number of activations statically."""
    h, w = cfg.TRAIN.IM_SIZE, cfg.TRAIN.IM_SIZE
    count = 0
    for n, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            if "se." in n:
                count += m.out_channels
                continue
            h_out = (h + 2 * m.padding[0] - m.kernel_size[0]) // m.stride[0] + 1
            w_out = (w + 2 * m.padding[1] - m.kernel_size[1]) // m.stride[1] + 1
            count += np.prod([m.out_channels, h_out, w_out])
            if ".proj" not in n:
                h, w = h_out, w_out
        elif isinstance(m, nn.MaxPool2d):
            h = (h + 2 * m.padding - m.kernel_size) // m.stride + 1
            w = (w + 2 * m.padding - m.kernel_size) // m.stride + 1
        elif isinstance(m, nn.Linear):
            count += m.out_features
    return count.item()


def gpu_mem_usage():
    """Computes the GPU memory usage for the current device (MB)."""
    mem_usage_bytes = torch.cuda.max_memory_allocated()
    return mem_usage_bytes / _B_IN_MB

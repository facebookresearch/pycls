#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Distributed helpers."""

import os

import torch
from pycls.core.config import cfg


# Make work w recent PyTorch versions (https://github.com/pytorch/pytorch/issues/37377)
os.environ["MKL_THREADING_LAYER"] = "GNU"


def is_master_proc():
    """Determines if the current process is the master process.

    Master process is responsible for logging, writing and loading checkpoints. In
    the multi GPU setting, we assign the master role to the rank 0 process. When
    training using a single GPU, there is a single process which is considered master.
    """
    return cfg.NUM_GPUS == 1 or torch.distributed.get_rank() == 0


def scaled_all_reduce(tensors):
    """Performs the scaled all_reduce operation on the provided tensors.

    The input tensors are modified in-place. Currently supports only the sum
    reduction operator. The reduced values are scaled by the inverse size of the
    process group (equivalent to cfg.NUM_GPUS).
    """
    # There is no need for reduction in the single-proc case
    if cfg.NUM_GPUS == 1:
        return tensors
    # Queue the reductions
    reductions = []
    for tensor in tensors:
        reduction = torch.distributed.all_reduce(tensor, async_op=True)
        reductions.append(reduction)
    # Wait for reductions to finish
    for reduction in reductions:
        reduction.wait()
    # Scale the results
    for tensor in tensors:
        tensor.mul_(1.0 / cfg.NUM_GPUS)
    return tensors


def init_distributed():
    """Initialize torch.distributed and set the CUDA device.

    Expects environment variables to be set as per
    https://pytorch.org/docs/stable/distributed.html#environment-variable-initialization
    along with the environment variable "LOCAL_RANK" which is used to set the
    CUDA device.
    """
    local_rank = int(os.environ["LOCAL_RANK"])

    # Initialize torch.distributed
    torch.distributed.init_process_group(backend=cfg.DIST_BACKEND)

    # Set the GPU to use
    torch.cuda.set_device(local_rank)

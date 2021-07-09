#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Distributed helpers."""

import os
import random

import submitit
import torch
from pycls.core.config import cfg


# Make work w recent PyTorch versions (https://github.com/pytorch/pytorch/issues/37377)
os.environ["MKL_THREADING_LAYER"] = "GNU"


class SubmititRunner(submitit.helpers.Checkpointable):
    """A callable which is passed to submitit to launch the jobs."""

    def __init__(self, port, fun, cfg_state):
        self.cfg_state = cfg_state
        self.port = port
        self.fun = fun

    def __call__(self):
        job_env = submitit.JobEnvironment()
        os.environ["MASTER_ADDR"] = job_env.hostnames[0]
        os.environ["MASTER_PORT"] = str(self.port)
        os.environ["RANK"] = str(job_env.global_rank)
        os.environ["LOCAL_RANK"] = str(job_env.local_rank)
        os.environ["WORLD_SIZE"] = str(job_env.num_tasks)
        setup_distributed(self.cfg_state)
        self.fun()


def is_master_proc(local=False):
    """
    Determines if the current process is the master process.

    Master process is responsible for logging, writing and loading checkpoints. In
    the multi GPU setting, we assign the master role to the rank 0 process. When
    training using a single GPU, there is a single process which is considered master.

    If local==True, then check if the current process is the master on the current node.
    """
    m = cfg.MAX_GPUS_PER_NODE if local else cfg.NUM_GPUS
    return cfg.NUM_GPUS == 1 or torch.distributed.get_rank() % m == 0


def scaled_all_reduce(tensors):
    """
    Performs the scaled all_reduce operation on the provided tensors.

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


def setup_distributed(cfg_state):
    """
    Initialize torch.distributed and set the CUDA device.

    Expects environment variables to be set as per
    https://pytorch.org/docs/stable/distributed.html#environment-variable-initialization
    along with the environ variable "LOCAL_RANK" which is used to set the CUDA device.

    This is run inside a new process, so the cfg is reset and must be set explicitly.
    """
    cfg.defrost()
    cfg.update(**cfg_state)
    cfg.freeze()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.distributed.init_process_group(backend=cfg.DIST_BACKEND)
    torch.cuda.set_device(local_rank)


def single_proc_run(local_rank, fun, master_port, cfg_state, world_size):
    """Executes fun() on a single GPU in a multi-GPU setup."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(local_rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    setup_distributed(cfg_state)
    fun()


def multi_proc_run(num_proc, fun):
    """Run a single or multi GPU job locally on the current node."""
    launch = cfg.LAUNCH
    if launch.MODE in ["submitit_local", "slurm"]:
        # Launch fun() using submitit either locally or on SLURM
        use_slurm = launch.MODE == "slurm"
        executor = submitit.AutoExecutor if use_slurm else submitit.LocalExecutor
        kwargs = {"slurm_max_num_timeout": launch.MAX_RETRY} if use_slurm else {}
        executor = executor(folder=cfg.OUT_DIR, **kwargs)
        num_gpus_per_node = min(cfg.NUM_GPUS, cfg.MAX_GPUS_PER_NODE)
        executor.update_parameters(
            mem_gb=launch.MEM_PER_GPU * num_gpus_per_node,
            gpus_per_node=num_gpus_per_node,
            tasks_per_node=num_gpus_per_node,
            cpus_per_task=launch.CPUS_PER_GPU,
            nodes=max(1, cfg.NUM_GPUS // cfg.MAX_GPUS_PER_NODE),
            timeout_min=launch.TIME_LIMIT,
            name=launch.NAME,
            slurm_partition=launch.PARTITION,
            slurm_comment=launch.COMMENT,
            slurm_constraint=launch.GPU_TYPE,
            slurm_additional_parameters={"mail-user": launch.EMAIL, "mail-type": "END"},
        )
        master_port = random.randint(cfg.PORT_RANGE[0], cfg.PORT_RANGE[1])
        job = executor.submit(SubmititRunner(master_port, fun, cfg))
        print("Submitted job_id {} with out_dir: {}".format(job.job_id, cfg.OUT_DIR))
        if not use_slurm:
            job.wait()
    elif num_proc > 1:
        master_port = random.randint(cfg.PORT_RANGE[0], cfg.PORT_RANGE[1])
        mp_runner = torch.multiprocessing.start_processes
        args = (fun, master_port, cfg, num_proc)
        # Note: using "fork" below, "spawn" causes time and error regressions. Using
        # spawn changes the default multiprocessing context to spawn, which doesn't
        # interact well with the dataloaders (likely due to the use of OpenCV).
        mp_runner(single_proc_run, args=args, nprocs=num_proc, start_method="fork")
    else:
        fun()

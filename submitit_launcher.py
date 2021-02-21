#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Submitit based launcher for pycls jobs. Can be used to submit multi node jobs
to SLURM, or for local single node multi GPU jobs. For local jobs, the
returned job id is the parent process id.
"""

import argparse
import copy
import itertools
import os
import random
import re
import uuid
from collections.abc import Iterable
from pathlib import Path
from typing import Dict

import submitit

import pycls.core.config as config
import pycls.core.distributed as dist
import pycls.core.trainer as trainer
from pycls.core.config import cfg, merge_from_file


class Trainer:
    def __init__(self, port, is_local=False):
        self.port = port
        self.is_local = is_local

    def init_distributed_env(self):
        job_env = submitit.JobEnvironment()

        os.environ["MASTER_ADDR"] = job_env.hostnames[0]
        os.environ["MASTER_PORT"] = str(self.port)
        os.environ["WORLD_SIZE"] = str(job_env.num_tasks)
        os.environ["RANK"] = str(job_env.global_rank)
        os.environ["LOCAL_RANK"] = str(job_env.local_rank)

    def __call__(self):
        # this is run inside a new process, so the variable cfg is restored,
        # but it will be restored as a separate variable in this script with
        # no connection to cfg in pycls.core.config. so, we update the cfg
        # within the pycls.core.config module with the restored state.
        config.cfg.update(**cfg)

        self.init_distributed_env()

        dist.init_distributed()
        trainer.train_model()


def parse_args():
    parser = argparse.ArgumentParser("Submitit pycls")

    parser.add_argument(
        "--partition", default="", type=str, help="Partition where to submit"
    )
    parser.add_argument(
        "--timeout", default=4200, type=int, help="Duration of the job (mins)"
    )
    parser.add_argument(
        "--max_timeout_retries",
        default=3,
        type=int,
        help="Maximun number of retries after timeouts",
    )
    parser.add_argument(
        "--gpu_type", default="volta", type=str, help="GPU Type to utilize"
    )
    parser.add_argument(
        "--mail",
        default="",
        type=str,
        help="Email this user when the job finishes if specified",
    )
    parser.add_argument(
        "--comment",
        default="",
        type=str,
        help="Comment to pass to scheduler, e.g. priority message",
    )
    parser.add_argument(
        "--cpus_per_gpu", default=10, type=int, help="Number of CPUs per GPU"
    )
    parser.add_argument(
        "--mem_per_gpu", default=60, type=int, help="Memory per GPU (GB)"
    )
    parser.add_argument("--use_local", action="store_true", default=False)
    parser.add_argument("--out_dir", help="Output directory", required=True, type=str)
    parser.add_argument("--cfg", help="Config file location", required=True, type=str)
    # help_s = "See pycls/core/config.py for all options"
    # parser.add_argument("opts", help=help_s, default=None, nargs=argparse.REMAINDER)
    return parser.parse_args()


def setup_config(args):
    merge_from_file(args.cfg)
    cfg.OUT_DIR = args.out_dir
    # cfg.merge_from_list(args.opts)
    config.assert_and_infer_cfg()
    cfg.freeze()


def main():
    args = parse_args()
    setup_config(args)

    if args.use_local:
        executor = submitit.LocalExecutor(folder=args.out_dir)
    else:
        executor = submitit.AutoExecutor(
            folder=args.out_dir, slurm_max_num_timeout=args.max_timeout_retries
        )

    num_gpus_per_node = cfg.NUM_GPUS // cfg.NUM_NODES
    nodes = cfg.NUM_NODES
    partition = args.partition
    timeout_min = args.timeout
    kwargs = {}
    kwargs["constraint"] = args.gpu_type
    if args.comment:
        kwargs["comment"] = args.comment
    if partition is not None:
        kwargs["slurm_partition"] = partition

    executor.update_parameters(
        mem_gb=args.mem_per_gpu * num_gpus_per_node,
        gpus_per_node=num_gpus_per_node,
        tasks_per_node=num_gpus_per_node,
        cpus_per_task=args.cpus_per_gpu,
        nodes=nodes,
        timeout_min=timeout_min,
        **kwargs,
    )

    executor.update_parameters(name="pycls")
    if args.mail:
        executor.update_parameters(
            additional_parameters={"mail-user": args.mail, "mail-type": "END"}
        )

    trainer = Trainer(
        port=random.randint(cfg.PORT_RANGE[0], cfg.PORT_RANGE[1]),
        is_local=args.use_local,
    )

    job = executor.submit(trainer)

    print(f"Submitted job_id {job.job_id} with out_dir: {args.out_dir}")

    if args.use_local:
        job.wait()


if __name__ == "__main__":
    main()

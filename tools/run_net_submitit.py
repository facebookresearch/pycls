#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Submitit based launcher for pycls jobs. Can be used to submit multi node jobs
to SLURM, or for local single node multi GPU jobs. For local jobs, the
returned job id is the parent process id.
"""

import argparse
import os
import random

import pycls.core.config as config
import pycls.core.distributed as dist
import pycls.core.trainer as trainer
import submitit
from pycls.core.config import cfg


def _runner(func):
    dist.init_distributed()
    func()


class Trainer(submitit.helpers.Checkpointable):
    """A callable which is passed to submitit to launch the jobs.

    Since we derive from submitit.helpers.Checkpointable, this job will
    requeued unlimited times if pre-empted, and max_timeout_retries times
    after timing out.
    """

    def __init__(self, run_mode, port):
        self.run_mode = run_mode
        self.port = port

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
        config.cfg.freeze()

        self.init_distributed_env()

        trainer.run(self.run_mode, _runner)


def parse_args():
    parser = argparse.ArgumentParser("Submitit pycls")

    parser.add_argument("--name", type=str, help="Name of the job", required=True)
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
        help="Maximum number of retries after timeouts",
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
    help_s, choices = "Run mode", ["info", "train", "test", "time", "scale"]
    parser.add_argument("--mode", help=help_s, choices=choices, required=True, type=str)
    parser.add_argument("--cfg", help="Config file location", required=True, type=str)
    help_s = "See pycls/core/config.py for all options"
    parser.add_argument("opts", help=help_s, default=None, nargs=argparse.REMAINDER)
    return parser.parse_args()


def setup_config(args):
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    config.assert_and_infer_cfg()
    cfg.freeze()


def main():
    args = parse_args()
    setup_config(args)

    out_dir = cfg.OUT_DIR

    if args.use_local:
        executor = submitit.LocalExecutor(folder=out_dir)
    else:
        executor = submitit.AutoExecutor(
            folder=out_dir, slurm_max_num_timeout=args.max_timeout_retries
        )

    num_gpus_per_node = config.get_num_gpus_per_node()
    nodes = config.get_num_nodes()
    partition = args.partition
    timeout_min = args.timeout
    kwargs = {}
    kwargs["slurm_constraint"] = args.gpu_type
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
        name=args.name,
        **kwargs,
    )

    if args.mail:
        executor.update_parameters(
            additional_parameters={"mail-user": args.mail, "mail-type": "END"}
        )

    trainer = Trainer(args.mode, random.randint(cfg.PORT_RANGE[0], cfg.PORT_RANGE[1]))

    job = executor.submit(trainer)

    print(f"Submitted job_id {job.job_id} with out_dir: {out_dir}")

    if args.use_local:
        job.wait()


if __name__ == "__main__":
    main()

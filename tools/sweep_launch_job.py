#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Launch a job on SLURM managed cluster, called via sweep_launch.py

Submitit based launcher which can be used to submit multi or single node
multi GPU jobs to SLURM. Supposed to be called from sweep_launch.py to
set up the correct environment and code isolation.
"""

import json
import os
import random

import pycls.core.config as config
import pycls.core.distributed as dist
import pycls.core.trainer as trainer
import submitit
from pycls.sweep.config import load_cfg_fom_args, sweep_cfg


class StreamToFile:
    """Stream class which clones an output stream over to a file.

    Used to copy the job logs to the specified paths since SLURM / submitit
    don't give the option to configure the paths directly.
    """

    def __init__(self, stream, dir_name, file_name, mode="a"):
        self.stream = stream
        self.stream_file = open(f"{dir_name}/{file_name}", mode)

    def write(self, message):
        self.stream.write(message)
        self.stream_file.write(message)

    def flush(self):
        self.stream.flush()
        self.stream_file.flush()

    def close(self):
        # we only close the stream file, not the original stream
        self.stream_file.close()


def _runner(func):
    dist.init_distributed()
    func()


class Trainer(submitit.helpers.Checkpointable):
    """A callable which is passed to submitit to launch the jobs.

    Since we derive from submitit.helpers.Checkpointable, this job will
    requeued unlimited times if pre-empted, and max_timeout_retries times
    after timing out.

    We take an array of ports corresponding to each job to ensure there
    are no port conflicts in jobs running on the same nodes.
    """

    def __init__(self, sweep_dir, run_mode, ports):
        self.sweep_dir = sweep_dir
        self.run_mode = run_mode
        self.ports = ports

    def init_distributed_env(self, job_env, port):
        os.environ["MASTER_ADDR"] = job_env.hostnames[0]
        os.environ["MASTER_PORT"] = str(port)
        os.environ["WORLD_SIZE"] = str(job_env.num_tasks)
        os.environ["RANK"] = str(job_env.global_rank)
        os.environ["LOCAL_RANK"] = str(job_env.local_rank)

    def __call__(self, array_id):
        # we import here because this is run in a new process.
        # objects created before this will be pickled and restored.
        # such objects will be disconnected from the modules they
        # are imported from.

        import sys

        import pycls
        import pycls.core.config as config
        from pycls.core.config import cfg

        cfgs_dir = f"{self.sweep_dir}/cfgs/"
        cfg_files = sorted(f for f in os.listdir(cfgs_dir) if f.endswith(".yaml"))
        cfg_file_name = cfg_files[int(array_id)]
        cfg_file = os.path.join(cfgs_dir, cfg_file_name)
        out_sub_dir = cfg_file_name.split(".")[0]
        out_dir = f"{self.sweep_dir}/logs/{out_sub_dir}"
        os.makedirs(out_dir, exist_ok=True)

        job_env = submitit.JobEnvironment()

        # save the stdout and stderr of the rank 0 job to files
        if job_env.global_rank == 0:
            sys.stdout = StreamToFile(sys.stdout, out_dir, "stdout.log")
            sys.stderr = StreamToFile(sys.stderr, out_dir, "stderr.log")

        # Create slurm_file with SLURM info
        slurm_file = os.path.join(out_dir, "SLURM.txt")
        with open(slurm_file, "a") as f:
            f.write("SLURM env variables for the job writing to this directory:\n")
            slurm_info = {
                k: os.environ[k] for k in os.environ if k.startswith("SLURM_")
            }
            f.write(json.dumps(slurm_info, indent=4))

        # print the pycls path to make sure we're running in the copied code base
        print("Path to pycls", pycls.__file__)

        cfg.merge_from_file(cfg_file)
        cfg.OUT_DIR = out_dir
        config.assert_and_infer_cfg()
        cfg.freeze()
        self.init_distributed_env(job_env, self.ports[array_id])
        trainer.run(self.run_mode, _runner)


def main():
    load_cfg_fom_args()
    sweep_cfg.freeze()

    sweep_dir = os.path.abspath(os.path.join(sweep_cfg.ROOT_DIR, sweep_cfg.NAME))
    cfgs_dir = os.path.join(sweep_dir, "cfgs")
    n_cfgs = len([c for c in os.listdir(cfgs_dir) if c.endswith(".yaml")])

    folder = f"{sweep_dir}/logs/slurm/%j"

    executor = submitit.AutoExecutor(
        folder=folder, slurm_max_num_timeout=sweep_cfg.LAUNCH.MAX_RETRY
    )

    num_gpus_per_node = config.get_num_gpus_per_node()
    nodes = config.get_num_nodes()
    partition = sweep_cfg.LAUNCH.PARTITION
    timeout_min = sweep_cfg.LAUNCH.TIME_LIMIT
    kwargs = {}
    kwargs["slurm_constraint"] = sweep_cfg.LAUNCH.GPU_TYPE
    if sweep_cfg.LAUNCH.COMMENT:
        kwargs["comment"] = sweep_cfg.LAUNCH.COMMENT
    if partition is not None:
        kwargs["slurm_partition"] = partition

    executor.update_parameters(
        mem_gb=sweep_cfg.LAUNCH.MEM_PER_GPU * num_gpus_per_node,
        gpus_per_node=num_gpus_per_node,
        tasks_per_node=num_gpus_per_node,
        cpus_per_task=sweep_cfg.LAUNCH.CPUS_PER_GPU,
        nodes=nodes,
        timeout_min=timeout_min,
        name=sweep_cfg.NAME,
        slurm_array_parallelism=sweep_cfg.LAUNCH.PARALLEL_JOBS,
        **kwargs,
    )

    trainer = Trainer(
        sweep_dir=sweep_dir,
        run_mode=sweep_cfg.LAUNCH.MODE,
        ports=[
            random.randint(
                sweep_cfg.SETUP.BASE_CFG.PORT_RANGE[0],
                sweep_cfg.SETUP.BASE_CFG.PORT_RANGE[1],
            )
            for _ in range(n_cfgs)
        ],
    )

    jobs = executor.map_array(trainer, list(range(n_cfgs)))

    print(f"Submitted jobs {[job.job_id for job in jobs]} with sweep_dir: {sweep_dir}")


if __name__ == "__main__":
    main()

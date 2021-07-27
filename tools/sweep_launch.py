#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Launch sweep on a SLURM managed cluster.

Submitit based launcher which can be used to submit multi or single node
multi GPU sweep jobs to SLURM.
"""


import json
import os
import random

import pycls.core.distributed as dist
import pycls.core.trainer as trainer
import submitit
from pycls.sweep.config import load_cfg_fom_args, sweep_cfg


class StreamToFile:
    """
    Stream class which clones an output stream over to a file.

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


class SubmititRunner(submitit.helpers.Checkpointable):
    """
    A callable which is passed to submitit to launch the jobs.

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
        # setup the distributed environment
        job_env = submitit.JobEnvironment()
        os.environ["MASTER_ADDR"] = job_env.hostnames[0]
        os.environ["MASTER_PORT"] = str(self.ports[array_id])
        os.environ["WORLD_SIZE"] = str(job_env.num_tasks)
        os.environ["RANK"] = str(job_env.global_rank)
        os.environ["LOCAL_RANK"] = str(job_env.local_rank)
        # save the stdout and stderr of the rank 0 job to files
        if job_env.global_rank == 0:
            sys.stdout = StreamToFile(sys.stdout, out_dir, "stdout.log")
            sys.stderr = StreamToFile(sys.stderr, out_dir, "stderr.log")
        # Create slurm_file with SLURM info
        slurm_file = os.path.join(out_dir, "SLURM.txt")
        with open(slurm_file, "a") as f:
            f.write("SLURM env variables for the job writing to this directory:\n")
            slurm_info = {k: v for k, v in os.environ.items() if k.startswith("SLURM_")}
            f.write(json.dumps(slurm_info, indent=4))
        # print the pycls path to make sure we're running in the copied code base
        print("Path to pycls", pycls.__file__)
        # load the config
        cfg.merge_from_file(cfg_file)
        cfg.OUT_DIR = out_dir
        config.assert_cfg()
        cfg.freeze()
        # run the trainer
        dist.setup_distributed()
        if self.run_mode == "train":
            trainer.train_model()
        elif self.run_mode == "test":
            trainer.test_model()
        elif self.run_mode == "time":
            trainer.time_model()


def sweep_launch():
    """Launch sweep on a SLURM managed cluster."""
    cfg = sweep_cfg.SETUP.BASE_CFG
    launch = cfg.LAUNCH
    sweep_dir = os.path.abspath(os.path.join(sweep_cfg.ROOT_DIR, sweep_cfg.NAME))
    cfgs_dir = os.path.join(sweep_dir, "cfgs")
    n_cfgs = len([c for c in os.listdir(cfgs_dir) if c.endswith(".yaml")])
    folder = f"{sweep_dir}/logs/slurm/%j"
    # setup executor
    kwargs = dict(folder=folder, slurm_max_num_timeout=launch.MAX_RETRY)
    executor = submitit.AutoExecutor(**kwargs)
    num_gpus_per_node = min(cfg.NUM_GPUS, cfg.MAX_GPUS_PER_NODE)
    executor.update_parameters(
        mem_gb=launch.MEM_PER_GPU * num_gpus_per_node,
        gpus_per_node=num_gpus_per_node,
        tasks_per_node=num_gpus_per_node,
        cpus_per_task=launch.CPUS_PER_GPU,
        nodes=max(1, cfg.NUM_GPUS // cfg.MAX_GPUS_PER_NODE),
        timeout_min=launch.TIME_LIMIT,
        name=sweep_cfg.NAME,
        slurm_partition=launch.PARTITION,
        slurm_comment=launch.COMMENT,
        slurm_constraint=launch.GPU_TYPE,
        slurm_additional_parameters={"mail-user": launch.EMAIL, "mail-type": "END"},
        slurm_array_parallelism=sweep_cfg.PARALLEL_JOBS,
    )
    # submit sweep
    port_range = cfg.PORT_RANGE
    ports = [random.randint(port_range[0], port_range[1]) for _ in range(n_cfgs)]
    run_mode = sweep_cfg.RUN_MODE
    trainer = SubmititRunner(sweep_dir=sweep_dir, run_mode=run_mode, ports=ports)
    jobs = executor.map_array(trainer, list(range(n_cfgs)))
    print(f"Submitted jobs {[job.job_id for job in jobs]} with sweep_dir: {sweep_dir}")


def sweep_setup_and_launch():
    """Setup and launch sweep on a SLURM managed cluster.
    
    Note: This function copies pycls to the sweep dir for code isolation and then calls
        sweep_launch() from the copied directory.
    """
    # Get and check directory and script locations
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sweep_dir = os.path.abspath(os.path.join(sweep_cfg.ROOT_DIR, sweep_cfg.NAME))

    assert os.path.exists(sweep_dir), "Sweep dir {} invalid".format(sweep_dir)

    # Replace path to be relative to copy of pycls
    pycls_copy_dir = os.path.join(sweep_dir, "pycls")
    pycls_dir = os.path.join(current_dir, "..")
    # Prepare command to copy pycls to sweep_dir/pycls
    cmd_to_copy_pycls = "cp -R {}/ {}".format(pycls_dir, pycls_copy_dir)
    print("Cmd to copy pycls:", cmd_to_copy_pycls)
    # Prepare launch command
    cmd_to_launch_sweep = (
        f'bash -c "source activate {sweep_cfg.CONDA_ENV} && '
        f"{pycls_copy_dir}/tools/sweep_launch.py --stage launch "
        f'--sweep-cfg {sweep_cfg.SWEEP_CFG_FILE}"'
    )
    print("\nCmd to launch sweep:", cmd_to_launch_sweep)
    # Prompt user to resume or launch sweep
    if os.path.exists(pycls_copy_dir):
        print("\nSweep exists! Relaunch ONLY if no jobs are running!")
        print("\nRelaunch sweep? [relaunch/n]")
        if input().lower() == "relaunch":
            # Change the cwd to pycls_copy_dir
            os.chdir(pycls_copy_dir)
            os.system(cmd_to_launch_sweep)
    else:
        print("\nLaunch sweep? [y/n]")
        if input().lower() == "y":
            os.system(cmd_to_copy_pycls)
            # Change the cwd to pycls_copy_dir
            os.chdir(pycls_copy_dir)
            os.system(cmd_to_launch_sweep)


def main():
    desc = "Launch a sweep on the cluster."
    stage = load_cfg_fom_args(desc, parse_stage=True)
    print(f"Running stage '{stage}'")
    sweep_cfg.freeze()
    if stage == "setup":
        sweep_setup_and_launch()
    else:
        sweep_launch()


if __name__ == "__main__":
    main()

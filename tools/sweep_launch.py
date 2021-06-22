#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Launch sweep on a SLURM managed cluster."""

import os

import pycls.sweep.config as sweep_config
from pycls.sweep.config import sweep_cfg


_SBATCH_CMD = (
    "sbatch"
    "  --job-name={name}"
    "  --partition={partition}"
    "  --gpus={num_gpus}"
    "  --constraint={gpu_type}"
    "  --mem={mem}GB"
    "  --cpus-per-task={cpus}"
    "  --array=0-{last_job}%{parallel_jobs}"
    "  --output={sweep_dir}/logs/sbatch/%A_%a.out"
    "  --error={sweep_dir}/logs/sbatch/%A_%a.out"
    "  --time={time_limit}"
    '  --comment="{comment}"'
    "  --signal=B:USR1@300"
    "  --nodes=1"
    "  --open-mode=append"
    "  --ntasks-per-node=1"
    "  {current_dir}/sweep_launch_job.py"
    "  --conda-env {conda_env}"
    "  --script-path {script_path}"
    "  --script-mode {script_mode}"
    "  --cfgs-dir {cfgs_dir}"
    "  --pycls-dir {pycls_dir}"
    "  --logs-dir {logs_dir}"
    "  --max-retry {max_retry}"
)


def sweep_launch():
    """Launch sweep on a SLURM managed cluster."""
    launch_cfg = sweep_cfg.LAUNCH
    # Get and check directory and script locations
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sweep_dir = os.path.abspath(os.path.join(sweep_cfg.ROOT_DIR, sweep_cfg.NAME))
    cfgs_dir = os.path.join(sweep_dir, "cfgs")
    logs_dir = os.path.join(sweep_dir, "logs")
    sbatch_dir = os.path.join(logs_dir, "sbatch")
    script_path = os.path.abspath("tools/run_net.py")
    assert os.path.exists(sweep_dir), "Sweep dir {} invalid".format(sweep_dir)
    assert os.path.exists(script_path), "Script path {} invalid".format(script_path)
    n_cfgs = len([c for c in os.listdir(cfgs_dir) if c.endswith(".yaml")])
    # Replace path to be relative to copy of pycls
    pycls_copy_dir = os.path.join(sweep_dir, "pycls")
    pycls_dir = os.path.abspath(os.path.join(current_dir, ".."))
    script_path = script_path.replace(pycls_dir, pycls_copy_dir)
    current_dir = current_dir.replace(pycls_dir, pycls_copy_dir)
    # Prepare command to copy pycls to sweep_dir/pycls
    cmd_to_copy_pycls = "cp -R {}/ {}".format(pycls_dir, pycls_copy_dir)
    print("Cmd to copy pycls:", cmd_to_copy_pycls)
    # Prepare launch command
    cmd_to_launch_sweep = _SBATCH_CMD.format(
        name=sweep_cfg.NAME,
        partition=launch_cfg.PARTITION,
        num_gpus=launch_cfg.NUM_GPUS,
        gpu_type=launch_cfg.GPU_TYPE,
        mem=launch_cfg.MEM_PER_GPU * launch_cfg.NUM_GPUS,
        cpus=launch_cfg.CPUS_PER_GPU * launch_cfg.NUM_GPUS,
        last_job=n_cfgs - 1,
        parallel_jobs=launch_cfg.PARALLEL_JOBS,
        time_limit=launch_cfg.TIME_LIMIT,
        comment=launch_cfg.COMMENT,
        sweep_dir=sweep_dir,
        current_dir=current_dir,
        conda_env=launch_cfg.CONDA_ENV,
        script_path=script_path,
        script_mode=launch_cfg.MODE,
        cfgs_dir=cfgs_dir,
        pycls_dir=pycls_copy_dir,
        logs_dir=logs_dir,
        max_retry=launch_cfg.MAX_RETRY,
    )
    print("Cmd to launch sweep:", cmd_to_launch_sweep.replace("  ", "\n  "), sep="\n\n")
    # Prompt user to resume or launch sweep
    if os.path.exists(sbatch_dir):
        print("\nSweep exists! Relaunch ONLY if no jobs are running!")
        print("\nRelaunch sweep? [relaunch/n]")
        if input().lower() == "relaunch":
            os.system(cmd_to_launch_sweep)
    else:
        print("\nLaunch sweep? [y/n]")
        if input().lower() == "y":
            os.makedirs(sbatch_dir, exist_ok=False)
            os.system(cmd_to_copy_pycls)
            os.system(cmd_to_launch_sweep)


def main():
    desc = "Launch a sweep on the cluster."
    sweep_config.load_cfg_fom_args(desc)
    sweep_cfg.freeze()
    sweep_launch()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Launch sweep on a SLURM managed cluster.

Submitit based launcher which can be used to submit multi or single node
multi GPU jobs to SLURM.
"""

import os

from pycls.sweep.config import load_cfg_fom_args, sweep_cfg


def sweep_launch():
    """Launch sweep on a SLURM managed cluster."""
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
        f'bash -c "source activate {sweep_cfg.LAUNCH.CONDA_ENV} && '
        f"{pycls_copy_dir}/tools/sweep_launch_job.py "
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
    load_cfg_fom_args(desc)
    sweep_cfg.freeze()
    sweep_launch()


if __name__ == "__main__":
    main()

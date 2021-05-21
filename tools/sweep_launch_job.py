#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Launch a job on SLURM managed cluster. Should only be called from sweep_launch.py"""

import argparse
import json
import os
import signal
import subprocess
import sys
from datetime import datetime


def prt(*args, **kwargs):
    """Wrapper for print that prepends a timestamp and flushes output."""
    print("[{}]".format(str(datetime.now())), *args, flush=True, **kwargs)


def run_os_cmd(cmd):
    """Runs commands in bash environment in foreground."""
    os.system('bash -c "{}"'.format(cmd))


def requeue_job():
    job_id = os.environ["SLURM_ARRAY_JOB_ID"]
    task_id = os.environ["SLURM_ARRAY_TASK_ID"]
    cmd_to_req = "scontrol requeue {}_{}".format(job_id, task_id)
    prt("Requeuing job using cmd: {}".format(cmd_to_req))
    os.system(cmd_to_req)
    prt("Requeued job {}. Exiting.\n\n".format(job_id))
    sys.exit(0)


def sigusr1_handler(signum, _):
    """Handles SIGUSR1 that is sent before a job is killed by requeuing it."""
    prt("Caught SIGUSR1 with code {}".format(signum))
    requeue_job()


def sigterm_handler(signum, _):
    """Handles SIGTERM that is sent before a job is preempted by bypassing it."""
    prt("Caught SIGTERM with code {}".format(signum))
    prt("Bypassing SIGTERM")


def main():
    # Parse arguments
    desc = "Launch a job on SLURM cluster. Should only be called from sweep_launch.py"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("--conda-env", required=True)
    parser.add_argument("--script-path", required=True)
    parser.add_argument("--cfgs-dir", required=True)
    parser.add_argument("--pycls-dir", required=True)
    parser.add_argument("--logs-dir", required=True)
    parser.add_argument("--max-retry", required=True, type=int)
    args = parser.parse_args()
    prt("Called with args: {}".format(args))
    # Attach signal handlers for SIGUSR1 and SIGTERM
    signal.signal(signal.SIGUSR1, sigusr1_handler)
    signal.signal(signal.SIGTERM, sigterm_handler)
    # Print info about run
    job_id = os.environ["SLURM_ARRAY_JOB_ID"]
    task_id = os.environ["SLURM_ARRAY_TASK_ID"]
    prt("Job array master job ID: {}".format(job_id))
    prt("Job array task ID (index): {}".format(task_id))
    prt("Running job on: {}".format(str(os.uname())))
    # Load what we need
    run_os_cmd("module purge")
    run_os_cmd("module load anaconda3")
    run_os_cmd("source deactivate")
    run_os_cmd("source activate {}".format(args.conda_env))
    # Get cfg_file to use
    cfg_files = sorted(f for f in os.listdir(args.cfgs_dir) if f.endswith(".yaml"))
    cfg_file = os.path.join(args.cfgs_dir, cfg_files[int(task_id)])
    prt("Using cfg_file: {}".format(cfg_file))
    # Create out_dir
    out_dir = os.path.join(args.logs_dir, "{:06}".format(int(task_id)))
    os.makedirs(out_dir, exist_ok=True)
    prt("Using out_dir: {}".format(out_dir))
    # Create slurm_file with SLURM info
    slurm_file = os.path.join(out_dir, "SLURM.txt")
    with open(slurm_file, "a") as f:
        f.write("SLURM env variables for the job writing to this directory:\n")
        slurm_info = {k: os.environ[k] for k in os.environ if k.startswith("SLURM_")}
        f.write(json.dumps(slurm_info, indent=4))
    prt("Dumped SLURM job info to {}".format(slurm_file))
    # Set PYTHONPATH to pycls copy for sweep
    os.environ["PYTHONPATH"] = args.pycls_dir
    prt("Using PYTHONPATH={}".format(args.pycls_dir))
    # Generate srun command to launch
    cmd_to_run = (
        "srun"
        "  --output {out_dir}/stdout.log"
        "  --error {out_dir}/stderr.log"
        "  python {script}"
        "  --cfg {cfg}"
        "  OUT_DIR {out_dir}"
    ).format(out_dir=out_dir, script=args.script_path, cfg=cfg_file)
    prt("Running cmd:\n", cmd_to_run.replace("  ", "\n  "))
    # Run command in background using subprocess and wait so that signals can be caught
    p = subprocess.Popen(cmd_to_run, shell=True)
    prt("Waiting for job to complete")
    p.wait()
    prt("Completed waiting. Return code for job: {}".format(p.returncode))
    if p.returncode != 0:
        retry_file = os.path.join(out_dir, "RETRY.txt")
        with open(retry_file, "a") as f:
            f.write("Encountered non-zero exit code\n")
        with open(retry_file, "r") as f:
            retry_count = len(f.readlines()) - 1
        prt("Retry count for job: {}".format(retry_count))
        if retry_count < args.max_retry:
            requeue_job()


if __name__ == "__main__":
    main()

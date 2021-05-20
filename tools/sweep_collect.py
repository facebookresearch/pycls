#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


"""Collect results of a sweep."""

import functools
import json
import multiprocessing
import os

import pycls.core.checkpoint as cp
import pycls.core.logging as logging
import pycls.sweep.config as sweep_config
from pycls.sweep.config import sweep_cfg


# Skip over these data types as they make sweep logs too large
_DATA_TYPES_TO_SKIP = ["train_iter", "test_iter"]


def load_data(log_file):
    """Loads and sorts log data or returns None."""
    data = logging.load_log_data(log_file, _DATA_TYPES_TO_SKIP)
    data = logging.sort_log_data(data)
    err_file = log_file.replace("stdout.log", "stderr.log")
    data["log_file"] = log_file
    data["err_file"] = err_file
    with open(err_file, "r") as f:
        data["err"] = f.read()
    return data


def sweep_collect():
    """Collects results of a sweep."""
    # Get cfg and log files
    sweep_dir = os.path.join(sweep_cfg.ROOT_DIR, sweep_cfg.NAME)
    print("Collecting jobs for {:s}... ".format(sweep_dir))
    cfgs_dir = os.path.join(sweep_dir, "cfgs")
    logs_dir = os.path.join(sweep_dir, "logs")
    assert os.path.exists(cfgs_dir), "Cfgs dir {} not found".format(cfgs_dir)
    assert os.path.exists(logs_dir), "Logs dir {} not found".format(logs_dir)
    cfg_files = [c for c in os.listdir(cfgs_dir) if c.endswith(".yaml")]
    log_files = logging.get_log_files(logs_dir)[0]
    # Create worker pool for collecting jobs
    process_pool = multiprocessing.Pool(sweep_cfg.NUM_PROC)
    # Load the sweep and keep only non-empty data
    print("Collecting jobs...")
    sweep = list(process_pool.map(load_data, log_files))
    # Print basic stats for sweep status
    key = "test_epoch"
    epoch_ind = [d[key]["epoch_ind"][-1] if key in d else 0 for d in sweep]
    epoch_max = [d[key]["epoch_max"][-1] if key in d else 1 for d in sweep]
    epoch = ["{}/{}".format(i, m) for i, m in zip(epoch_ind, epoch_max)]
    epoch = [e.ljust(len(max(epoch, key=len))) for e in epoch]
    job_done = sum(i == m for i, m in zip(epoch_ind, epoch_max))
    for d, e, i, m in zip(sweep, epoch, epoch_ind, epoch_max):
        out_str = "  {} [{:3d}%] [{:}]" + (" [stderr]" if d["err"] else "")
        print(out_str.format(d["log_file"], int(i / m * 100), e))
    jobs_start = "jobs_started={}/{}".format(len(sweep), len(cfg_files))
    jobs_done = "jobs_done={}/{}".format(job_done, len(cfg_files))
    ep_done = "epochs_done={}/{}".format(sum(epoch_ind), sum(epoch_max))
    print("Status: {}, {}, {}".format(jobs_start, jobs_done, ep_done))
    # Save the sweep data
    sweep_file = os.path.join(sweep_dir, "sweep.json")
    print("Writing sweep data to: {}".format(sweep_file))
    with open(sweep_file, "w") as f:
        json.dump(sweep, f, sort_keys=True)
    # Clean up checkpoints after saving sweep data, if needed
    keep = sweep_cfg.COLLECT.CHECKPOINTS_KEEP
    cp_dirs = [f.replace("stdout.log", "checkpoints/") for f in log_files]
    delete_cps = functools.partial(cp.delete_checkpoints, keep=keep)
    num_cleaned = sum(process_pool.map(delete_cps, cp_dirs))
    print("Deleted {} total checkpoints".format(num_cleaned))


def main():
    desc = "Collect results of a sweep."
    sweep_config.load_cfg_fom_args(desc)
    sweep_cfg.freeze()
    sweep_collect()


if __name__ == "__main__":
    main()

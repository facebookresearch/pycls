#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Execute training operation on a classification model with code isolation."""

import argparse
import os
import sys

import pycls.core.config as config
from pycls.core.config import cfg


_LAUNCH_CMD = "python3 {script} --mode {mode} --cfg {cfg} {opts}"


def parse_args():
    """Parse command line options (mode and config)."""
    parser = argparse.ArgumentParser(description="Run a model.")
    help_s = "Config file location"
    parser.add_argument("--cfg", help=help_s, required=True, type=str)
    help_s = "See pycls/core/config.py for all options"
    parser.add_argument("opts", help=help_s, default=None, nargs=argparse.REMAINDER)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def main():
    """Execute operation (train, test, time, etc.)."""
    args = parse_args()
    config.load_cfg(args.cfg)
    cfg.merge_from_list(args.opts)
    config.assert_cfg()
    cfg.freeze()
    assert cfg.LAUNCH.MODE == "slurm", "Only slurm is supported for code isolation."
    # Get and check directory and script locations
    current_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = cfg.OUT_DIR
    # Replace path to be relative to copy of pycls
    pycls_copy_dir = os.path.join(out_dir, "pycls")
    pycls_dir = os.path.abspath(os.path.join(current_dir, ".."))
    script_path = os.path.abspath("tools/run_net.py")
    script_path = script_path.replace(pycls_dir, pycls_copy_dir)
    current_dir = current_dir.replace(pycls_dir, pycls_copy_dir)
    # Prepare command to copy pycls to sweep_dir/pycls
    cmd_to_copy_pycls = "cp -R {}/ {}".format(pycls_dir, pycls_copy_dir)
    print("Cmd to copy pycls:", cmd_to_copy_pycls)
    # Prepare launch command
    opts = " ".join(args.opts)
    cmd_to_launch_job = _LAUNCH_CMD.format(
        script=script_path, mode="train", cfg=args.cfg, opts=opts
    )
    print("Cmd to launch job:", cmd_to_launch_job)
    # Prompt user to resume or launch job
    if os.path.exists(out_dir):
        print("\nJob exists! Relaunch ONLY if no jobs are running!")
        print("\nRelaunch job? [relaunch/n]")
        if input().lower() == "relaunch":
            os.system(cmd_to_launch_job)
    else:
        print("\nLaunch job? [y/n]")
        if input().lower() == "y":
            os.makedirs(out_dir, exist_ok=False)
            os.environ["PYTHONPATH"] = pycls_copy_dir
            print("Using PYTHONPATH={}".format(pycls_copy_dir))
            os.system(cmd_to_copy_pycls)
            os.system(cmd_to_launch_job)


if __name__ == "__main__":
    main()

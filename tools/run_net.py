#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Execute various operations (train, test, time, etc.) on a classification model."""

import argparse
import sys

import pycls.core.builders as builders
import pycls.core.config as config
import pycls.core.distributed as dist
import pycls.core.net as net
import pycls.core.trainer as trainer
import pycls.models.scaler as scaler
from pycls.core.config import cfg


def parse_args():
    """Parse command line options (mode and config)."""
    parser = argparse.ArgumentParser(description="Run a model.")
    help_s, choices = "Run mode", ["info", "train", "test", "time", "scale"]
    parser.add_argument("--mode", help=help_s, choices=choices, required=True, type=str)
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
    mode = args.mode
    config.load_cfg(args.cfg)
    cfg.merge_from_list(args.opts)
    config.assert_cfg()
    cfg.freeze()
    if mode == "info":
        print(builders.get_model()())
        print("complexity:", net.complexity(builders.get_model()))
    elif mode == "train":
        dist.multi_proc_run(num_proc=cfg.NUM_GPUS, fun=trainer.train_model)
    elif mode == "test":
        dist.multi_proc_run(num_proc=cfg.NUM_GPUS, fun=trainer.test_model)
    elif mode == "time":
        dist.multi_proc_run(num_proc=cfg.NUM_GPUS, fun=trainer.time_model)
    elif mode == "scale":
        cfg.defrost()
        cx_orig = net.complexity(builders.get_model())
        scaler.scale_model()
        cx_scaled = net.complexity(builders.get_model())
        cfg_file = config.dump_cfg()
        print("Scaled config dumped to:", cfg_file)
        print("Original model complexity:", cx_orig)
        print("Scaled model complexity:", cx_scaled)


if __name__ == "__main__":
    main()

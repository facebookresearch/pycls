#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Execute various operations (train, test, time, etc.) on a classification model."""

import argparse
import os
import random
import sys

import pycls.core.config as config
import pycls.core.distributed as dist
import pycls.core.trainer as trainer
import torch
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


def _run_local(func):
    """Run a job locally on the current node.

    Sets up torch.distributed and launches multiple processes if NUM_GPUS > 1.
    Otherwise, runs the function in the same process.
    """
    assert config.get_num_nodes() == 1, "Cannot use run_local for multi node jobs"
    if config.get_num_gpus_per_node() > 1:
        master_port = random.randint(cfg.PORT_RANGE[0], cfg.PORT_RANGE[1])
        torch.multiprocessing.spawn(
            _run_local_distributed, args=(func, master_port, cfg), nprocs=cfg.NUM_GPUS
        )
    else:
        func()


def _run_local_distributed(local_rank, func, master_port, cfg_state):
    # this is run inside a new process, so the state of cfg is reset and
    # we set it back again
    cfg.update(**cfg_state)
    cfg.freeze()

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(local_rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(cfg.NUM_GPUS)

    dist.init_distributed()
    func()


def main():
    """Execute operation (train, test, time, etc.)."""
    args = parse_args()
    mode = args.mode
    config.load_cfg(args.cfg)
    cfg.merge_from_list(args.opts)
    config.assert_and_infer_cfg()
    cfg.freeze()
    trainer.run(mode, _run_local)


if __name__ == "__main__":
    main()

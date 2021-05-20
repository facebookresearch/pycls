#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Sample cfgs for a sweep using a sweep_cfg."""

import multiprocessing
import os

import numpy as np
import pycls.models.scaler as scaler
import pycls.sweep.config as sweep_config
import pycls.sweep.samplers as samplers
import yaml
from pycls.core.config import cfg, reset_cfg
from pycls.core.timer import Timer
from pycls.sweep.config import sweep_cfg


def sample_cfgs(seed):
    """Samples chunk configs and return those that are unique and valid."""
    # Fix RNG seed (every call to this function should use a unique seed)
    np.random.seed(seed)
    setup_cfg = sweep_cfg.SETUP
    cfgs = {}
    for _ in range(setup_cfg.CHUNK_SIZE):
        # Sample parameters [key, val, ...] list based on the samplers
        params = samplers.sample_parameters(setup_cfg.SAMPLERS)
        # Check if config is unique, if not continue
        key = zip(params[0::2], params[1::2])
        key = " ".join(["{} {}".format(k, v) for k, v in key])
        if key in cfgs:
            continue
        # Generate config from parameters
        reset_cfg()
        cfg.merge_from_other_cfg(setup_cfg.BASE_CFG)
        cfg.merge_from_list(params)
        # Check if config is valid, if not continue
        is_valid = samplers.check_regnet_constraints(setup_cfg.CONSTRAINTS)
        if not is_valid:
            continue
        # Special logic for dealing w model scaling (side effect is to standardize cfg)
        scaler.scale_model()
        # Check if config is valid, if not continue
        is_valid = samplers.check_complexity_constraints(setup_cfg.CONSTRAINTS)
        if not is_valid:
            continue
        # Set config description to key
        cfg.DESC = key
        # Store copy of config if unique and valid
        cfgs[key] = cfg.clone()
        # Stop sampling if already reached quota
        if len(cfgs) == setup_cfg.NUM_CONFIGS:
            break
    return cfgs


def dump_cfg(cfg_file, cfg):
    """Dumps the config to the specified location."""
    with open(cfg_file, "w") as f:
        cfg.dump(stream=f)


def sweep_setup():
    """Samples cfgs for the sweep."""
    setup_cfg = sweep_cfg.SETUP
    # Create output directories
    sweep_dir = os.path.join(sweep_cfg.ROOT_DIR, sweep_cfg.NAME)
    cfgs_dir = os.path.join(sweep_dir, "cfgs")
    logs_dir = os.path.join(sweep_dir, "logs")
    print("Sweep directory is: {}".format(sweep_dir))
    assert not os.path.exists(logs_dir), "Sweep already started: " + sweep_dir
    if os.path.exists(logs_dir) or os.path.exists(cfgs_dir):
        print("Overwriting sweep which has not yet launched")
    os.makedirs(sweep_dir, exist_ok=True)
    os.makedirs(cfgs_dir, exist_ok=True)
    # Dump the original sweep_cfg
    sweep_cfg_file = os.path.join(sweep_dir, "sweep_cfg.yaml")
    os.system("cp {} {}".format(sweep_cfg.SWEEP_CFG_FILE, sweep_cfg_file))
    # Create worker pool for sampling and saving configs
    n_proc, chunk = sweep_cfg.NUM_PROC, setup_cfg.CHUNK_SIZE
    process_pool = multiprocessing.Pool(n_proc)
    # Fix random number generator seed and generate per chunk seeds
    np.random.seed(setup_cfg.RNG_SEED)
    n_chunks = int(np.ceil(setup_cfg.NUM_SAMPLES / chunk))
    chunk_seeds = np.random.choice(1000000, size=n_chunks, replace=False)
    # Sample configs in chunks using multiple workers each with a unique seed
    info_str = "Number configs sampled: {}, configs kept: {} [t={:.2f}s]"
    n_samples, n_cfgs, i, cfgs, timer = 0, 0, 0, {}, Timer()
    while n_samples < setup_cfg.NUM_SAMPLES and n_cfgs < setup_cfg.NUM_CONFIGS:
        timer.tic()
        seeds = chunk_seeds[i * n_proc : i * n_proc + n_proc]
        cfgs_all = process_pool.map(sample_cfgs, seeds)
        cfgs = dict(cfgs, **{k: v for d in cfgs_all for k, v in d.items()})
        n_samples, n_cfgs, i = n_samples + chunk * n_proc, len(cfgs), i + 1
        timer.toc()
        print(info_str.format(n_samples, n_cfgs, timer.total_time))
    # Randomize cfgs order and subsample if oversampled
    keys, cfgs = list(cfgs.keys()), list(cfgs.values())
    n_cfgs = min(n_cfgs, setup_cfg.NUM_CONFIGS)
    ids = np.random.choice(len(cfgs), n_cfgs, replace=False)
    keys, cfgs = [keys[i] for i in ids], [cfgs[i] for i in ids]
    # Save the cfgs and a cfgs_summary
    timer.tic()
    cfg_names = ["{:06}.yaml".format(i) for i in range(n_cfgs)]
    cfgs_summary = {cfg_name: key for cfg_name, key in zip(cfg_names, keys)}
    with open(os.path.join(sweep_dir, "cfgs_summary.yaml"), "w") as f:
        yaml.dump(cfgs_summary, f, width=float("inf"))
    cfg_files = [os.path.join(cfgs_dir, cfg_name) for cfg_name in cfg_names]
    process_pool.starmap(dump_cfg, zip(cfg_files, cfgs))
    timer.toc()
    print(info_str.format(n_samples, n_cfgs, timer.total_time))


def main():
    desc = "Set up sweep by generating job configs."
    sweep_config.load_cfg_fom_args(desc)
    sweep_cfg.freeze()
    sweep_setup()


if __name__ == "__main__":
    main()

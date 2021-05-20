#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Configuration file (powered by YACS)."""

import argparse
import getpass
import multiprocessing
import os
import sys

from pycls.core.config import cfg
from pycls.sweep.samplers import validate_sampler
from yacs.config import CfgNode as CfgNode


# Example usage: from sweep.config import sweep_cfg
sweep_cfg = _C = CfgNode()


# ------------------------------ General sweep options ------------------------------- #

# Sweeps root directory where all sweep output subdirectories will be placed
_C.ROOT_DIR = "/checkpoint/{}/sweeps/".format(getpass.getuser())

# Sweep name must be unique per sweep and defines the output subdirectory
_C.NAME = ""

# Optional description of a sweep useful to keep track of sweeps
_C.DESC = ""

# Number of processes to use for various sweep steps except for running jobs
_C.NUM_PROC = multiprocessing.cpu_count()

# Automatically overwritten to the file from which the sweep_cfg is loaded
_C.SWEEP_CFG_FILE = ""


# ------------------------------- Sweep setup options -------------------------------- #
_C.SETUP = CfgNode()

# Max number of unique job configs to generate
_C.SETUP.NUM_CONFIGS = 0

# Max number of attempts for generating NUM_CONFIGS valid configs
_C.SETUP.NUM_SAMPLES = 1000000

# Specifies the chunk size to use per process while sampling configs
_C.SETUP.CHUNK_SIZE = 5000

# Random seed for generating job configs
_C.SETUP.RNG_SEED = 0

# Base config for all jobs, any valid config option in core.config is valid here
_C.SETUP.BASE_CFG = cfg.clone()

# Samplers to use for generating job configs (see SAMPLERS defined toward end of file)
# SETUP.SAMPLERS should consists of a dictionary of SAMPLERS
# Each dict key should be a valid parameter in the BASE_CFG (e.g. "MODEL.DEPTH")
# Each dict val should be a valid SAMPLER that defines how to sample (e.g. INT_SAMPLER)
# See the example sweep configs for more usage information
_C.SETUP.SAMPLERS = CfgNode(new_allowed=True)

# Constraints on generated configs
_C.SETUP.CONSTRAINTS = CfgNode()

# Complexity constraints CX on models specified as a [LOW, HIGH] range, e.g. [0, 1.0e+6]
# If LOW == HIGH == 0 for a given complexity constraint that constraint is not applied
# For RegNets, if flops<F (B), setting params<3+5.5F and acts<6.5*sqrt(F) (M) works well
_C.SETUP.CONSTRAINTS.CX = CfgNode()
_C.SETUP.CONSTRAINTS.CX.FLOPS = [0, 0]
_C.SETUP.CONSTRAINTS.CX.PARAMS = [0, 0]
_C.SETUP.CONSTRAINTS.CX.ACTS = [0, 0]

# RegNet specific constraints
_C.SETUP.CONSTRAINTS.REGNET = CfgNode()
_C.SETUP.CONSTRAINTS.REGNET.NUM_STAGES = [4, 4]


# ------------------------------- Sweep launch options ------------------------------- #
_C.LAUNCH = CfgNode()

# Actual script to run for each job (should be in pycls directory)
_C.LAUNCH.SCRIPT = "tools/train_net.py"

# CONDA environment to use for jobs (defaults to current environment)
_C.LAUNCH.CONDA_ENV = os.environ["CONDA_PREFIX"]

# Max number of parallel jobs to run (subject to resource constraints)
_C.LAUNCH.PARALLEL_JOBS = 128

# Max number of times to retry a job
_C.LAUNCH.MAX_RETRY = 3

# Optional comment for sbatch (may be required when using high priority partitions)
_C.LAUNCH.COMMENT = ""

# Resources to request per job
_C.LAUNCH.NUM_GPUS = 1
_C.LAUNCH.CPUS_PER_GPU = 10
_C.LAUNCH.MEM_PER_GPU = 60
_C.LAUNCH.PARTITION = "learnfair"
_C.LAUNCH.GPU_TYPE = "volta"
_C.LAUNCH.TIME_LIMIT = 4200


# ------------------------------ Sweep collect options ------------------------------- #
_C.COLLECT = CfgNode()

# Determines which checkpoints to keep, supported options are "all", "last", or "none"
_C.COLLECT.CHECKPOINTS_KEEP = "last"


# ------------------------------ Sweep analysis options ------------------------------ #
_C.ANALYZE = CfgNode()

# List of metrics for which to generate analysis, may be any valid field in log
# An example metric is "cfg.OPTIM.BASE_LR" or "complexity.acts" or "test_epoch.mem"
# Some metrics have shortcuts defined, for example "error" or "lr", see analysis.py
_C.ANALYZE.METRICS = []

# List of complexity metrics for which to generate analysis, same format as metrics
_C.ANALYZE.COMPLEXITY = ["flops", "params", "acts"]

# Controls number of plots of various types to show in analysis
_C.ANALYZE.PLOT_METRIC_VALUES = True
_C.ANALYZE.PLOT_METRIC_TRENDS = True
_C.ANALYZE.PLOT_COMPLEXITY_VALUES = True
_C.ANALYZE.PLOT_COMPLEXITY_TRENDS = True
_C.ANALYZE.PLOT_CURVES_BEST = 0
_C.ANALYZE.PLOT_CURVES_WORST = 0
_C.ANALYZE.PLOT_MODELS_BEST = 0
_C.ANALYZE.PLOT_MODELS_WORST = 0

# Undocumented "use at your own risk" feature used to "pre-filter" a sweep
_C.ANALYZE.PRE_FILTERS = CfgNode(new_allowed=True)

# Undocumented "use at your own risk" feature used to "split" a sweep into sets
_C.ANALYZE.SPLIT_FILTERS = CfgNode(new_allowed=True)

# Undocumented "use at your own risk" feature used to load other sweeps
_C.ANALYZE.EXTRA_SWEEP_NAMES = []


# --------------------------- Samplers for SETUP.SAMPLERS ---------------------------- #
SAMPLERS = CfgNode()

# Sampler for uniform sampling from a list of values
SAMPLERS.VALUE_SAMPLER = CfgNode()
SAMPLERS.VALUE_SAMPLER.TYPE = "value_sampler"
SAMPLERS.VALUE_SAMPLER.VALUES = []

# Sampler for floats with RAND_TYPE sampling in RANGE quantized to QUANTIZE
# RAND_TYPE can be "uniform", "log_uniform", "power2_uniform", "normal", "log_normal"
# Uses the closed interval RANGE = [LOW, HIGH] (so the HIGH value can be sampled)
# Note that both LOW and HIGH must be divisible by QUANTIZE
# For the (clipped) normal samplers mu/sigma are set so ~99.7% of samples are in RANGE
SAMPLERS.FLOAT_SAMPLER = CfgNode()
SAMPLERS.FLOAT_SAMPLER.TYPE = "float_sampler"
SAMPLERS.FLOAT_SAMPLER.RAND_TYPE = "uniform"
SAMPLERS.FLOAT_SAMPLER.RANGE = [0.0, 0.0]
SAMPLERS.FLOAT_SAMPLER.QUANTIZE = 0.00001

# Sampler for ints with RAND_TYPE sampling in RANGE quantized to QUANTIZE
# RAND_TYPE can be "uniform", "log_uniform", "power2_uniform", "normal", "log_normal"
# Uses the closed interval RANGE = [LOW, HIGH] (so the HIGH value can be sampled)
# Note that both LOW and HIGH must be divisible by QUANTIZE
# For the (clipped) normal samplers mu/sigma are set so ~99.7% of samples are in RANGE
SAMPLERS.INT_SAMPLER = CfgNode()
SAMPLERS.INT_SAMPLER.TYPE = "int_sampler"
SAMPLERS.INT_SAMPLER.RAND_TYPE = "uniform"
SAMPLERS.INT_SAMPLER.RANGE = [0, 0]
SAMPLERS.INT_SAMPLER.QUANTIZE = 1

# Sampler for a list of LENGTH items each sampled independently by the ITEM_SAMPLER
# The ITEM_SAMPLER can be any sampler (like INT_SAMPLER or even anther LIST_SAMPLER)
SAMPLERS.LIST_SAMPLER = CfgNode()
SAMPLERS.LIST_SAMPLER.TYPE = "list_sampler"
SAMPLERS.LIST_SAMPLER.LENGTH = 0
SAMPLERS.LIST_SAMPLER.ITEM_SAMPLER = CfgNode(new_allowed=True)

# RegNet Sampler with ranges for REGNET params (see base config for meaning of params)
# This sampler simply allows a compact specification of a number of RegNet params
# QUANTIZE for each params below is fixed to: 1, 8, 0.1, 0.001, 8, 1/128, respectively
# RAND_TYPE for each is fixed to uni, log, log, log, power2_or_log, power2, respectively
# Default parameter ranges are set to generate fairly good performing models up to 16GF
# For models over 16GF, higher ranges for GROUP_W, W0, and WA are necessary
# If including this sampler set SETUP.CONSTRAINTS as needed
SAMPLERS.REGNET_SAMPLER = CfgNode()
SAMPLERS.REGNET_SAMPLER.TYPE = "regnet_sampler"
SAMPLERS.REGNET_SAMPLER.DEPTH = [12, 28]
SAMPLERS.REGNET_SAMPLER.W0 = [8, 256]
SAMPLERS.REGNET_SAMPLER.WA = [8.0, 256.0]
SAMPLERS.REGNET_SAMPLER.WM = [2.0, 3.0]
SAMPLERS.REGNET_SAMPLER.GROUP_W = [8, 128]
SAMPLERS.REGNET_SAMPLER.BOT_MUL = [1.0, 1.0]


# -------------------------------- Utility functions --------------------------------- #
def load_cfg(sweep_cfg_file):
    """Loads config from specified sweep_cfg_file."""
    _C.merge_from_file(sweep_cfg_file)
    _C.SWEEP_CFG_FILE = os.path.abspath(sweep_cfg_file)
    # Check for required arguments
    err_msg = "{} has to be specified."
    assert _C.ROOT_DIR, err_msg.format("ROOT_DIR")
    assert _C.NAME, err_msg.format("NAME")
    assert _C.SETUP.NUM_CONFIGS, err_msg.format("SETUP.NUM_CONFIGS")
    # Check for allowed arguments
    opts = ["all", "last", "none"]
    err_msg = "COLLECT.CHECKPOINTS_KEEP has to be one of {}".format(opts)
    assert _C.COLLECT.CHECKPOINTS_KEEP in opts, err_msg
    # Setup the base config (note: this only alters the loaded global cfg)
    cfg.merge_from_other_cfg(_C.SETUP.BASE_CFG)
    # Load and validate each sampler against one of the SAMPLERS templates
    for param, sampler in _C.SETUP.SAMPLERS.items():
        _C.SETUP.SAMPLERS[param] = load_sampler(param, sampler)


def load_sampler(param, sampler):
    """Loads and validates a sampler against one of the SAMPLERS templates."""
    sampler_type = sampler.TYPE.upper() if "TYPE" in sampler else None
    err_msg = "Sampler for '{}' has an unknown or missing TYPE:\n{}"
    assert sampler_type in SAMPLERS, err_msg.format(param, sampler)
    full_sampler = SAMPLERS[sampler_type].clone()
    full_sampler.merge_from_other_cfg(sampler)
    validate_sampler(param, full_sampler)
    if sampler_type == "LIST_SAMPLER":
        full_sampler.ITEM_SAMPLER = load_sampler(param, sampler.ITEM_SAMPLER)
    return full_sampler


def load_cfg_fom_args(description="Config file options."):
    """Loads sweep cfg from command line argument."""
    parser = argparse.ArgumentParser(description=description)
    help_str = "Path to sweep_cfg file"
    parser.add_argument("--sweep-cfg", help=help_str, required=True)
    args = parser.parse_args()
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    load_cfg(args.sweep_cfg)

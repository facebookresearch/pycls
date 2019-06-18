#!/usr/bin/env python3

"""Config system (based on Detectron's)."""

# TODO(ilijar): net naming (e.g. RESNET -> RES_NET)
# TODO(ilijar): configurable train and test resolution
# TODO(ilijar): don't include stem in stage config lists
# TODO(ilijar): remove deprecated keys from configs
# TODO(ilijar): remove unused keys (e.g. alt transforms)

from ast import literal_eval

import yaml

from pycls.utils.collections import AttrDict


# Global config object
__C = AttrDict()
# Example usage:
#   from core.config import cfg
cfg = __C


# ---------------------------------------------------------------------------- #
# Model options
# ---------------------------------------------------------------------------- #
__C.MODEL = AttrDict()

# Model type to use
__C.MODEL.TYPE = ''

# Number of weight layers
__C.MODEL.DEPTH = 0

# Number of classes
__C.MODEL.NUM_CLASSES = 10

# Loss function (see pycls/models/loss.py for options)
__C.MODEL.LOSS_FUN = 'cross_entropy'


# ---------------------------------------------------------------------------- #
# Batch norm options
# ---------------------------------------------------------------------------- #
__C.BN = AttrDict()

# BN epsilon
__C.BN.EPSILON = 1e-5

# BN momentum (BN momentum in PyTorch = 1 - BN momentum in Caffe2)
__C.BN.MOMENTUM = 0.1

# Precise BN stats
__C.BN.USE_PRECISE_STATS = False
__C.BN.NUM_SAMPLES_PRECISE = 1024


# ---------------------------------------------------------------------------- #
# VGG options
# ---------------------------------------------------------------------------- #
__C.VGG = AttrDict()

# Number of stages
__C.VGG.NUM_STAGES = 5

# Indices of stages whose first block uses stride 2 conv
__C.VGG.STRIDE2_INDS = []

# Indices of stages after which max pooling is performed
__C.VGG.MAX_POOL_INDS = [0, 1, 2, 3, 4]

# Depths multiplier (relative to the original VGG)
__C.VGG.DS_MULT = 1.0

# Widths multiplier (relative to the original VGG)
__C.VGG.WS_MULT = 1.0


# ---------------------------------------------------------------------------- #
# ResNet options
# ---------------------------------------------------------------------------- #
__C.RESNET = AttrDict()

# Transformation function (see pycls/models/resnet.py for options)
__C.RESNET.TRANS_FUN = 'basic_transform'

# Number of groups to use (1 -> ResNet; > 1 -> ResNeXt)
__C.RESNET.NUM_GROUPS = 1

# Width of each group (64 -> ResNet; 4 -> ResNeXt)
__C.RESNET.WIDTH_PER_GROUP = 64

# Apply stride to 1x1 conv (True -> MSRA; False -> fb.torch)
__C.RESNET.STRIDE_1X1 = True

# Alternative transformation function (e.g. to use in a subset of blocks)
__C.RESNET.ALT_TRANS_FUN = ''

# Blocks that use the alternative transformation function (e.g. res4_4)
__C.RESNET.ALT_TRANS_FUN_BLOCKS = []

# Initialize the "gamma" scale parameters of the final BN operation of each
# residual block transformation function to zero
__C.RESNET.ZERO_INIT_FINAL_TRANSFORM_BN = False


# ---------------------------------------------------------------------------- #
# UniNet options
# ---------------------------------------------------------------------------- #
__C.UNINET = AttrDict()

# Block type (applies to stages > 1; stage 1 is always fixed)
__C.UNINET.BLOCK_TYPE = 'plain_block'

# Depth for each stage (number of blocks in the stage)
__C.UNINET.DEPTHS = []

# Width for each stage (width of each block in the stage)
__C.UNINET.WIDTHS = []

# Strides for each stage (applies to the first block of each stage)
__C.UNINET.STRIDES = []


# ---------------------------------------------------------------------------- #
# Optimizer options
# ---------------------------------------------------------------------------- #
__C.OPTIM = AttrDict()

# Base learning rate
__C.OPTIM.BASE_LR = 0.1

# Learning rate policy select from {'cos', 'exp', 'steps'}
__C.OPTIM.LR_POLICY = 'cos'

# Exponential decay factor
__C.OPTIM.GAMMA = 0.1

# Steps for 'steps' policy (in epochs)
__C.OPTIM.STEPS = []

# Learning rate multiplier for 'steps' policy
__C.OPTIM.LR_MULT = 0.1

# Maximal number of epochs
__C.OPTIM.MAX_EPOCH = 200

# Momentum
__C.OPTIM.MOMENTUM = 0.9

# Momentum dampening
__C.OPTIM.DAMPENING = 0.0

# Nesterov momentum
__C.OPTIM.NESTEROV = True

# L2 regularization
__C.OPTIM.WEIGHT_DECAY = 5e-4

# Start the warm up from OPTIM.BASE_LR * OPTIM.WARMUP_FACTOR
__C.OPTIM.WARMUP_FACTOR = 0.1

# Gradually warm up the OPTIM.BASE_LR over this number of epochs
__C.OPTIM.WARMUP_EPOCHS = 0


# ---------------------------------------------------------------------------- #
# Training options
# ---------------------------------------------------------------------------- #
__C.TRAIN = AttrDict()

# Dataset
__C.TRAIN.DATASET = ''
__C.TRAIN.SPLIT = 'train'

# Total mini-batch size
__C.TRAIN.BATCH_SIZE = 128

# Evaluate model on test data every eval period epochs
__C.TRAIN.EVAL_PERIOD = 1

# Save model checkpoint every checkpoint period epochs
__C.TRAIN.CHECKPOINT_PERIOD = 1

# Resume training from the latest checkpoint in the output directory
__C.TRAIN.AUTO_RESUME = True

# Checkpoint to start training from (if no automatic checkpoint saved)
__C.TRAIN.START_CHECKPOINT = ''


# ---------------------------------------------------------------------------- #
# Testing options
# ---------------------------------------------------------------------------- #
__C.TEST = AttrDict()

# Dataset
__C.TEST.DATASET = ''
__C.TEST.SPLIT = 'val'

# Total mini-batch size
__C.TEST.BATCH_SIZE = 200


# ---------------------------------------------------------------------------- #
# Common train/test data loader options
# ---------------------------------------------------------------------------- #
__C.DATA_LOADER = AttrDict()

# Number of data loader workers per training process
__C.DATA_LOADER.NUM_WORKERS = 4

# Load data to pinned host memory
__C.DATA_LOADER.PIN_MEMORY = True


# ---------------------------------------------------------------------------- #
# Memory options
# ---------------------------------------------------------------------------- #
__C.MEM = AttrDict()

# Perform ReLU inplace
__C.MEM.RELU_INPLACE = True


# ---------------------------------------------------------------------------- #
# CUDNN options
# ---------------------------------------------------------------------------- #
__C.CUDNN = AttrDict()

# Perform benchmarking to select the fastest CUDNN algorithms to use
# Note that this may increase the memory usage and will likely not result
# in overall speedups when variable size inputs are used (e.g. COCO training)
__C.CUDNN.BENCHMARK = False


# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #

# Number of GPUs to use (applies to both training and testing)
__C.NUM_GPUS = 1

# Output directory
__C.OUT_DIR = '/tmp'

# Note that non-determinism may still be present due to non-deterministic
# operator implementations in GPU operator libraries
__C.RNG_SEED = 1

# Log destination ('stdout' or 'file')
__C.LOG_DEST = 'stdout'

# Log period in iters
__C.LOG_PERIOD = 10

# Distributed backend
__C.DIST_BACKEND = 'nccl'

# Hostname and port for initializing multi-process groups
__C.HOST = 'localhost'
__C.PORT = 10001


# ---------------------------------------------------------------------------- #
# Functions for overriding the default config values
# ---------------------------------------------------------------------------- #

def _merge_dicts(dict_a, dict_b):
    for key, value in dict_a.items():
        if key not in dict_b:
            raise KeyError('Invalid key in config file: {}'.format(key))
        if type(value) is dict:
            dict_a[key] = value = AttrDict(value)
        if isinstance(value, str):
            try:
                value = literal_eval(value)
            except BaseException:
                pass
        # the types must match, too
        old_type = type(dict_b[key])
        if old_type is not type(value) and value is not None:
            raise ValueError(
                'Type mismatch ({} vs. {}) for config key: {}'.format(
                    type(dict_b[key]), type(value), key
                )
            )
        # recursively merge dicts
        if isinstance(value, AttrDict):
            _merge_dicts(dict_a[key], dict_b[key])
        else:
            dict_b[key] = value


def assert_and_infer_cfg():
    assert not __C.OPTIM.STEPS or __C.OPTIM.STEPS[0] == 0, \
        'The first lr step must start at 0'
    assert __C.TRAIN.SPLIT in ['train', 'val', 'test'], \
        'Train split \'{}\' not supported'.format(__C.TRAIN.SPLIT)
    assert __C.TRAIN.BATCH_SIZE % __C.NUM_GPUS == 0, \
        'Train mini-batch size should be a multiple of NUM_GPUS.'
    assert __C.TEST.SPLIT in ['train', 'val', 'test'], \
        'Test split \'{}\' not supported'.format(__C.TEST.SPLIT)
    assert __C.TEST.BATCH_SIZE % __C.NUM_GPUS == 0, \
        'Test mini-batch size should be a multiple of NUM_GPUS.'
    assert not __C.BN.USE_PRECISE_STATS or __C.NUM_GPUS == 1, \
        'Precise BN stats computation not verified for > 1 GPU'
    assert cfg.LOG_DEST in ['stdout', 'file'], \
        'Log destination \'{}\' not supported'.format(__C.LOG_DEST)


def merge_cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    with open(filename, 'r') as fopen:
        yaml_config = AttrDict(yaml.load(fopen))
    _merge_dicts(yaml_config, __C)


def merge_cfg_from_list(args_list):
    """Set config keys via list (e.g., from command line)."""
    assert len(args_list) % 2 == 0, 'Specify values or keys for args'
    for key, value in zip(args_list[0::2], args_list[1::2]):
        key_list = key.split('.')
        cfg = __C
        for subkey in key_list[:-1]:
            assert subkey in cfg, 'Config key {} not found'.format(subkey)
            cfg = cfg[subkey]
        subkey = key_list[-1]
        assert subkey in cfg, 'Config key {} not found'.format(subkey)
        try:
            # handle the case when v is a string literal
            val = literal_eval(value)
        except BaseException:
            val = value
        assert isinstance(val, type(cfg[subkey])) or cfg[subkey] is None, \
            'type {} does not match original type {}'.format(
                type(val), type(cfg[subkey])
            )
        cfg[subkey] = val

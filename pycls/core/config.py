#!/usr/bin/env python3

"""Configuration file."""

# TODO(ilijar): net naming (e.g. RESNET -> RES_NET)
# TODO(ilijar): configurable train and test resolution
# TODO(ilijar): don't include stem in stage config lists
# TODO(ilijar): move zero init entry from resnet to bn

from yacs.config import CfgNode as CN


# Global config object
_C = CN()

# Example usage:
#   from core.config import cfg
cfg = _C


# ---------------------------------------------------------------------------- #
# Model options
# ---------------------------------------------------------------------------- #
_C.MODEL = CN()

# Model type to use
_C.MODEL.TYPE = ''

# Number of weight layers
_C.MODEL.DEPTH = 0

# Number of classes
_C.MODEL.NUM_CLASSES = 10

# Loss function (see pycls/models/loss.py for options)
_C.MODEL.LOSS_FUN = 'cross_entropy'


# ---------------------------------------------------------------------------- #
# Batch norm options
# ---------------------------------------------------------------------------- #
_C.BN = CN()

# BN epsilon
_C.BN.EPS = 1e-5

# BN momentum (BN momentum in PyTorch = 1 - BN momentum in Caffe2)
_C.BN.MOM = 0.1

# Precise BN stats
_C.BN.USE_PRECISE_STATS = False
_C.BN.NUM_SAMPLES_PRECISE = 1024


# ---------------------------------------------------------------------------- #
# VGG options
# ---------------------------------------------------------------------------- #
_C.VGG = CN()

# Number of stages
_C.VGG.NUM_STAGES = 5

# Indices of stages whose first block uses stride 2 conv
_C.VGG.STRIDE2_INDS = []

# Indices of stages after which max pooling is performed
_C.VGG.MAX_POOL_INDS = [0, 1, 2, 3, 4]

# Depths multiplier (relative to the original VGG)
_C.VGG.DS_MULT = 1.0

# Widths multiplier (relative to the original VGG)
_C.VGG.WS_MULT = 1.0


# ---------------------------------------------------------------------------- #
# ResNet options
# ---------------------------------------------------------------------------- #
_C.RESNET = CN()

# Transformation function (see pycls/models/resnet.py for options)
_C.RESNET.TRANS_FUN = 'basic_transform'

# Number of groups to use (1 -> ResNet; > 1 -> ResNeXt)
_C.RESNET.NUM_GROUPS = 1

# Width of each group (64 -> ResNet; 4 -> ResNeXt)
_C.RESNET.WIDTH_PER_GROUP = 64

# Apply stride to 1x1 conv (True -> MSRA; False -> fb.torch)
_C.RESNET.STRIDE_1X1 = True

# Initialize the "gamma" scale parameters of the final BN operation of each
# residual block transformation function to zero
_C.RESNET.ZERO_INIT_FINAL_TRANSFORM_BN = False


# ---------------------------------------------------------------------------- #
# UniNet options
# ---------------------------------------------------------------------------- #
_C.UNINET = CN()

# Stem type (stage 0)
_C.UNINET.STEM_TYPE = 'plain_block'

# Block type (stages > 0)
_C.UNINET.BLOCK_TYPE = 'plain_block'

# Depth for each stage (number of blocks in the stage)
_C.UNINET.DEPTHS = []

# Width for each stage (width of each block in the stage)
_C.UNINET.WIDTHS = []

# Strides for each stage (applies to the first block of each stage)
_C.UNINET.STRIDES = []

# Bottleneck multipliers for each stage (applies to bottleneck block)
_C.UNINET.BOT_MULS = []

# Number of groups for each stage (applies to bottleneck block)
_C.UNINET.NUM_GS = []


# ---------------------------------------------------------------------------- #
# Optimizer options
# ---------------------------------------------------------------------------- #
_C.OPTIM = CN()

# Base learning rate
_C.OPTIM.BASE_LR = 0.1

# Learning rate policy select from {'cos', 'exp', 'steps'}
_C.OPTIM.LR_POLICY = 'cos'

# Exponential decay factor
_C.OPTIM.GAMMA = 0.1

# Steps for 'steps' policy (in epochs)
_C.OPTIM.STEPS = []

# Learning rate multiplier for 'steps' policy
_C.OPTIM.LR_MULT = 0.1

# Maximal number of epochs
_C.OPTIM.MAX_EPOCH = 200

# Momentum
_C.OPTIM.MOMENTUM = 0.9

# Momentum dampening
_C.OPTIM.DAMPENING = 0.0

# Nesterov momentum
_C.OPTIM.NESTEROV = True

# L2 regularization
_C.OPTIM.WEIGHT_DECAY = 5e-4

# Start the warm up from OPTIM.BASE_LR * OPTIM.WARMUP_FACTOR
_C.OPTIM.WARMUP_FACTOR = 0.1

# Gradually warm up the OPTIM.BASE_LR over this number of epochs
_C.OPTIM.WARMUP_EPOCHS = 0


# ---------------------------------------------------------------------------- #
# Training options
# ---------------------------------------------------------------------------- #
_C.TRAIN = CN()

# Dataset and split
_C.TRAIN.DATASET = ''
_C.TRAIN.SPLIT = 'train'

# Total mini-batch size
_C.TRAIN.BATCH_SIZE = 128

# Evaluate model on test data every eval period epochs
_C.TRAIN.EVAL_PERIOD = 1

# Save model checkpoint every checkpoint period epochs
_C.TRAIN.CHECKPOINT_PERIOD = 1

# Resume training from the latest checkpoint in the output directory
_C.TRAIN.AUTO_RESUME = True

# Checkpoint to start training from (if no automatic checkpoint saved)
_C.TRAIN.START_CHECKPOINT = ''


# ---------------------------------------------------------------------------- #
# Testing options
# ---------------------------------------------------------------------------- #
_C.TEST = CN()

# Dataset and split
_C.TEST.DATASET = ''
_C.TEST.SPLIT = 'val'

# Total mini-batch size
_C.TEST.BATCH_SIZE = 200


# ---------------------------------------------------------------------------- #
# Common train/test data loader options
# ---------------------------------------------------------------------------- #
_C.DATA_LOADER = CN()

# Number of data loader workers per training process
_C.DATA_LOADER.NUM_WORKERS = 4

# Load data to pinned host memory
_C.DATA_LOADER.PIN_MEMORY = True


# ---------------------------------------------------------------------------- #
# Memory options
# ---------------------------------------------------------------------------- #
_C.MEM = CN()

# Perform ReLU inplace
_C.MEM.RELU_INPLACE = True


# ---------------------------------------------------------------------------- #
# CUDNN options
# ---------------------------------------------------------------------------- #
_C.CUDNN = CN()

# Perform benchmarking to select the fastest CUDNN algorithms to use
# Note that this may increase the memory usage and will likely not result
# in overall speedups when variable size inputs are used (e.g. COCO training)
_C.CUDNN.BENCHMARK = False


# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #

# Number of GPUs to use (applies to both training and testing)
_C.NUM_GPUS = 1

# Output directory
_C.OUT_DIR = '/tmp'

# Note that non-determinism may still be present due to non-deterministic
# operator implementations in GPU operator libraries
_C.RNG_SEED = 1

# Log destination ('stdout' or 'file')
_C.LOG_DEST = 'stdout'

# Log period in iters
_C.LOG_PERIOD = 10

# Distributed backend
_C.DIST_BACKEND = 'nccl'

# Hostname and port for initializing multi-process groups
_C.HOST = 'localhost'
_C.PORT = 10001


def assert_cfg():
    """Checks config values invariants."""
    assert not _C.OPTIM.STEPS or _C.OPTIM.STEPS[0] == 0, \
        'The first lr step must start at 0'
    assert _C.TRAIN.SPLIT in ['train', 'val', 'test'], \
        'Train split \'{}\' not supported'.format(_C.TRAIN.SPLIT)
    assert _C.TRAIN.BATCH_SIZE % _C.NUM_GPUS == 0, \
        'Train mini-batch size should be a multiple of NUM_GPUS.'
    assert _C.TEST.SPLIT in ['train', 'val', 'test'], \
        'Test split \'{}\' not supported'.format(_C.TEST.SPLIT)
    assert _C.TEST.BATCH_SIZE % _C.NUM_GPUS == 0, \
        'Test mini-batch size should be a multiple of NUM_GPUS.'
    assert not _C.BN.USE_PRECISE_STATS or _C.NUM_GPUS == 1, \
        'Precise BN stats computation not verified for > 1 GPU'
    assert _C.LOG_DEST in ['stdout', 'file'], \
        'Log destination \'{}\' not supported'.format(_C.LOG_DEST)

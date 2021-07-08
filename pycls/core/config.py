#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Configuration file (powered by YACS)."""

import os

from pycls.core.io import pathmgr
from yacs.config import CfgNode


# Global config object (example usage: from core.config import cfg)
_C = CfgNode()
cfg = _C


# ---------------------------------- Model options ----------------------------------- #
_C.MODEL = CfgNode()

# Model type
_C.MODEL.TYPE = ""

# Number of weight layers
_C.MODEL.DEPTH = 0

# Number of classes
_C.MODEL.NUM_CLASSES = 10

# Loss function (see pycls/models/loss.py for options)
_C.MODEL.LOSS_FUN = "cross_entropy"

# Activation function (relu or silu/swish)
_C.MODEL.ACTIVATION_FUN = "relu"

# Perform activation inplace if implemented
_C.MODEL.ACTIVATION_INPLACE = True

# Model scaling parameters, see models/scaler.py (has no effect unless scaler is used)
_C.MODEL.SCALING_TYPE = ""
_C.MODEL.SCALING_FACTOR = 1.0


# ---------------------------------- ResNet options ---------------------------------- #
_C.RESNET = CfgNode()

# Transformation function (see pycls/models/resnet.py for options)
_C.RESNET.TRANS_FUN = "basic_transform"

# Number of groups to use (1 -> ResNet; > 1 -> ResNeXt)
_C.RESNET.NUM_GROUPS = 1

# Width of each group (64 -> ResNet; 4 -> ResNeXt)
_C.RESNET.WIDTH_PER_GROUP = 64

# Apply stride to 1x1 conv (True -> MSRA; False -> fb.torch)
_C.RESNET.STRIDE_1X1 = True


# ---------------------------------- AnyNet options ---------------------------------- #
_C.ANYNET = CfgNode()

# Stem type
_C.ANYNET.STEM_TYPE = "simple_stem_in"

# Stem width
_C.ANYNET.STEM_W = 32

# Block type
_C.ANYNET.BLOCK_TYPE = "res_bottleneck_block"

# Depth for each stage (number of blocks in the stage)
_C.ANYNET.DEPTHS = []

# Width for each stage (width of each block in the stage)
_C.ANYNET.WIDTHS = []

# Strides for each stage (applies to the first block of each stage)
_C.ANYNET.STRIDES = []

# Bottleneck multipliers for each stage (applies to bottleneck block)
_C.ANYNET.BOT_MULS = []

# Group widths for each stage (applies to bottleneck block)
_C.ANYNET.GROUP_WS = []

# Head width for first conv in head (if 0 conv is omitted, as is the default)
_C.ANYNET.HEAD_W = 0

# Whether SE is enabled for res_bottleneck_block
_C.ANYNET.SE_ON = False

# SE ratio
_C.ANYNET.SE_R = 0.25


# ---------------------------------- RegNet options ---------------------------------- #
_C.REGNET = CfgNode()

# Stem type
_C.REGNET.STEM_TYPE = "simple_stem_in"

# Stem width
_C.REGNET.STEM_W = 32

# Block type
_C.REGNET.BLOCK_TYPE = "res_bottleneck_block"

# Stride of each stage
_C.REGNET.STRIDE = 2

# Squeeze-and-Excitation (RegNetY)
_C.REGNET.SE_ON = False
_C.REGNET.SE_R = 0.25

# Depth
_C.REGNET.DEPTH = 10

# Initial width
_C.REGNET.W0 = 32

# Slope
_C.REGNET.WA = 5.0

# Quantization
_C.REGNET.WM = 2.5

# Group width
_C.REGNET.GROUP_W = 16

# Bottleneck multiplier (bm = 1 / b from the paper)
_C.REGNET.BOT_MUL = 1.0

# Head width for first conv in head (if 0 conv is omitted, as is the default)
_C.REGNET.HEAD_W = 0


# ------------------------------- EfficientNet options ------------------------------- #
_C.EN = CfgNode()

# Stem width
_C.EN.STEM_W = 32

# Depth for each stage (number of blocks in the stage)
_C.EN.DEPTHS = []

# Width for each stage (width of each block in the stage)
_C.EN.WIDTHS = []

# Expansion ratios for MBConv blocks in each stage
_C.EN.EXP_RATIOS = []

# Squeeze-and-Excitation (SE) ratio
_C.EN.SE_R = 0.25

# Strides for each stage (applies to the first block of each stage)
_C.EN.STRIDES = []

# Kernel sizes for each stage
_C.EN.KERNELS = []

# Head width
_C.EN.HEAD_W = 1280

# Drop connect ratio
_C.EN.DC_RATIO = 0.0

# Dropout ratio
_C.EN.DROPOUT_RATIO = 0.0


# ---------------------------- Vision Transformer options ---------------------------- #
_C.VIT = CfgNode()

# Patch Size (TRAIN.IM_SIZE must be divisible by PATCH_SIZE)
_C.VIT.PATCH_SIZE = 16

# Type of stem select from {'patchify', 'conv'}
_C.VIT.STEM_TYPE = "patchify"

# C-stem conv kernel sizes (https://arxiv.org/abs/2106.14881)
_C.VIT.C_STEM_KERNELS = []

# C-stem conv strides (the product of which must equal PATCH_SIZE)
_C.VIT.C_STEM_STRIDES = []

# C-stem conv output dims (last dim must equal HIDDEN_DIM)
_C.VIT.C_STEM_DIMS = []

# Number of layers in the encoder
_C.VIT.NUM_LAYERS = 12

# Number of self attention heads
_C.VIT.NUM_HEADS = 12

# Hidden dimension
_C.VIT.HIDDEN_DIM = 768

# Dimension of the MLP in the encoder
_C.VIT.MLP_DIM = 3072

# Type of classifier select from {'token', 'pooled'}
_C.VIT.CLASSIFIER_TYPE = "token"


# -------------------------------- Batch norm options -------------------------------- #
_C.BN = CfgNode()

# BN epsilon
_C.BN.EPS = 1e-5

# BN momentum (BN momentum in PyTorch = 1 - BN momentum in Caffe2)
_C.BN.MOM = 0.1

# Precise BN stats
_C.BN.USE_PRECISE_STATS = True
_C.BN.NUM_SAMPLES_PRECISE = 8192

# Initialize the gamma of the final BN of each block to zero
_C.BN.ZERO_INIT_FINAL_GAMMA = False

# Use a different weight decay for BN layers
_C.BN.USE_CUSTOM_WEIGHT_DECAY = False
_C.BN.CUSTOM_WEIGHT_DECAY = 0.0


# -------------------------------- Layer norm options -------------------------------- #
_C.LN = CfgNode()

# LN epsilon
_C.LN.EPS = 1e-5

# Use a different weight decay for LN layers
_C.LN.USE_CUSTOM_WEIGHT_DECAY = False
_C.LN.CUSTOM_WEIGHT_DECAY = 0.0


# -------------------------------- Optimizer options --------------------------------- #
_C.OPTIM = CfgNode()

# Type of optimizer select from {'sgd', 'adam', 'adamw'}
_C.OPTIM.OPTIMIZER = "sgd"

# Learning rate ranges from BASE_LR to MIN_LR*BASE_LR according to the LR_POLICY
_C.OPTIM.BASE_LR = 0.1
_C.OPTIM.MIN_LR = 0.0

# Learning rate policy select from {'cos', 'exp', 'lin', 'steps'}
_C.OPTIM.LR_POLICY = "cos"

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

# Betas (for Adam/AdamW optimizer)
_C.OPTIM.BETA1 = 0.9
_C.OPTIM.BETA2 = 0.999

# L2 regularization
_C.OPTIM.WEIGHT_DECAY = 5e-4

# Use a different weight decay for all biases (excluding those in BN/LN layers)
_C.OPTIM.BIAS_USE_CUSTOM_WEIGHT_DECAY = False
_C.OPTIM.BIAS_CUSTOM_WEIGHT_DECAY = 0.0

# Start the warm up from OPTIM.BASE_LR * OPTIM.WARMUP_FACTOR
_C.OPTIM.WARMUP_FACTOR = 0.1

# Gradually warm up the OPTIM.BASE_LR over this number of epochs
_C.OPTIM.WARMUP_EPOCHS = 0

# Exponential Moving Average (EMA) update value
_C.OPTIM.EMA_ALPHA = 1e-5

# Iteration frequency with which to update EMA weights
_C.OPTIM.EMA_UPDATE_PERIOD = 32


# --------------------------------- Training options --------------------------------- #
_C.TRAIN = CfgNode()

# Dataset and split
_C.TRAIN.DATASET = ""
_C.TRAIN.SPLIT = "train"

# Total mini-batch size
_C.TRAIN.BATCH_SIZE = 128

# Image size
_C.TRAIN.IM_SIZE = 224

# Resume training from the latest checkpoint in the output directory
_C.TRAIN.AUTO_RESUME = True

# Weights to start training from
_C.TRAIN.WEIGHTS = ""

# If True train using mixed precision
_C.TRAIN.MIXED_PRECISION = False

# Label smoothing value in 0 to 1 where (0 gives no smoothing)
_C.TRAIN.LABEL_SMOOTHING = 0.0

# Batch mixup regularization value in 0 to 1 (0 gives no mixup)
_C.TRAIN.MIXUP_ALPHA = 0.0

# Batch cutmix regularization value in 0 to 1 (0 gives no cutmix)
_C.TRAIN.CUTMIX_ALPHA = 0.0

# Standard deviation for AlexNet-style PCA jitter (0 gives no PCA jitter)
_C.TRAIN.PCA_STD = 0.1

# Data augmentation to use ("", "AutoAugment", "RandAugment_N2_M0.5", etc.)
_C.TRAIN.AUGMENT = ""


# --------------------------------- Testing options ---------------------------------- #
_C.TEST = CfgNode()

# Dataset and split
_C.TEST.DATASET = ""
_C.TEST.SPLIT = "val"

# Total mini-batch size
_C.TEST.BATCH_SIZE = 200

# Image size
_C.TEST.IM_SIZE = 256

# Weights to use for testing
_C.TEST.WEIGHTS = ""


# ------------------------------- Data loader options -------------------------------- #
_C.DATA_LOADER = CfgNode()

# Number of data loader workers per process
_C.DATA_LOADER.NUM_WORKERS = 8

# Load data to pinned host memory
_C.DATA_LOADER.PIN_MEMORY = True


# ---------------------------------- CUDNN options ----------------------------------- #
_C.CUDNN = CfgNode()

# Perform benchmarking to select fastest CUDNN algorithms (best for fixed input sizes)
_C.CUDNN.BENCHMARK = True


# ------------------------------- Precise time options ------------------------------- #
_C.PREC_TIME = CfgNode()

# Number of iterations to warm up the caches
_C.PREC_TIME.WARMUP_ITER = 3

# Number of iterations to compute avg time
_C.PREC_TIME.NUM_ITER = 30


# ----------------------------------- Misc options ----------------------------------- #
# Optional description of a config
_C.DESC = ""

# If True output additional info to log
_C.VERBOSE = True

# Number of GPUs to use (applies to both training and testing)
_C.NUM_GPUS = 1

# Output directory
_C.OUT_DIR = "/tmp"

# Config destination (in OUT_DIR)
_C.CFG_DEST = "config.yaml"

# Note that non-determinism is still be present due to non-deterministic GPU ops
_C.RNG_SEED = 1

# Log destination ('stdout' or 'file')
_C.LOG_DEST = "stdout"

# Log period in iters
_C.LOG_PERIOD = 10

# Distributed backend
_C.DIST_BACKEND = "nccl"

# Hostname and port range for multi-process groups (actual port selected randomly)
_C.HOST = "localhost"
_C.PORT_RANGE = [10000, 65000]

# Models weights referred to by URL are downloaded to this local cache
_C.DOWNLOAD_CACHE = "/tmp/pycls-download-cache"


# ---------------------------------- Default config ---------------------------------- #
_CFG_DEFAULT = _C.clone()
_CFG_DEFAULT.freeze()


# --------------------------------- Deprecated keys ---------------------------------- #
_C.register_deprecated_key("MEM")
_C.register_deprecated_key("MEM.RELU_INPLACE")
_C.register_deprecated_key("OPTIM.GAMMA")
_C.register_deprecated_key("PREC_TIME.BATCH_SIZE")
_C.register_deprecated_key("PREC_TIME.ENABLED")
_C.register_deprecated_key("PORT")
_C.register_deprecated_key("TRAIN.EVAL_PERIOD")
_C.register_deprecated_key("TRAIN.CHECKPOINT_PERIOD")


def assert_cfg():
    """Checks config values invariants."""
    err_str = "The first lr step must start at 0"
    assert not _C.OPTIM.STEPS or _C.OPTIM.STEPS[0] == 0, err_str
    data_splits = ["train", "val", "test"]
    err_str = "Data split '{}' not supported"
    assert _C.TRAIN.SPLIT in data_splits, err_str.format(_C.TRAIN.SPLIT)
    assert _C.TEST.SPLIT in data_splits, err_str.format(_C.TEST.SPLIT)
    err_str = "Mini-batch size should be a multiple of NUM_GPUS."
    assert _C.TRAIN.BATCH_SIZE % _C.NUM_GPUS == 0, err_str
    assert _C.TEST.BATCH_SIZE % _C.NUM_GPUS == 0, err_str
    err_str = "Log destination '{}' not supported"
    assert _C.LOG_DEST in ["stdout", "file"], err_str.format(_C.LOG_DEST)


def dump_cfg():
    """Dumps the config to the output directory."""
    cfg_file = os.path.join(_C.OUT_DIR, _C.CFG_DEST)
    with pathmgr.open(cfg_file, "w") as f:
        _C.dump(stream=f)
    return cfg_file


def load_cfg(cfg_file):
    """Loads config from specified file."""
    with pathmgr.open(cfg_file, "r") as f:
        _C.merge_from_other_cfg(_C.load_cfg(f))


def reset_cfg():
    """Reset config to initial state."""
    _C.merge_from_other_cfg(_CFG_DEFAULT)

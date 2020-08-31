#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Model zoo."""

import os

import pycls.core.checkpoint as checkpoint
from pycls.core.io import cache_url
from pycls.models.anynet import AnyNet
from pycls.models.effnet import EffNet


# URL prefix for pretrained models
_URL_PREFIX = "https://dl.fbaipublicfiles.com/pycls/dds_baselines"
# Model weights download cache directory
_DOWNLOAD_CACHE = "/tmp/pycls-download-cache"


# ------------------------------------------------------------------------------------ #
# RegNetX
# ------------------------------------------------------------------------------------ #

# RegNetX -> URL
_REGNETX_URLS = {
    "200MF": "160905981/RegNetX-200MF_dds_8gpu.pyth",
    "400MF": "160905967/RegNetX-400MF_dds_8gpu.pyth",
    "600MF": "160906442/RegNetX-600MF_dds_8gpu.pyth",
    "800MF": "160906036/RegNetX-800MF_dds_8gpu.pyth",
    "1.6GF": "160990626/RegNetX-1.6GF_dds_8gpu.pyth",
    "3.2GF": "160906139/RegNetX-3.2GF_dds_8gpu.pyth",
    "4.0GF": "160906383/RegNetX-4.0GF_dds_8gpu.pyth",
    "6.4GF": "161116590/RegNetX-6.4GF_dds_8gpu.pyth",
    "8.0GF": "161107726/RegNetX-8.0GF_dds_8gpu.pyth",
    "12GF": "160906020/RegNetX-12GF_dds_8gpu.pyth",
    "16GF": "158460855/RegNetX-16GF_dds_8gpu.pyth",
    "32GF": "158188473/RegNetX-32GF_dds_8gpu.pyth",
}

# RegNetX -> cfg
_REGNETX_CFGS = {
    "200MF": {"ds": [1, 1, 4, 7], "ws": [24, 56, 152, 368], "g": 8},
    "400MF": {"ds": [1, 2, 7, 12], "ws": [32, 64, 160, 384], "g": 16},
    "600MF": {"ds": [1, 3, 5, 7], "ws": [48, 96, 240, 528], "g": 24},
    "800MF": {"ds": [1, 3, 7, 5], "ws": [64, 128, 288, 672], "g": 16},
    "1.6GF": {"ds": [2, 4, 10, 2], "ws": [72, 168, 408, 912], "g": 24},
    "3.2GF": {"ds": [2, 6, 15, 2], "ws": [96, 192, 432, 1008], "g": 48},
    "4.0GF": {"ds": [2, 5, 14, 2], "ws": [80, 240, 560, 1360], "g": 40},
    "6.4GF": {"ds": [2, 4, 10, 1], "ws": [168, 392, 784, 1624], "g": 56},
    "8.0GF": {"ds": [2, 5, 15, 1], "ws": [80, 240, 720, 1920], "g": 120},
    "12GF": {"ds": [2, 5, 11, 1], "ws": [224, 448, 896, 2240], "g": 112},
    "16GF": {"ds": [2, 6, 13, 1], "ws": [256, 512, 896, 2048], "g": 128},
    "32GF": {"ds": [2, 7, 13, 1], "ws": [336, 672, 1344, 2520], "g": 168},
}


def regnetx(name, pretrained=False, nc=1000):
    """Constructs a RegNetX model."""
    is_valid = name in _REGNETX_URLS.keys() and name in _REGNETX_CFGS.keys()
    assert is_valid, "RegNetX-{} not found in the model zoo.".format(name)
    # Construct the model
    cfg = _REGNETX_CFGS[name]
    kwargs = {
        "stem_type": "simple_stem_in",
        "stem_w": 32,
        "block_type": "res_bottleneck_block",
        "ss": [2, 2, 2, 2],
        "bms": [1.0, 1.0, 1.0, 1.0],
        "se_r": None,
        "nc": nc,
        "ds": cfg["ds"],
        "ws": cfg["ws"],
        "gws": [cfg["g"] for _ in range(4)],
    }
    model = AnyNet(**kwargs)
    # Download and load the weights
    if pretrained:
        url = os.path.join(_URL_PREFIX, _REGNETX_URLS[name])
        ws_path = cache_url(url, _DOWNLOAD_CACHE)
        checkpoint.load_checkpoint(ws_path, model)
    return model


# ------------------------------------------------------------------------------------ #
# RegNetY
# ------------------------------------------------------------------------------------ #

# RegNetY -> URL
_REGNETY_URLS = {
    "200MF": "176245422/RegNetY-200MF_dds_8gpu.pyth",
    "400MF": "160906449/RegNetY-400MF_dds_8gpu.pyth",
    "600MF": "160981443/RegNetY-600MF_dds_8gpu.pyth",
    "800MF": "160906567/RegNetY-800MF_dds_8gpu.pyth",
    "1.6GF": "160906681/RegNetY-1.6GF_dds_8gpu.pyth",
    "3.2GF": "160906834/RegNetY-3.2GF_dds_8gpu.pyth",
    "4.0GF": "160906838/RegNetY-4.0GF_dds_8gpu.pyth",
    "6.4GF": "160907112/RegNetY-6.4GF_dds_8gpu.pyth",
    "8.0GF": "161160905/RegNetY-8.0GF_dds_8gpu.pyth",
    "12GF": "160907100/RegNetY-12GF_dds_8gpu.pyth",
    "16GF": "161303400/RegNetY-16GF_dds_8gpu.pyth",
    "32GF": "161277763/RegNetY-32GF_dds_8gpu.pyth",
}

# RegNetY -> cfg
_REGNETY_CFGS = {
    "200MF": {"ds": [1, 1, 4, 7], "ws": [24, 56, 152, 368], "g": 8},
    "400MF": {"ds": [1, 3, 6, 6], "ws": [48, 104, 208, 440], "g": 8},
    "600MF": {"ds": [1, 3, 7, 4], "ws": [48, 112, 256, 608], "g": 16},
    "800MF": {"ds": [1, 3, 8, 2], "ws": [64, 128, 320, 768], "g": 16},
    "1.6GF": {"ds": [2, 6, 17, 2], "ws": [48, 120, 336, 888], "g": 24},
    "3.2GF": {"ds": [2, 5, 13, 1], "ws": [72, 216, 576, 1512], "g": 24},
    "4.0GF": {"ds": [2, 6, 12, 2], "ws": [128, 192, 512, 1088], "g": 64},
    "6.4GF": {"ds": [2, 7, 14, 2], "ws": [144, 288, 576, 1296], "g": 72},
    "8.0GF": {"ds": [2, 4, 10, 1], "ws": [168, 448, 896, 2016], "g": 56},
    "12GF": {"ds": [2, 5, 11, 1], "ws": [224, 448, 896, 2240], "g": 112},
    "16GF": {"ds": [2, 4, 11, 1], "ws": [224, 448, 1232, 3024], "g": 112},
    "32GF": {"ds": [2, 5, 12, 1], "ws": [232, 696, 1392, 3712], "g": 116},
}


def regnety(name, pretrained=False, nc=1000):
    """Constructs a RegNetY model."""
    is_valid = name in _REGNETY_URLS.keys() and name in _REGNETY_CFGS.keys()
    assert is_valid, "RegNetY-{} not found in the model zoo.".format(name)
    # Construct the model
    cfg = _REGNETY_CFGS[name]
    kwargs = {
        "stem_type": "simple_stem_in",
        "stem_w": 32,
        "block_type": "res_bottleneck_block",
        "ss": [2, 2, 2, 2],
        "bms": [1.0, 1.0, 1.0, 1.0],
        "se_r": 0.25,
        "nc": nc,
        "ds": cfg["ds"],
        "ws": cfg["ws"],
        "gws": [cfg["g"] for _ in range(4)],
    }
    model = AnyNet(**kwargs)
    # Download and load the weights
    if pretrained:
        url = os.path.join(_URL_PREFIX, _REGNETY_URLS[name])
        ws_path = cache_url(url, _DOWNLOAD_CACHE)
        checkpoint.load_checkpoint(ws_path, model)
    return model


# ------------------------------------------------------------------------------------ #
# ResNet
# ------------------------------------------------------------------------------------ #

# ResNet -> URL
_RESNET_URLS = {
    "50": "161235311/R-50-1x64d_dds_8gpu.pyth",
    "101": "161167170/R-101-1x64d_dds_8gpu.pyth",
    "152": "161167467/R-152-1x64d_dds_8gpu.pyth",
}

# ResNet -> cfg
_RESNET_CFGS = {
    "50": {"ds": [3, 4, 6, 3]},
    "101": {"ds": [3, 4, 23, 3]},
    "152": {"ds": [3, 8, 36, 3]},
}


def resnet(name, pretrained=False, nc=1000):
    """Constructs a ResNet model."""
    is_valid = name in _RESNET_URLS.keys() and name in _RESNET_CFGS.keys()
    assert is_valid, "ResNet-{} not found in the model zoo.".format(name)
    # Construct the model
    cfg = _RESNET_CFGS[name]
    kwargs = {
        "stem_type": "res_stem_in",
        "stem_w": 64,
        "block_type": "res_bottleneck_block",
        "ss": [1, 2, 2, 2],
        "bms": [0.25, 0.25, 0.25, 0.25],
        "se_r": None,
        "nc": nc,
        "ds": cfg["ds"],
        "ws": [256, 512, 1024, 2048],
        "gws": [64, 128, 256, 512],
    }
    model = AnyNet(**kwargs)
    # Download and load the weights
    if pretrained:
        url = os.path.join(_URL_PREFIX, _RESNET_URLS[name])
        ws_path = cache_url(url, _DOWNLOAD_CACHE)
        checkpoint.load_checkpoint(ws_path, model)
    return model


# ------------------------------------------------------------------------------------ #
# ResNeXt
# ------------------------------------------------------------------------------------ #

# ResNeXt -> URL
_RESNEXT_URLS = {
    "50": "161167411/X-50-32x4d_dds_8gpu.pyth",
    "101": "161167590/X-101-32x4d_dds_8gpu.pyth",
    "152": "162471172/X-152-32x4d_dds_8gpu.pyth",
}

# ResNeXt -> cfg
_RESNEXT_CFGS = {
    "50": {"ds": [3, 4, 6, 3]},
    "101": {"ds": [3, 4, 23, 3]},
    "152": {"ds": [3, 8, 36, 3]},
}


def resnext(name, pretrained=False, nc=1000):
    """Constructs a ResNeXt model."""
    is_valid = name in _RESNEXT_URLS.keys() and name in _RESNEXT_CFGS.keys()
    assert is_valid, "ResNet-{} not found in the model zoo.".format(name)
    # Construct the model
    cfg = _RESNEXT_CFGS[name]
    kwargs = {
        "stem_type": "res_stem_in",
        "stem_w": 64,
        "block_type": "res_bottleneck_block",
        "ss": [1, 2, 2, 2],
        "bms": [0.5, 0.5, 0.5, 0.5],
        "se_r": None,
        "nc": nc,
        "ds": cfg["ds"],
        "ws": [256, 512, 1024, 2048],
        "gws": [4, 8, 16, 32],
    }
    model = AnyNet(**kwargs)
    # Download and load the weights
    if pretrained:
        url = os.path.join(_URL_PREFIX, _RESNEXT_URLS[name])
        ws_path = cache_url(url, _DOWNLOAD_CACHE)
        checkpoint.load_checkpoint(ws_path, model)
    return model


# ------------------------------------------------------------------------------------ #
# EfficientNet
# ------------------------------------------------------------------------------------ #

# EN -> URL
_EN_URLS = {
    "B0": "161305613/EN-B0_dds_8gpu.pyth",
    "B1": "161304979/EN-B1_dds_8gpu.pyth",
    "B2": "161305015/EN-B2_dds_8gpu.pyth",
    "B3": "161305060/EN-B3_dds_8gpu.pyth",
    "B4": "161305098/EN-B4_dds_8gpu.pyth",
    "B5": "161305138/EN-B5_dds_8gpu.pyth",
}

# EN -> cfg
_EN_CFGS = {
    "B0": {
        "sw": 32,
        "ds": [1, 2, 2, 3, 3, 4, 1],
        "ws": [16, 24, 40, 80, 112, 192, 320],
        "hw": 1280,
    },
    "B1": {
        "sw": 32,
        "ds": [2, 3, 3, 4, 4, 5, 2],
        "ws": [16, 24, 40, 80, 112, 192, 320],
        "hw": 1280,
    },
    "B2": {
        "sw": 32,
        "ds": [2, 3, 3, 4, 4, 5, 2],
        "ws": [16, 24, 48, 88, 120, 208, 352],
        "hw": 1408,
    },
    "B3": {
        "sw": 40,
        "ds": [2, 3, 3, 5, 5, 6, 2],
        "ws": [24, 32, 48, 96, 136, 232, 384],
        "hw": 1536,
    },
    "B4": {
        "sw": 48,
        "ds": [2, 4, 4, 6, 6, 8, 2],
        "ws": [24, 32, 56, 112, 160, 272, 448],
        "hw": 1792,
    },
    "B5": {
        "sw": 48,
        "ds": [3, 5, 5, 7, 7, 9, 3],
        "ws": [24, 40, 64, 128, 176, 304, 512],
        "hw": 2048,
    },
}


def effnet(name, pretrained=False, nc=1000):
    """Constructs an EfficientNet model."""
    is_valid = name in _EN_URLS.keys() and name in _EN_CFGS.keys()
    assert is_valid, "EfficientNet-{} not found in the model zoo.".format(name)
    # Construct the model
    cfg = _EN_CFGS[name]
    kwargs = {
        "exp_rs": [1, 6, 6, 6, 6, 6, 6],
        "se_r": 0.25,
        "nc": nc,
        "ss": [1, 2, 2, 2, 1, 2, 1],
        "ks": [3, 3, 5, 3, 5, 5, 3],
        "stem_w": cfg["sw"],
        "ds": cfg["ds"],
        "ws": cfg["ws"],
        "head_w": cfg["hw"],
    }
    model = EffNet(**kwargs)
    # Download and load the weights
    if pretrained:
        url = os.path.join(_URL_PREFIX, _EN_URLS[name])
        ws_path = cache_url(url, _DOWNLOAD_CACHE)
        checkpoint.load_checkpoint(ws_path, model)
    return model

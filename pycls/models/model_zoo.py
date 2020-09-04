#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Model zoo."""

import os

import pycls.core.builders as builders
import pycls.core.checkpoint as cp
from pycls.core.config import cfg, reset_cfg
from pycls.core.io import cache_url


# URL prefix for pretrained models
_URL_WEIGHTS = "https://dl.fbaipublicfiles.com/pycls"

# URL prefix for model config files
_URL_CONFIGS = "https://raw.githubusercontent.com/facebookresearch/pycls/master/configs"

# Model weights download cache directory
_DOWNLOAD_CACHE = "/tmp/pycls-download-cache"

# Predefined model config files
_MODEL_ZOO_CONFIGS = {
    "RegNetX-200MF": "dds_baselines/regnetx/RegNetX-200MF_dds_8gpu.yaml",
    "RegNetX-400MF": "dds_baselines/regnetx/RegNetX-400MF_dds_8gpu.yaml",
    "RegNetX-600MF": "dds_baselines/regnetx/RegNetX-600MF_dds_8gpu.yaml",
    "RegNetX-800MF": "dds_baselines/regnetx/RegNetX-800MF_dds_8gpu.yaml",
    "RegNetX-1.6GF": "dds_baselines/regnetx/RegNetX-1.6GF_dds_8gpu.yaml",
    "RegNetX-3.2GF": "dds_baselines/regnetx/RegNetX-3.2GF_dds_8gpu.yaml",
    "RegNetX-4.0GF": "dds_baselines/regnetx/RegNetX-4.0GF_dds_8gpu.yaml",
    "RegNetX-6.4GF": "dds_baselines/regnetx/RegNetX-6.4GF_dds_8gpu.yaml",
    "RegNetX-8.0GF": "dds_baselines/regnetx/RegNetX-8.0GF_dds_8gpu.yaml",
    "RegNetX-12GF": "dds_baselines/regnetx/RegNetX-12GF_dds_8gpu.yaml",
    "RegNetX-16GF": "dds_baselines/regnetx/RegNetX-16GF_dds_8gpu.yaml",
    "RegNetX-32GF": "dds_baselines/regnetx/RegNetX-32GF_dds_8gpu.yaml",
    "RegNetY-200MF": "dds_baselines/regnety/RegNetY-200MF_dds_8gpu.yaml",
    "RegNetY-400MF": "dds_baselines/regnety/RegNetY-400MF_dds_8gpu.yaml",
    "RegNetY-600MF": "dds_baselines/regnety/RegNetY-600MF_dds_8gpu.yaml",
    "RegNetY-800MF": "dds_baselines/regnety/RegNetY-800MF_dds_8gpu.yaml",
    "RegNetY-1.6GF": "dds_baselines/regnety/RegNetY-1.6GF_dds_8gpu.yaml",
    "RegNetY-3.2GF": "dds_baselines/regnety/RegNetY-3.2GF_dds_8gpu.yaml",
    "RegNetY-4.0GF": "dds_baselines/regnety/RegNetY-4.0GF_dds_8gpu.yaml",
    "RegNetY-6.4GF": "dds_baselines/regnety/RegNetY-6.4GF_dds_8gpu.yaml",
    "RegNetY-8.0GF": "dds_baselines/regnety/RegNetY-8.0GF_dds_8gpu.yaml",
    "RegNetY-12GF": "dds_baselines/regnety/RegNetY-12GF_dds_8gpu.yaml",
    "RegNetY-16GF": "dds_baselines/regnety/RegNetY-16GF_dds_8gpu.yaml",
    "RegNetY-32GF": "dds_baselines/regnety/RegNetY-32GF_dds_8gpu.yaml",
    "ResNet-50": "dds_baselines/resnet/R-50-1x64d_dds_8gpu.yaml",
    "ResNet-101": "dds_baselines/resnet/R-101-1x64d_dds_8gpu.yaml",
    "ResNet-152": "dds_baselines/resnet/R-152-1x64d_dds_8gpu.yaml",
    "ResNeXt-50": "dds_baselines/resnext/X-50-32x4d_dds_8gpu.yaml",
    "ResNeXt-101": "dds_baselines/resnext/X-101-32x4d_dds_8gpu.yaml",
    "ResNeXt-152": "dds_baselines/resnext/X-152-32x4d_dds_8gpu.yaml",
    "EfficientNet-B0": "dds_baselines/effnet/EN-B0_dds_8gpu.yaml",
    "EfficientNet-B1": "dds_baselines/effnet/EN-B1_dds_8gpu.yaml",
    "EfficientNet-B2": "dds_baselines/effnet/EN-B2_dds_8gpu.yaml",
    "EfficientNet-B3": "dds_baselines/effnet/EN-B3_dds_8gpu.yaml",
    "EfficientNet-B4": "dds_baselines/effnet/EN-B4_dds_8gpu.yaml",
    "EfficientNet-B5": "dds_baselines/effnet/EN-B5_dds_8gpu.yaml",
}

# Predefined model weight files
_MODEL_ZOO_WEIGHTS = {
    "RegNetX-200MF": "dds_baselines/160905981/RegNetX-200MF_dds_8gpu.pyth",
    "RegNetX-400MF": "dds_baselines/160905967/RegNetX-400MF_dds_8gpu.pyth",
    "RegNetX-600MF": "dds_baselines/160906442/RegNetX-600MF_dds_8gpu.pyth",
    "RegNetX-800MF": "dds_baselines/160906036/RegNetX-800MF_dds_8gpu.pyth",
    "RegNetX-1.6GF": "dds_baselines/160990626/RegNetX-1.6GF_dds_8gpu.pyth",
    "RegNetX-3.2GF": "dds_baselines/160906139/RegNetX-3.2GF_dds_8gpu.pyth",
    "RegNetX-4.0GF": "dds_baselines/160906383/RegNetX-4.0GF_dds_8gpu.pyth",
    "RegNetX-6.4GF": "dds_baselines/161116590/RegNetX-6.4GF_dds_8gpu.pyth",
    "RegNetX-8.0GF": "dds_baselines/161107726/RegNetX-8.0GF_dds_8gpu.pyth",
    "RegNetX-12GF": "dds_baselines/160906020/RegNetX-12GF_dds_8gpu.pyth",
    "RegNetX-16GF": "dds_baselines/158460855/RegNetX-16GF_dds_8gpu.pyth",
    "RegNetX-32GF": "dds_baselines/158188473/RegNetX-32GF_dds_8gpu.pyth",
    "RegNetY-200MF": "dds_baselines/176245422/RegNetY-200MF_dds_8gpu.pyth",
    "RegNetY-400MF": "dds_baselines/160906449/RegNetY-400MF_dds_8gpu.pyth",
    "RegNetY-600MF": "dds_baselines/160981443/RegNetY-600MF_dds_8gpu.pyth",
    "RegNetY-800MF": "dds_baselines/160906567/RegNetY-800MF_dds_8gpu.pyth",
    "RegNetY-1.6GF": "dds_baselines/160906681/RegNetY-1.6GF_dds_8gpu.pyth",
    "RegNetY-3.2GF": "dds_baselines/160906834/RegNetY-3.2GF_dds_8gpu.pyth",
    "RegNetY-4.0GF": "dds_baselines/160906838/RegNetY-4.0GF_dds_8gpu.pyth",
    "RegNetY-6.4GF": "dds_baselines/160907112/RegNetY-6.4GF_dds_8gpu.pyth",
    "RegNetY-8.0GF": "dds_baselines/161160905/RegNetY-8.0GF_dds_8gpu.pyth",
    "RegNetY-12GF": "dds_baselines/160907100/RegNetY-12GF_dds_8gpu.pyth",
    "RegNetY-16GF": "dds_baselines/161303400/RegNetY-16GF_dds_8gpu.pyth",
    "RegNetY-32GF": "dds_baselines/161277763/RegNetY-32GF_dds_8gpu.pyth",
    "ResNet-50": "dds_baselines/161235311/R-50-1x64d_dds_8gpu.pyth",
    "ResNet-101": "dds_baselines/161167170/R-101-1x64d_dds_8gpu.pyth",
    "ResNet-152": "dds_baselines/161167467/R-152-1x64d_dds_8gpu.pyth",
    "ResNeXt-50": "dds_baselines/161167411/X-50-32x4d_dds_8gpu.pyth",
    "ResNeXt-101": "dds_baselines/161167590/X-101-32x4d_dds_8gpu.pyth",
    "ResNeXt-152": "dds_baselines/162471172/X-152-32x4d_dds_8gpu.pyth",
    "EfficientNet-B0": "dds_baselines/161305613/EN-B0_dds_8gpu.pyth",
    "EfficientNet-B1": "dds_baselines/161304979/EN-B1_dds_8gpu.pyth",
    "EfficientNet-B2": "dds_baselines/161305015/EN-B2_dds_8gpu.pyth",
    "EfficientNet-B3": "dds_baselines/161305060/EN-B3_dds_8gpu.pyth",
    "EfficientNet-B4": "dds_baselines/161305098/EN-B4_dds_8gpu.pyth",
    "EfficientNet-B5": "dds_baselines/161305138/EN-B5_dds_8gpu.pyth",
}


def get_model_list():
    """Get list of all valid models in model zoo."""
    return _MODEL_ZOO_WEIGHTS.keys()


def get_config_file(name):
    """Get file with model config (downloads if necessary)."""
    err_str = "Model {} not found in the model zoo.".format(name)
    assert name in _MODEL_ZOO_CONFIGS.keys(), err_str
    config_url = os.path.join(_URL_CONFIGS, _MODEL_ZOO_CONFIGS[name])
    return cache_url(config_url, _DOWNLOAD_CACHE, _URL_CONFIGS)


def get_weights_file(name):
    """Get file with model weights (downloads if necessary)."""
    err_str = "Model {} not found in the model zoo.".format(name)
    assert name in _MODEL_ZOO_WEIGHTS.keys(), err_str
    weights_url = os.path.join(_URL_WEIGHTS, _MODEL_ZOO_WEIGHTS[name])
    return cache_url(weights_url, _DOWNLOAD_CACHE, _URL_WEIGHTS)


def get_model_info(name):
    """Return model info (useful for debugging)."""
    config_url = _MODEL_ZOO_CONFIGS[name]
    weight_url = _MODEL_ZOO_WEIGHTS[name]
    model_id = weight_url.split("/")[1]
    config_url_full = os.path.join(_URL_CONFIGS, _MODEL_ZOO_CONFIGS[name])
    weight_url_full = os.path.join(_URL_WEIGHTS, _MODEL_ZOO_WEIGHTS[name])
    return config_url, weight_url, model_id, config_url_full, weight_url_full


def build_model(name, pretrained=False, cfg_list=()):
    """Constructs a predefined model (note: loads global config as well)."""
    # Load the config
    reset_cfg()
    config_file = get_config_file(name)
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(cfg_list)
    # Construct model
    model = builders.build_model()
    # Load pretrained weights
    if pretrained:
        weights_file = get_weights_file(name)
        cp.load_checkpoint(weights_file, model)
    return model


def regnetx(name, pretrained=False, cfg_list=()):
    """Constructs a RegNetX model (note: loads global config as well)."""
    name = name if "RegNetX-" in name else "RegNetX-" + name
    return build_model(name, pretrained, cfg_list)


def regnety(name, pretrained=False, cfg_list=()):
    """Constructs a RegNetY model (note: loads global config as well)."""
    name = name if "RegNetY-" in name else "RegNetY-" + name
    return build_model(name, pretrained, cfg_list)


def resnet(name, pretrained=False, cfg_list=()):
    """Constructs a ResNet model (note: loads global config as well)."""
    name = name if "ResNet-" in name else "ResNet-" + name
    return build_model(name, pretrained, cfg_list)


def resnext(name, pretrained=False, cfg_list=()):
    """Constructs a ResNeXt model (note: loads global config as well)."""
    name = name if "ResNeXt-" in name else "ResNeXt-" + name
    return build_model(name, pretrained, cfg_list)


def effnet(name, pretrained=False, cfg_list=()):
    """Constructs an EfficientNet model (note: loads global config as well)."""
    name = name if "EfficientNet-" in name else "EfficientNet-" + name
    return build_model(name, pretrained, cfg_list)

#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Model construction functions."""

import pycls.utils.logging as lu
import torch
from pycls.core.config import cfg
from pycls.models.anynet import AnyNet
from pycls.models.effnet import EffNet
from pycls.models.regnet import RegNet
from pycls.models.resnet import ResNet


logger = lu.get_logger(__name__)

# Supported models
_models = {"anynet": AnyNet, "effnet": EffNet, "resnet": ResNet, "regnet": RegNet}


def get_model():
    """Gets the model specified in the config."""
    model_type = cfg.MODEL.TYPE
    err_str = "Model type '{}' not supported"
    assert model_type in _models.keys(), err_str.format(model_type)
    return _models[cfg.MODEL.TYPE]


def build_model():
    """Builds the model."""
    # Construct the model
    model = get_model()()
    # Transfer the model to the current GPU device
    err_str = "Cannot use more GPU devices than available"
    assert cfg.NUM_GPUS <= torch.cuda.device_count(), err_str
    cur_device = torch.cuda.current_device()
    model = model.cuda(device=cur_device)
    # Use multi-process data parallel model in the multi-gpu setting
    if cfg.NUM_GPUS > 1:
        # Make model replica operate on the current device
        model = torch.nn.parallel.DistributedDataParallel(
            module=model, device_ids=[cur_device], output_device=cur_device
        )
    return model


def register_model(name, ctor):
    """Registers a model dynamically."""
    _models[name] = ctor

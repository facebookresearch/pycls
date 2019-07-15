#!/usr/bin/env python3

"""Model construction functions."""

# TODO(ilijar): file naming (e.g. resnet -> res_net)
# TODO(ilijar): consider renaming the models subdir
# TODO(ilijar): function to register models dynamically

import torch

from pycls.config import cfg
from pycls.models.resnet import ResNet
from pycls.models.uninet import UniNet

import pycls.utils.logging as lu

logger = lu.get_logger(__name__)

# Supported model types
_MODEL_TYPES = {
    'resnet': ResNet,
    'uninet': UniNet,
}


def build_model():
    """Builds the model."""
    assert cfg.MODEL.TYPE in _MODEL_TYPES.keys(), \
        'Model type \'{}\' not supported'.format(cfg.MODEL.TYPE)
    assert cfg.NUM_GPUS <= torch.cuda.device_count(), \
        'Cannot use more GPU devices than available'
    # Construct the model
    model = _MODEL_TYPES[cfg.MODEL.TYPE]()
    # Determine the GPU used by the current process
    cur_device = torch.cuda.current_device()
    # Transfer the model to the current GPU device
    model = model.cuda(device=cur_device)
    # Use multi-process data parallel model in the multi-gpu setting
    if cfg.NUM_GPUS > 1:
        # Make model replica operate on the current device
        model = torch.nn.parallel.DistributedDataParallel(
            module=model,
            device_ids=[cur_device],
            output_device=cur_device
        )
    return model

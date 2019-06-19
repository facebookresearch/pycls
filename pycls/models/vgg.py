#!/usr/bin/env python3

"""VGG models."""

# TODO(ilijar): Refactor weight init to reduce duplication
# TODO(ilijar): Not VGG (delete, rename, or update)

import math
import torch.nn as nn

from pycls.config import cfg

import pycls.models.modules as modules
import pycls.utils.logging as lu

logger = lu.get_logger(__name__)


# Stage configurations (supports VGG11 VGG13, VGG16 and VGG19)
_STAGES = {
    11: [
        [64],
        [128],
        [256, 256],
        [512, 512],
        [512, 512]
    ],
    13: [
        [64, 64],
        [128, 128],
        [256, 256],
        [512, 512],
        [512, 512]
    ],
    16: [
        [64, 64],
        [128, 128],
        [256, 256, 256],
        [512, 512, 512],
        [512, 512, 512]
    ],
    19: [
        [64, 64],
        [128, 128],
        [256, 256, 256, 256],
        [512, 512, 512, 512],
        [512, 512, 512, 512]
    ]
}


class VGG(nn.Module):
    """VGG model."""

    def __init__(self):
        assert 'cifar' in cfg.TRAIN.DATASET and 'cifar' in cfg.TEST.DATASET, \
            'Only VGG for cifar is supported at this time'
        assert cfg.MODEL.DEPTH in _STAGES.keys(), \
            'VGG{} not supported'.format(cfg.MODEL.DEPTH)
        super(VGG, self).__init__()
        self._construct()
        self._init_weights()

    def _construct(self):
        # Note that the depth is misused in case of VGG for cifar.
        # The actual depth is cfg.MODEL.DEPTH - 2 (FC layers)
        logger.info(
            'Constructing: VGG-{}, stgs {}, s2c {}, mp {}, ds {}, ws {}'.format(
                cfg.MODEL.DEPTH,
                cfg.VGG.NUM_STAGES,
                cfg.VGG.STRIDE2_INDS,
                cfg.VGG.MAX_POOL_INDS,
                cfg.VGG.DS_MULT,
                cfg.VGG.WS_MULT
            )
        )

        # Retrieve the original body structure
        stages = _STAGES[cfg.MODEL.DEPTH][:cfg.VGG.NUM_STAGES]

        # Adjust the widths and depths
        stages = [
            [int(stage[0] * cfg.VGG.WS_MULT)] * len(stage) for stage in stages
        ]
        stages = [
            [stage[0]] * int(len(stage) * cfg.VGG.DS_MULT) for stage in stages
        ]

        # Construct the body
        body_layers = []
        dim_in = 3

        for i, stage in enumerate(stages):
            # Construct the blocks for the stage
            for j, dim_out in enumerate(stage):
                # Determine the block stride
                stride = 2 if j == 0 and i in cfg.VGG.STRIDE2_INDS else 1
                # Basic block: Conv, BN, ReLU
                body_layers += [
                    nn.Conv2d(
                        dim_in, dim_out, kernel_size=3,
                        stride=stride, padding=1, bias=False
                    ),
                    nn.BatchNorm2d(
                        dim_out, eps=cfg.BN.EPS, momentum=cfg.BN.MOM
                    ),
                    nn.ReLU(inplace=cfg.MEM.RELU_INPLACE)
                ]
                dim_in = dim_out
            # Perform reduction by pooling
            if i in cfg.VGG.MAX_POOL_INDS:
                body_layers += [
                    nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
                ]
        self.body = nn.Sequential(*body_layers)

        # Compute the final resolution
        num_reds = len(cfg.VGG.MAX_POOL_INDS) + len(cfg.VGG.STRIDE2_INDS)
        assert num_reds <= 5, \
            'Cannot perform more than 5 reductions on cifar (2^5 = 32)'
        final_res = 2 ** (5 - num_reds)

        # Construct the head
        self.head = nn.Sequential(
            nn.AvgPool2d(kernel_size=final_res, stride=1),
            modules.Flatten(),
            nn.Linear(dim_in, cfg.MODEL.NUM_CLASSES, bias=True)
        )

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Note that there is no bias due to BN
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(mean=0.0, std=math.sqrt(2.0 / fan_out))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(mean=0.0, std=0.01)
                m.bias.data.zero_()

    def forward(self, im):
        body_out = self.body(im)
        head_out = self.head(body_out)
        return head_out

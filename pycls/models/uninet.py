#!/usr/bin/env python3

"""Universal network."""

# TODO(ilijar): Update with changes from nms
# TODO(ilijar): Zero init final BN for each block
# TODO(ilijar): Refactor wieght init to reduce duplication

import math
import numpy as np

import torch.nn as nn

from pycls.core.config import cfg

import pycls.utils.logging as logging

logger = logging.get_logger(__name__)


def get_block_fun(block_type):
    """Retrieves the block function by name."""
    block_funs = {
        'plain_block': PlainBlock,
        'res_basic_block': ResBasicBlock,
    }
    assert block_type in block_funs.keys(), \
        'Block type \'{}\' not supported'.format(block_type)
    return block_funs[block_type]


def init_weights(model):
    """Performs ResNet style weight initialization."""
    for m in model.modules():
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


class UniHead(nn.Module):
    """UniNet head."""

    def __init__(self, dim_in, num_classes):
        super(UniHead, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(dim_in, num_classes, bias=True)

    def forward(self, x):
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class PlainBlock(nn.Module):
    """Plain block: 3x3 conv, BN, Relu"""

    def __init__(self, dim_in, dim_out, stride):
        super(PlainBlock, self).__init__()
        self._construct(dim_in, dim_out, stride)

    def _construct(self, dim_in, dim_out, stride):
        # 3x3, BN, ReLU
        self.conv = nn.Conv2d(
            dim_in, dim_out, kernel_size=3,
            stride=stride, padding=1, bias=False
        )
        self.bn = nn.BatchNorm2d(
            dim_out, eps=cfg.BN.EPSILON, momentum=cfg.BN.MOMENTUM
        )
        self.relu = nn.ReLU(cfg.MEM.RELU_INPLACE)

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x


class BasicTransform(nn.Module):
    """Basic transformation: [3x3 conv, BN, Relu] x2"""

    def __init__(self, dim_in, dim_out, stride):
        super(BasicTransform, self).__init__()
        self._construct(dim_in, dim_out, stride)

    def _construct(self, dim_in, dim_out, stride):
        # 3x3, BN, ReLU
        self.a = nn.Conv2d(
            dim_in, dim_out, kernel_size=3,
            stride=stride, padding=1, bias=False
        )
        self.a_bn = nn.BatchNorm2d(
            dim_out, eps=cfg.BN.EPSILON, momentum=cfg.BN.MOMENTUM
        )
        self.a_relu = nn.ReLU(inplace=cfg.MEM.RELU_INPLACE)

        # 3x3, BN
        self.b = nn.Conv2d(
            dim_out, dim_out, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.b_bn = nn.BatchNorm2d(
            dim_out, eps=cfg.BN.EPSILON, momentum=cfg.BN.MOMENTUM
        )

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x


class ResBasicBlock(nn.Module):
    """Residual basic block: x + F(x), F = basic transform"""

    def __init__(self, dim_in, dim_out, stride):
        super(ResBasicBlock, self).__init__()
        self._construct(dim_in, dim_out, stride)

    def _add_skip_proj(self, dim_in, dim_out, stride):
        self.proj = nn.Conv2d(
            dim_in, dim_out, kernel_size=1,
            stride=stride, padding=0, bias=False
        )
        self.bn = nn.BatchNorm2d(
            dim_out, eps=cfg.BN.EPSILON, momentum=cfg.BN.MOMENTUM
        )

    def _construct(self, dim_in, dim_out, stride):
        # Use skip connection with projection if dim or res change
        self.proj_block = (dim_in != dim_out) or (stride != 1)
        if self.proj_block:
            self._add_skip_proj(dim_in, dim_out, stride)
        self.f = BasicTransform(dim_in, dim_out, stride)
        self.relu = nn.ReLU(cfg.MEM.RELU_INPLACE)

    def forward(self, x):
        if self.proj_block:
            x = self.bn(self.proj(x)) + self.f(x)
        else:
            x = x + self.f(x)
        x = self.relu(x)
        return x


class UniStage(nn.Module):
    """UniNet stage."""

    def __init__(self, dim_in, dim_out, stride, num_blocks, block_fun):
        super(UniStage, self).__init__()
        self._construct(dim_in, dim_out, stride, num_blocks, block_fun)

    def _construct(self, dim_in, dim_out, stride, num_blocks, block_fun):
        for i in range(num_blocks):
            # Stride and dim_in apply to the first block of the stage
            b_stride = stride if i == 0 else 1
            b_dim_in = dim_in if i == 0 else dim_out
            # Construct the block
            self.add_module(
                'b{}'.format(i + 1),
                block_fun(b_dim_in, dim_out, b_stride)
            )

    def forward(self, x):
        for block in self.children():
            x = block(x)
        return x


class UniNet(nn.Module):
    """UniNet model."""

    def __init__(self):
        assert 'cifar' in cfg.TRAIN.DATASET and 'cifar' in cfg.TEST.DATASET, \
            'Only UniNet for cifar is supported at this time'
        assert len(cfg.UNINET.DEPTHS) == len(cfg.UNINET.WIDTHS), \
            'Depths and widths must be specified for each stage'
        assert len(cfg.UNINET.DEPTHS) == len(cfg.UNINET.STRIDES), \
            'Depths and strides must be specified for each stage'
        super(UniNet, self).__init__()
        self._construct(
            block_type=cfg.UNINET.BLOCK_TYPE,
            ds=cfg.UNINET.DEPTHS,
            ws=cfg.UNINET.WIDTHS,
            strides=cfg.UNINET.STRIDES,
            num_classes=cfg.MODEL.NUM_CLASSES
        )
        init_weights(self)

    def _construct(self, block_type, ds, ws, strides, num_classes):
        stage_params = list(zip(ds, ws, strides))
        logger.info('Constructing: UniNet-{}'.format(stage_params))

        # Construct the backbone
        block_fun = get_block_fun(block_type)
        prev_w = 3

        for i, (d, w, stride) in enumerate(stage_params):
            block_fun_i = PlainBlock if i == 0 else block_fun
            self.add_module(
                's{}'.format(i + 1),
                UniStage(prev_w, w, stride, d, block_fun_i)
            )
            prev_w = w

        # Construct the head
        self.head = UniHead(dim_in=prev_w, num_classes=num_classes)

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x

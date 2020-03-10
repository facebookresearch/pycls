#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""AnyNet models."""

import torch.nn as nn

from pycls.core.config import cfg

import pycls.utils.logging as lu
import pycls.utils.net as nu

logger = lu.get_logger(__name__)


def get_stem_fun(stem_type):
    """Retrives the stem function by name."""
    stem_funs = {
        'res_stem_cifar': ResStemCifar,
        'res_stem_in': ResStemIN,
    }
    assert stem_type in stem_funs.keys(), \
        'Stem type \'{}\' not supported'.format(stem_type)
    return stem_funs[stem_type]


def get_block_fun(block_type):
    """Retrieves the block function by name."""
    block_funs = {
        'vanilla_block': VanillaBlock,
        'res_basic_block': ResBasicBlock,
        'res_bottleneck_block': ResBottleneckBlock,
    }
    assert block_type in block_funs.keys(), \
        'Block type \'{}\' not supported'.format(block_type)
    return block_funs[block_type]


class AnyHead(nn.Module):
    """AnyNet head."""

    def __init__(self, w_in, nc):
        super(AnyHead, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(w_in, nc, bias=True)

    def forward(self, x):
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class VanillaBlock(nn.Module):
    """Vanilla block: [3x3 conv, BN, Relu] x2"""

    def __init__(self, w_in, w_out, stride, bm=None, g=None):
        assert bm is None and g is None, \
            'Vanilla block does not support bm and g options'
        super(VanillaBlock, self).__init__()
        self._construct(w_in, w_out, stride)

    def _construct(self, w_in, w_out, stride):
        # 3x3, BN, ReLU
        self.a = nn.Conv2d(
            w_in, w_out, kernel_size=3,
            stride=stride, padding=1, bias=False
        )
        self.a_bn = nn.BatchNorm2d(w_out, eps=cfg.BN.EPS, momentum=cfg.BN.MOM)
        self.a_relu = nn.ReLU(inplace=cfg.MEM.RELU_INPLACE)
        # 3x3, BN, ReLU
        self.b = nn.Conv2d(
            w_out, w_out, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.b_bn = nn.BatchNorm2d(w_out, eps=cfg.BN.EPS, momentum=cfg.BN.MOM)
        self.b_relu = nn.ReLU(inplace=cfg.MEM.RELU_INPLACE)

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x


class BasicTransform(nn.Module):
    """Basic transformation: [3x3 conv, BN, Relu] x2"""

    def __init__(self, w_in, w_out, stride):
        super(BasicTransform, self).__init__()
        self._construct(w_in, w_out, stride)

    def _construct(self, w_in, w_out, stride):
        # 3x3, BN, ReLU
        self.a = nn.Conv2d(
            w_in, w_out, kernel_size=3,
            stride=stride, padding=1, bias=False
        )
        self.a_bn = nn.BatchNorm2d(w_out, eps=cfg.BN.EPS, momentum=cfg.BN.MOM)
        self.a_relu = nn.ReLU(inplace=cfg.MEM.RELU_INPLACE)
        # 3x3, BN
        self.b = nn.Conv2d(
            w_out, w_out, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.b_bn = nn.BatchNorm2d(w_out, eps=cfg.BN.EPS, momentum=cfg.BN.MOM)
        self.b_bn.final_bn = True

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x


class ResBasicBlock(nn.Module):
    """Residual basic block: x + F(x), F = basic transform"""

    def __init__(self, w_in, w_out, stride, bm=None, g=None):
        assert bm is None and g is None, \
            'Basic transform does not support bm and g options'
        super(ResBasicBlock, self).__init__()
        self._construct(w_in, w_out, stride)

    def _add_skip_proj(self, w_in, w_out, stride):
        self.proj = nn.Conv2d(
            w_in, w_out, kernel_size=1,
            stride=stride, padding=0, bias=False
        )
        self.bn = nn.BatchNorm2d(w_out, eps=cfg.BN.EPS, momentum=cfg.BN.MOM)

    def _construct(self, w_in, w_out, stride):
        # Use skip connection with projection if shape changes
        self.proj_block = (w_in != w_out) or (stride != 1)
        if self.proj_block:
            self._add_skip_proj(w_in, w_out, stride)
        self.f = BasicTransform(w_in, w_out, stride)
        self.relu = nn.ReLU(cfg.MEM.RELU_INPLACE)

    def forward(self, x):
        if self.proj_block:
            x = self.bn(self.proj(x)) + self.f(x)
        else:
            x = x + self.f(x)
        x = self.relu(x)
        return x


class BottleneckTransform(nn.Module):
    """Bottlenect transformation: 1x1, 3x3, 1x1"""

    def __init__(self, w_in, w_out, stride, bm, g):
        super(BottleneckTransform, self).__init__()
        self._construct(w_in, w_out, stride, bm, g)

    def _construct(self, w_in, w_out, stride, bm, g):
        # Compute the bottleneck width
        w_b = int(round(w_out * bm))
        # Compute the number of groups
        num_gs = w_b // g if cfg.ANYNET.GW_PARAM else g
        # 1x1, BN, ReLU
        self.a = nn.Conv2d(
            w_in, w_b, kernel_size=1,
            stride=1, padding=0, bias=False
        )
        self.a_bn = nn.BatchNorm2d(
            w_b, eps=cfg.BN.EPS, momentum=cfg.BN.MOM)
        self.a_relu = nn.ReLU(inplace=cfg.MEM.RELU_INPLACE)
        # 3x3, BN, ReLU
        self.b = nn.Conv2d(
            w_b, w_b, kernel_size=3,
            stride=stride, padding=1, groups=num_gs, bias=False
        )
        self.b_bn = nn.BatchNorm2d(
            w_b, eps=cfg.BN.EPS, momentum=cfg.BN.MOM
        )
        self.b_relu = nn.ReLU(inplace=cfg.MEM.RELU_INPLACE)
        # 1x1, BN
        self.c = nn.Conv2d(
            w_b, w_out, kernel_size=1,
            stride=1, padding=0, bias=False
        )
        self.c_bn = nn.BatchNorm2d(w_out, eps=cfg.BN.EPS, momentum=cfg.BN.MOM)
        self.c_bn.final_bn = True

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x


class ResBottleneckBlock(nn.Module):
    """Residual bottleneck block: x + F(x), F = bottleneck transform"""

    def __init__(self, w_in, w_out, stride, bm=1.0, g=1):
        super(ResBottleneckBlock, self).__init__()
        self._construct(w_in, w_out, stride, bm, g)

    def _add_skip_proj(self, w_in, w_out, stride):
        self.proj = nn.Conv2d(
            w_in, w_out, kernel_size=1,
            stride=stride, padding=0, bias=False
        )
        self.bn = nn.BatchNorm2d(w_out, eps=cfg.BN.EPS, momentum=cfg.BN.MOM)

    def _construct(self, w_in, w_out, stride, bm, g):
        # Use skip connection with projection if shape changes
        self.proj_block = (w_in != w_out) or (stride != 1)
        if self.proj_block:
            self._add_skip_proj(w_in, w_out, stride)
        self.f = BottleneckTransform(w_in, w_out, stride, bm, g)
        self.relu = nn.ReLU(cfg.MEM.RELU_INPLACE)

    def forward(self, x):
        if self.proj_block:
            x = self.bn(self.proj(x)) + self.f(x)
        else:
            x = x + self.f(x)
        x = self.relu(x)
        return x


class ResStemCifar(nn.Module):
    """ResNet stem for CIFAR."""

    def __init__(self, w_in, w_out):
        super(ResStemCifar, self).__init__()
        self._construct(w_in, w_out, stride)

    def _construct(self, w_in, w_out, stride):
        # 3x3, BN, ReLU
        self.conv = nn.Conv2d(
            w_in, w_out, kernel_size=3,
            stride=stride, padding=1, bias=False
        )
        self.bn = nn.BatchNorm2d(w_out, eps=cfg.BN.EPS, momentum=cfg.BN.MOM)
        self.relu = nn.ReLU(cfg.MEM.RELU_INPLACE)

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x


class ResStemIN(nn.Module):
    """ResNet stem for ImageNet."""

    def __init__(self, w_in, w_out):
        super(ResStemIN, self).__init__()
        self._construct(w_in, w_out)

    def _construct(self, w_in, w_out):
        # 7x7, BN, ReLU, maxpool
        self.conv = nn.Conv2d(
            w_in, w_out, kernel_size=7,
            stride=2, padding=3, bias=False
        )
        self.bn = nn.BatchNorm2d(w_out, eps=cfg.BN.EPS, momentum=cfg.BN.MOM)
        self.relu = nn.ReLU(cfg.MEM.RELU_INPLACE)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x


class AnyStage(nn.Module):
    """AnyNet stage (sequence of blocks w/ the same output shape)."""

    def __init__(self, w_in, w_out, stride, d, block_fun, bm, g):
        super(AnyStage, self).__init__()
        self._construct(w_in, w_out, stride, d, block_fun, bm, g)

    def _construct(self, w_in, w_out, stride, d, block_fun, bm, g):
        # Construct the blocks
        for i in range(d):
            # Stride and w_in apply to the first block of the stage
            b_stride = stride if i == 0 else 1
            b_w_in = w_in if i == 0 else w_out
            # Construct the block
            self.add_module(
                'b{}'.format(i + 1),
                block_fun(b_w_in, w_out, b_stride, bm, g)
            )

    def forward(self, x):
        for block in self.children():
            x = block(x)
        return x


class AnyNet(nn.Module):
    """AnyNet model."""

    def __init__(self):
        assert len(cfg.ANYNET.DEPTHS) == len(cfg.ANYNET.WIDTHS), \
            'Depths and widths must be specified for each stage'
        assert len(cfg.ANYNET.DEPTHS) == len(cfg.ANYNET.STRIDES), \
            'Depths and strides must be specified for each stage'
        super(AnyNet, self).__init__()
        self._construct(
            stem_type=cfg.ANYNET.STEM_TYPE,
            stem_w=cfg.ANYNET.STEM_W,
            block_type=cfg.ANYNET.BLOCK_TYPE,
            ds=cfg.ANYNET.DEPTHS,
            ws=cfg.ANYNET.WIDTHS,
            ss=cfg.ANYNET.STRIDES,
            bms=cfg.ANYNET.BOT_MULS,
            gs=cfg.ANYNET.GROUPS,
            nc=cfg.MODEL.NUM_CLASSES
        )
        self.apply(nu.init_weights)

    def _construct(self, stem_type, stem_w, block_type, ds, ws, ss, bms, gs, nc):
        # Generate dummy bot muls and gs for models that do not use them
        bms = bms if bms else [1.0 for _d in ds]
        gs = gs if gs else [1 for _d in ds]
        # Group params by stage
        stage_params = list(zip(ds, ws, ss, bms, gs))
        logger.info('Constructing: AnyNet-{}'.format(stage_params))
        # Construct the stem
        stem_fun = get_stem_fun(stem_type)
        self.stem = stem_fun(3, stem_w)
        # Construct the stages
        block_fun = get_block_fun(block_type)
        prev_w = stem_w
        for i, (d, w, s, bm, g) in enumerate(stage_params):
            self.add_module(
                's{}'.format(i + 1),
                AnyStage(prev_w, w, s, d, block_fun, bm, g)
            )
            prev_w = w
        # Construct the head
        self.head = AnyHead(w_in=prev_w, nc=nc)

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x

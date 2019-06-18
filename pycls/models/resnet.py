#!/usr/bin/env python3

"""ResNet models."""

# TODO(ilijar): rename base to stem (like in uninet)

import math

import torch.nn as nn

from pycls.core.config import cfg

import pycls.utils.logging as lu

logger = lu.get_logger(__name__)


# Stage depths for an ImageNet model {model depth -> (d2, d3, d4, d5)}
_IN_MODEL_STAGE_DS = {
    50: (3, 4, 6, 3),
    101: (3, 4, 23, 3),
    152: (3, 8, 36, 3),
}


def get_trans_fun(name):
    """Retrieves the transformation function by name."""
    trans_funs = {
        'basic_transform': BasicTransform,
        'bottleneck_transform': BottleneckTransform,
    }
    assert name in trans_funs.keys(), \
        'Transformation function \'{}\' not supported'.format(name)
    return trans_funs[name]


def init_weights(model):
    """Performs ResNet style weight initialization."""
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            # Note that there is no bias due to BN
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(mean=0.0, std=math.sqrt(2.0 / fan_out))
        elif isinstance(m, nn.BatchNorm2d):
            fill = (
                0.0 if hasattr(m, 'final_transform_bn')
                and m.final_transform_bn
                and cfg.RESNET.ZERO_INIT_FINAL_TRANSFORM_BN
                else 1.0
            )
            m.weight.data.fill_(fill)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(mean=0.0, std=0.01)
            m.bias.data.zero_()


class ResHead(nn.Module):
    """ResNet head."""

    def __init__(self, dim_in, num_classes):
        super(ResHead, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(dim_in, num_classes, bias=True)

    def forward(self, x):
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class BasicTransform(nn.Module):
    """Basic transformation: 3x3, 3x3"""

    def __init__(self, dim_in, dim_out, stride, dim_inner=None, num_gs=1):
        assert dim_inner is None and num_gs == 1, \
            'Basic transform does not support dim_inner and num_gs options'
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
        self.b_bn.final_transform_bn = True

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x


class BottleneckTransform(nn.Module):
    """Bottleneck transformation: 1x1, 3x3, 1x1"""

    def __init__(self, dim_in, dim_out, stride, dim_inner, num_gs):
        super(BottleneckTransform, self).__init__()
        self._construct(dim_in, dim_out, stride, dim_inner, num_gs)

    def _construct(self, dim_in, dim_out, stride, dim_inner, num_gs):
        # MSRA -> stride=2 is on 1x1; TH/C2 -> stride=2 is on 3x3
        (str1x1, str3x3) = (stride, 1) if cfg.RESNET.STRIDE_1X1 else (1, stride)

        # 1x1, BN, ReLU
        self.a = nn.Conv2d(
            dim_in, dim_inner, kernel_size=1,
            stride=str1x1, padding=0, bias=False
        )
        self.a_bn = nn.BatchNorm2d(
            dim_inner, eps=cfg.BN.EPSILON, momentum=cfg.BN.MOMENTUM
        )
        self.a_relu = nn.ReLU(inplace=cfg.MEM.RELU_INPLACE)

        # 3x3, BN, ReLU
        self.b = nn.Conv2d(
            dim_inner, dim_inner, kernel_size=3,
            stride=str3x3, padding=1, groups=num_gs, bias=False
        )
        self.b_bn = nn.BatchNorm2d(
            dim_inner, eps=cfg.BN.EPSILON, momentum=cfg.BN.MOMENTUM
        )
        self.b_relu = nn.ReLU(inplace=cfg.MEM.RELU_INPLACE)

        # 1x1, BN
        self.c = nn.Conv2d(
            dim_inner, dim_out, kernel_size=1,
            stride=1, padding=0, bias=False
        )
        self.c_bn = nn.BatchNorm2d(
            dim_out, eps=cfg.BN.EPSILON, momentum=cfg.BN.MOMENTUM
        )
        self.c_bn.final_transform_bn = True

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x


class ResBlock(nn.Module):
    """Residual block: x + F(x)"""

    def __init__(
        self, dim_in, dim_out, stride, trans_fun, dim_inner=None, num_gs=1
    ):
        super(ResBlock, self).__init__()
        self._construct(dim_in, dim_out, stride, trans_fun, dim_inner, num_gs)

    def _add_skip_proj(self, dim_in, dim_out, stride):
        self.proj = nn.Conv2d(
            dim_in, dim_out, kernel_size=1,
            stride=stride, padding=0, bias=False
        )
        self.bn = nn.BatchNorm2d(
            dim_out, eps=cfg.BN.EPSILON, momentum=cfg.BN.MOMENTUM
        )

    def _construct(self, dim_in, dim_out, stride, trans_fun, dim_inner, num_gs):
        # Use skip connection with projection if dim or res change
        self.proj_block = (dim_in != dim_out) or (stride != 1)
        if self.proj_block:
            self._add_skip_proj(dim_in, dim_out, stride)
        self.f = trans_fun(dim_in, dim_out, stride, dim_inner, num_gs)
        self.relu = nn.ReLU(cfg.MEM.RELU_INPLACE)

    def forward(self, x):
        if self.proj_block:
            x = self.bn(self.proj(x)) + self.f(x)
        else:
            x = x + self.f(x)
        x = self.relu(x)
        return x


class ResStage(nn.Module):
    """Stage of ResNet."""

    def __init__(
        self, dim_in, dim_out, stride, num_bs, dim_inner=None, num_gs=1
    ):
        super(ResStage, self).__init__()
        self._construct(dim_in, dim_out, stride, num_bs, dim_inner, num_gs)

    def _construct(self, dim_in, dim_out, stride, num_bs, dim_inner, num_gs):
        for i in range(num_bs):
            # Stride and dim_in apply to the first block of the stage
            b_stride = stride if i == 0 else 1
            b_dim_in = dim_in if i == 0 else dim_out
            # Retrieve the transformation function
            trans_fun = get_trans_fun(cfg.RESNET.TRANS_FUN)
            # Construct the block
            res_block = ResBlock(
                b_dim_in, dim_out, b_stride, trans_fun, dim_inner, num_gs
            )
            self.add_module('b{}'.format(i + 1), res_block)

    def forward(self, x):
        for block in self.children():
            x = block(x)
        return x


class ResBase(nn.Module):
    """Base of ResNet."""

    def __init__(self, dim_in, dim_out):
        assert cfg.TRAIN.DATASET == cfg.TEST.DATASET, \
            'Train and test dataset must be the same for now'
        super(ResBase, self).__init__()
        if cfg.TRAIN.DATASET == 'cifar10':
            self._construct_cifar(dim_in, dim_out)
        else:
            self._construct_imagenet(dim_in, dim_out)

    def _construct_cifar(self, dim_in, dim_out):
        self.conv = nn.Conv2d(
            dim_in, dim_out, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.bn = nn.BatchNorm2d(
            dim_out, eps=cfg.BN.EPSILON, momentum=cfg.BN.MOMENTUM
        )
        self.relu = nn.ReLU(cfg.MEM.RELU_INPLACE)

    def _construct_imagenet(self, dim_in, dim_out):
        self.conv = nn.Conv2d(
            dim_in, dim_out, kernel_size=7,
            stride=2, padding=3, bias=False
        )
        self.bn = nn.BatchNorm2d(
            dim_out, eps=cfg.BN.EPSILON, momentum=cfg.BN.MOMENTUM
        )
        self.relu = nn.ReLU(cfg.MEM.RELU_INPLACE)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x


class ResNet(nn.Module):
    """ResNet model."""

    def __init__(self):
        assert cfg.TRAIN.DATASET in ['cifar10', 'imagenet'], \
            'Training ResNet on {} is not supported'.format(cfg.TRAIN.DATASET)
        assert cfg.TEST.DATASET in ['cifar10', 'imagenet'], \
            'Testing ResNet on {} is not supported'.format(cfg.TEST.DATASET)
        assert cfg.TRAIN.DATASET == cfg.TEST.DATASET, \
            'Train and test dataset must be the same for now'
        super(ResNet, self).__init__()
        if cfg.TRAIN.DATASET == 'cifar10':
            self._construct_cifar()
        else:
            self._construct_imagenet()
        init_weights(self)

    def _construct_cifar(self):
        assert (cfg.MODEL.DEPTH - 2) % 6 == 0, \
            'Model depth should be of the format 6n + 2 for cifar'
        logger.info('Constructing: ResNet-{}, cifar10'.format(cfg.MODEL.DEPTH))

        # Each stage has the same number of blocks for cifar
        num_blocks = int((cfg.MODEL.DEPTH - 2) / 6)
        # Stage 1: (N, 3, 32, 32) -> (N, 16, 32, 32)
        self.s1 = ResBase(dim_in=3, dim_out=16)
        # Stage 2: (N, 16, 32, 32) -> (N, 16, 32, 32)
        self.s2 = ResStage(dim_in=16, dim_out=16, stride=1, num_bs=num_blocks)
        # Stage 3: (N, 16, 32, 32) -> (N, 32, 16, 16)
        self.s3 = ResStage(dim_in=16, dim_out=32, stride=2, num_bs=num_blocks)
        # Stage 4: (N, 32, 16, 16) -> (N, 64, 8, 8)
        self.s4 = ResStage(dim_in=32, dim_out=64, stride=2, num_bs=num_blocks)
        # Head: (N, 64, 8, 8) -> (N, num_classes)
        self.head = ResHead(dim_in=64, num_classes=cfg.MODEL.NUM_CLASSES)

    def _construct_imagenet(self):
        logger.info('Constructing: ResNe(X)t-{}-{}x{}, {}, imagenet'.format(
            cfg.MODEL.DEPTH,
            cfg.RESNET.NUM_GROUPS,
            cfg.RESNET.WIDTH_PER_GROUP,
            cfg.RESNET.TRANS_FUN
        ))

        # Retrieve the number of blocks per stage (excluding base)
        (d2, d3, d4, d5) = _IN_MODEL_STAGE_DS[cfg.MODEL.DEPTH]

        # Compute the initial inner block dim
        num_groups = cfg.RESNET.NUM_GROUPS
        width_per_group = cfg.RESNET.WIDTH_PER_GROUP
        dim_inner = num_groups * width_per_group

        # Stage 1: (N, 3, 224, 224) -> (N, 64, 56, 56)
        self.s1 = ResBase(dim_in=3, dim_out=64)
        # Stage 2: (N, 64, 56, 56) -> (N, 256, 56, 56)
        self.s2 = ResStage(
            dim_in=64, dim_out=256, stride=1, num_bs=d2,
            dim_inner=dim_inner, num_gs=num_groups
        )
        # Stage 3: (N, 256, 56, 56) -> (N, 512, 28, 28)
        self.s3 = ResStage(
            dim_in=256, dim_out=512, stride=2, num_bs=d3,
            dim_inner=dim_inner * 2, num_gs=num_groups
        )
        # Stage 4: (N, 512, 56, 56) -> (N, 1024, 14, 14)
        self.s4 = ResStage(
            dim_in=512, dim_out=1024, stride=2, num_bs=d4,
            dim_inner=dim_inner * 4, num_gs=num_groups
        )
        # Stage 5: (N, 1024, 14, 14) -> (N, 2048, 7, 7)
        self.s5 = ResStage(
            dim_in=1024, dim_out=2048, stride=2, num_bs=d5,
            dim_inner=dim_inner * 8, num_gs=num_groups
        )
        # Head: (N, 2048, 7, 7) -> (N, num_classes)
        self.head = ResHead(dim_in=2048, num_classes=cfg.MODEL.NUM_CLASSES)

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x

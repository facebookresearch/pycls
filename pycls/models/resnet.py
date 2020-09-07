#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""ResNe(X)t models."""

from pycls.core.config import cfg
from pycls.models.blocks import (
    activation,
    conv2d,
    conv2d_cx,
    gap2d,
    gap2d_cx,
    init_weights,
    linear,
    linear_cx,
    norm2d,
    norm2d_cx,
    pool2d,
    pool2d_cx,
)
from torch.nn import Module


# Stage depths for ImageNet models
_IN_STAGE_DS = {50: (3, 4, 6, 3), 101: (3, 4, 23, 3), 152: (3, 8, 36, 3)}


def get_trans_fun(name):
    """Retrieves the transformation function by name."""
    trans_funs = {
        "basic_transform": BasicTransform,
        "bottleneck_transform": BottleneckTransform,
    }
    err_str = "Transformation function '{}' not supported"
    assert name in trans_funs.keys(), err_str.format(name)
    return trans_funs[name]


class ResHead(Module):
    """ResNet head: AvgPool, 1x1."""

    def __init__(self, w_in, num_classes):
        super(ResHead, self).__init__()
        self.avg_pool = gap2d(w_in)
        self.fc = linear(w_in, num_classes, bias=True)

    def forward(self, x):
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    @staticmethod
    def complexity(cx, w_in, num_classes):
        cx = gap2d_cx(cx, w_in)
        cx = linear_cx(cx, w_in, num_classes, bias=True)
        return cx


class BasicTransform(Module):
    """Basic transformation: 3x3, BN, AF, 3x3, BN."""

    def __init__(self, w_in, w_out, stride, w_b=None, groups=1):
        err_str = "Basic transform does not support w_b and groups options"
        assert w_b is None and groups == 1, err_str
        super(BasicTransform, self).__init__()
        self.a = conv2d(w_in, w_out, 3, stride=stride)
        self.a_bn = norm2d(w_out)
        self.a_af = activation()
        self.b = conv2d(w_out, w_out, 3)
        self.b_bn = norm2d(w_out)
        self.b_bn.final_bn = True

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x

    @staticmethod
    def complexity(cx, w_in, w_out, stride, w_b=None, groups=1):
        err_str = "Basic transform does not support w_b and groups options"
        assert w_b is None and groups == 1, err_str
        cx = conv2d_cx(cx, w_in, w_out, 3, stride=stride)
        cx = norm2d_cx(cx, w_out)
        cx = conv2d_cx(cx, w_out, w_out, 3)
        cx = norm2d_cx(cx, w_out)
        return cx


class BottleneckTransform(Module):
    """Bottleneck transformation: 1x1, BN, AF, 3x3, BN, AF, 1x1, BN."""

    def __init__(self, w_in, w_out, stride, w_b, groups):
        super(BottleneckTransform, self).__init__()
        # MSRA -> stride=2 is on 1x1; TH/C2 -> stride=2 is on 3x3
        (s1, s3) = (stride, 1) if cfg.RESNET.STRIDE_1X1 else (1, stride)
        self.a = conv2d(w_in, w_b, 1, stride=s1)
        self.a_bn = norm2d(w_b)
        self.a_af = activation()
        self.b = conv2d(w_b, w_b, 3, stride=s3, groups=groups)
        self.b_bn = norm2d(w_b)
        self.b_af = activation()
        self.c = conv2d(w_b, w_out, 1)
        self.c_bn = norm2d(w_out)
        self.c_bn.final_bn = True

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x

    @staticmethod
    def complexity(cx, w_in, w_out, stride, w_b, groups):
        (s1, s3) = (stride, 1) if cfg.RESNET.STRIDE_1X1 else (1, stride)
        cx = conv2d_cx(cx, w_in, w_b, 1, stride=s1)
        cx = norm2d_cx(cx, w_b)
        cx = conv2d_cx(cx, w_b, w_b, 3, stride=s3, groups=groups)
        cx = norm2d_cx(cx, w_b)
        cx = conv2d_cx(cx, w_b, w_out, 1)
        cx = norm2d_cx(cx, w_out)
        return cx


class ResBlock(Module):
    """Residual block: x + f(x)."""

    def __init__(self, w_in, w_out, stride, trans_fun, w_b=None, groups=1):
        super(ResBlock, self).__init__()
        self.proj, self.bn = None, None
        if (w_in != w_out) or (stride != 1):
            self.proj = conv2d(w_in, w_out, 1, stride=stride)
            self.bn = norm2d(w_out)
        self.f = trans_fun(w_in, w_out, stride, w_b, groups)
        self.af = activation()

    def forward(self, x):
        x_p = self.bn(self.proj(x)) if self.proj else x
        return self.af(x_p + self.f(x))

    @staticmethod
    def complexity(cx, w_in, w_out, stride, trans_fun, w_b, groups):
        if (w_in != w_out) or (stride != 1):
            h, w = cx["h"], cx["w"]
            cx = conv2d_cx(cx, w_in, w_out, 1, stride=stride)
            cx = norm2d_cx(cx, w_out)
            cx["h"], cx["w"] = h, w
        cx = trans_fun.complexity(cx, w_in, w_out, stride, w_b, groups)
        return cx


class ResStage(Module):
    """Stage of ResNet."""

    def __init__(self, w_in, w_out, stride, d, w_b=None, groups=1):
        super(ResStage, self).__init__()
        for i in range(d):
            b_stride = stride if i == 0 else 1
            b_w_in = w_in if i == 0 else w_out
            trans_fun = get_trans_fun(cfg.RESNET.TRANS_FUN)
            res_block = ResBlock(b_w_in, w_out, b_stride, trans_fun, w_b, groups)
            self.add_module("b{}".format(i + 1), res_block)

    def forward(self, x):
        for block in self.children():
            x = block(x)
        return x

    @staticmethod
    def complexity(cx, w_in, w_out, stride, d, w_b=None, groups=1):
        for i in range(d):
            b_stride = stride if i == 0 else 1
            b_w_in = w_in if i == 0 else w_out
            trans_f = get_trans_fun(cfg.RESNET.TRANS_FUN)
            cx = ResBlock.complexity(cx, b_w_in, w_out, b_stride, trans_f, w_b, groups)
        return cx


class ResStemCifar(Module):
    """ResNet stem for CIFAR: 3x3, BN, AF."""

    def __init__(self, w_in, w_out):
        super(ResStemCifar, self).__init__()
        self.conv = conv2d(w_in, w_out, 3)
        self.bn = norm2d(w_out)
        self.af = activation()

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x

    @staticmethod
    def complexity(cx, w_in, w_out):
        cx = conv2d_cx(cx, w_in, w_out, 3)
        cx = norm2d_cx(cx, w_out)
        return cx


class ResStemIN(Module):
    """ResNet stem for ImageNet: 7x7, BN, AF, MaxPool."""

    def __init__(self, w_in, w_out):
        super(ResStemIN, self).__init__()
        self.conv = conv2d(w_in, w_out, 7, stride=2)
        self.bn = norm2d(w_out)
        self.af = activation()
        self.pool = pool2d(w_out, 3, stride=2)

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x

    @staticmethod
    def complexity(cx, w_in, w_out):
        cx = conv2d_cx(cx, w_in, w_out, 7, stride=2)
        cx = norm2d_cx(cx, w_out)
        cx = pool2d_cx(cx, w_out, 3, stride=2)
        return cx


class ResNet(Module):
    """ResNet model."""

    def __init__(self):
        datasets = ["cifar10", "imagenet"]
        err_str = "Dataset {} is not supported"
        assert cfg.TRAIN.DATASET in datasets, err_str.format(cfg.TRAIN.DATASET)
        assert cfg.TEST.DATASET in datasets, err_str.format(cfg.TEST.DATASET)
        super(ResNet, self).__init__()
        if "cifar" in cfg.TRAIN.DATASET:
            self._construct_cifar()
        else:
            self._construct_imagenet()
        self.apply(init_weights)

    def _construct_cifar(self):
        err_str = "Model depth should be of the format 6n + 2 for cifar"
        assert (cfg.MODEL.DEPTH - 2) % 6 == 0, err_str
        d = int((cfg.MODEL.DEPTH - 2) / 6)
        self.stem = ResStemCifar(3, 16)
        self.s1 = ResStage(16, 16, stride=1, d=d)
        self.s2 = ResStage(16, 32, stride=2, d=d)
        self.s3 = ResStage(32, 64, stride=2, d=d)
        self.head = ResHead(64, cfg.MODEL.NUM_CLASSES)

    def _construct_imagenet(self):
        g, gw = cfg.RESNET.NUM_GROUPS, cfg.RESNET.WIDTH_PER_GROUP
        (d1, d2, d3, d4) = _IN_STAGE_DS[cfg.MODEL.DEPTH]
        w_b = gw * g
        self.stem = ResStemIN(3, 64)
        self.s1 = ResStage(64, 256, stride=1, d=d1, w_b=w_b, groups=g)
        self.s2 = ResStage(256, 512, stride=2, d=d2, w_b=w_b * 2, groups=g)
        self.s3 = ResStage(512, 1024, stride=2, d=d3, w_b=w_b * 4, groups=g)
        self.s4 = ResStage(1024, 2048, stride=2, d=d4, w_b=w_b * 8, groups=g)
        self.head = ResHead(2048, cfg.MODEL.NUM_CLASSES)

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x

    @staticmethod
    def complexity(cx):
        """Computes model complexity. If you alter the model, make sure to update."""
        if "cifar" in cfg.TRAIN.DATASET:
            d = int((cfg.MODEL.DEPTH - 2) / 6)
            cx = ResStemCifar.complexity(cx, 3, 16)
            cx = ResStage.complexity(cx, 16, 16, stride=1, d=d)
            cx = ResStage.complexity(cx, 16, 32, stride=2, d=d)
            cx = ResStage.complexity(cx, 32, 64, stride=2, d=d)
            cx = ResHead.complexity(cx, 64, cfg.MODEL.NUM_CLASSES)
        else:
            g, gw = cfg.RESNET.NUM_GROUPS, cfg.RESNET.WIDTH_PER_GROUP
            (d1, d2, d3, d4) = _IN_STAGE_DS[cfg.MODEL.DEPTH]
            w_b = gw * g
            cx = ResStemIN.complexity(cx, 3, 64)
            cx = ResStage.complexity(cx, 64, 256, 1, d=d1, w_b=w_b, groups=g)
            cx = ResStage.complexity(cx, 256, 512, 2, d=d2, w_b=w_b * 2, groups=g)
            cx = ResStage.complexity(cx, 512, 1024, 2, d=d3, w_b=w_b * 4, groups=g)
            cx = ResStage.complexity(cx, 1024, 2048, 2, d=d4, w_b=w_b * 8, groups=g)
            cx = ResHead.complexity(cx, 2048, cfg.MODEL.NUM_CLASSES)
        return cx

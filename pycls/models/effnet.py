#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""EfficientNet models."""

import pycls.core.net as net
import torch
import torch.nn as nn
from pycls.core.config import cfg


class EffHead(nn.Module):
    """EfficientNet head: 1x1, BN, Swish, AvgPool, Dropout, FC."""

    def __init__(self, w_in, w_out, nc):
        super(EffHead, self).__init__()
        self.conv = nn.Conv2d(w_in, w_out, 1, stride=1, padding=0, bias=False)
        self.conv_bn = nn.BatchNorm2d(w_out, eps=cfg.BN.EPS, momentum=cfg.BN.MOM)
        self.conv_swish = Swish()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        if cfg.EN.DROPOUT_RATIO > 0.0:
            self.dropout = nn.Dropout(p=cfg.EN.DROPOUT_RATIO)
        self.fc = nn.Linear(w_out, nc, bias=True)

    def forward(self, x):
        x = self.conv_swish(self.conv_bn(self.conv(x)))
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x) if hasattr(self, "dropout") else x
        x = self.fc(x)
        return x

    @staticmethod
    def complexity(cx, w_in, w_out, nc):
        cx = net.complexity_conv2d(cx, w_in, w_out, 1, 1, 0)
        cx = net.complexity_batchnorm2d(cx, w_out)
        cx["h"], cx["w"] = 1, 1
        cx = net.complexity_conv2d(cx, w_out, nc, 1, 1, 0, bias=True)
        return cx


class Swish(nn.Module):
    """Swish activation function: x * sigmoid(x)."""

    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class SE(nn.Module):
    """Squeeze-and-Excitation (SE) block w/ Swish: AvgPool, FC, Swish, FC, Sigmoid."""

    def __init__(self, w_in, w_se):
        super(SE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.f_ex = nn.Sequential(
            nn.Conv2d(w_in, w_se, 1, bias=True),
            Swish(),
            nn.Conv2d(w_se, w_in, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.f_ex(self.avg_pool(x))

    @staticmethod
    def complexity(cx, w_in, w_se):
        h, w = cx["h"], cx["w"]
        cx["h"], cx["w"] = 1, 1
        cx = net.complexity_conv2d(cx, w_in, w_se, 1, 1, 0, bias=True)
        cx = net.complexity_conv2d(cx, w_se, w_in, 1, 1, 0, bias=True)
        cx["h"], cx["w"] = h, w
        return cx


class MBConv(nn.Module):
    """Mobile inverted bottleneck block w/ SE (MBConv)."""

    def __init__(self, w_in, exp_r, kernel, stride, se_r, w_out):
        # expansion, 3x3 dwise, BN, Swish, SE, 1x1, BN, skip_connection
        super(MBConv, self).__init__()
        self.exp = None
        w_exp = int(w_in * exp_r)
        if w_exp != w_in:
            self.exp = nn.Conv2d(w_in, w_exp, 1, stride=1, padding=0, bias=False)
            self.exp_bn = nn.BatchNorm2d(w_exp, eps=cfg.BN.EPS, momentum=cfg.BN.MOM)
            self.exp_swish = Swish()
        dwise_args = {"groups": w_exp, "padding": (kernel - 1) // 2, "bias": False}
        self.dwise = nn.Conv2d(w_exp, w_exp, kernel, stride=stride, **dwise_args)
        self.dwise_bn = nn.BatchNorm2d(w_exp, eps=cfg.BN.EPS, momentum=cfg.BN.MOM)
        self.dwise_swish = Swish()
        self.se = SE(w_exp, int(w_in * se_r))
        self.lin_proj = nn.Conv2d(w_exp, w_out, 1, stride=1, padding=0, bias=False)
        self.lin_proj_bn = nn.BatchNorm2d(w_out, eps=cfg.BN.EPS, momentum=cfg.BN.MOM)
        # Skip connection if in and out shapes are the same (MN-V2 style)
        self.has_skip = stride == 1 and w_in == w_out

    def forward(self, x):
        f_x = x
        if self.exp:
            f_x = self.exp_swish(self.exp_bn(self.exp(f_x)))
        f_x = self.dwise_swish(self.dwise_bn(self.dwise(f_x)))
        f_x = self.se(f_x)
        f_x = self.lin_proj_bn(self.lin_proj(f_x))
        if self.has_skip:
            if self.training and cfg.EN.DC_RATIO > 0.0:
                f_x = net.drop_connect(f_x, cfg.EN.DC_RATIO)
            f_x = x + f_x
        return f_x

    @staticmethod
    def complexity(cx, w_in, exp_r, kernel, stride, se_r, w_out):
        w_exp = int(w_in * exp_r)
        if w_exp != w_in:
            cx = net.complexity_conv2d(cx, w_in, w_exp, 1, 1, 0)
            cx = net.complexity_batchnorm2d(cx, w_exp)
        padding = (kernel - 1) // 2
        cx = net.complexity_conv2d(cx, w_exp, w_exp, kernel, stride, padding, w_exp)
        cx = net.complexity_batchnorm2d(cx, w_exp)
        cx = SE.complexity(cx, w_exp, int(w_in * se_r))
        cx = net.complexity_conv2d(cx, w_exp, w_out, 1, 1, 0)
        cx = net.complexity_batchnorm2d(cx, w_out)
        return cx


class EffStage(nn.Module):
    """EfficientNet stage."""

    def __init__(self, w_in, exp_r, kernel, stride, se_r, w_out, d):
        super(EffStage, self).__init__()
        for i in range(d):
            b_stride = stride if i == 0 else 1
            b_w_in = w_in if i == 0 else w_out
            name = "b{}".format(i + 1)
            self.add_module(name, MBConv(b_w_in, exp_r, kernel, b_stride, se_r, w_out))

    def forward(self, x):
        for block in self.children():
            x = block(x)
        return x

    @staticmethod
    def complexity(cx, w_in, exp_r, kernel, stride, se_r, w_out, d):
        for i in range(d):
            b_stride = stride if i == 0 else 1
            b_w_in = w_in if i == 0 else w_out
            cx = MBConv.complexity(cx, b_w_in, exp_r, kernel, b_stride, se_r, w_out)
        return cx


class StemIN(nn.Module):
    """EfficientNet stem for ImageNet: 3x3, BN, Swish."""

    def __init__(self, w_in, w_out):
        super(StemIN, self).__init__()
        self.conv = nn.Conv2d(w_in, w_out, 3, stride=2, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(w_out, eps=cfg.BN.EPS, momentum=cfg.BN.MOM)
        self.swish = Swish()

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x

    @staticmethod
    def complexity(cx, w_in, w_out):
        cx = net.complexity_conv2d(cx, w_in, w_out, 3, 2, 1)
        cx = net.complexity_batchnorm2d(cx, w_out)
        return cx


class EffNet(nn.Module):
    """EfficientNet model."""

    @staticmethod
    def get_args():
        return {
            "stem_w": cfg.EN.STEM_W,
            "ds": cfg.EN.DEPTHS,
            "ws": cfg.EN.WIDTHS,
            "exp_rs": cfg.EN.EXP_RATIOS,
            "se_r": cfg.EN.SE_R,
            "ss": cfg.EN.STRIDES,
            "ks": cfg.EN.KERNELS,
            "head_w": cfg.EN.HEAD_W,
            "nc": cfg.MODEL.NUM_CLASSES,
        }

    def __init__(self):
        err_str = "Dataset {} is not supported"
        assert cfg.TRAIN.DATASET in ["imagenet"], err_str.format(cfg.TRAIN.DATASET)
        assert cfg.TEST.DATASET in ["imagenet"], err_str.format(cfg.TEST.DATASET)
        super(EffNet, self).__init__()
        self._construct(**EffNet.get_args())
        self.apply(net.init_weights)

    def _construct(self, stem_w, ds, ws, exp_rs, se_r, ss, ks, head_w, nc):
        stage_params = list(zip(ds, ws, exp_rs, ss, ks))
        self.stem = StemIN(3, stem_w)
        prev_w = stem_w
        for i, (d, w, exp_r, stride, kernel) in enumerate(stage_params):
            name = "s{}".format(i + 1)
            self.add_module(name, EffStage(prev_w, exp_r, kernel, stride, se_r, w, d))
            prev_w = w
        self.head = EffHead(prev_w, head_w, nc)

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x

    @staticmethod
    def complexity(cx):
        """Computes model complexity. If you alter the model, make sure to update."""
        return EffNet._complexity(cx, **EffNet.get_args())

    @staticmethod
    def _complexity(cx, stem_w, ds, ws, exp_rs, se_r, ss, ks, head_w, nc):
        stage_params = list(zip(ds, ws, exp_rs, ss, ks))
        cx = StemIN.complexity(cx, 3, stem_w)
        prev_w = stem_w
        for d, w, exp_r, stride, kernel in stage_params:
            cx = EffStage.complexity(cx, prev_w, exp_r, kernel, stride, se_r, w, d)
            prev_w = w
        cx = EffHead.complexity(cx, prev_w, head_w, nc)
        return cx

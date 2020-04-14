#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""EfficientNet models."""

import pycls.utils.logging as logging
import pycls.utils.net as nu
import torch
import torch.nn as nn
from pycls.core.config import cfg


logger = logging.get_logger(__name__)


class EffHead(nn.Module):
    """EfficientNet head."""

    def __init__(self, w_in, w_out, nc):
        super(EffHead, self).__init__()
        self._construct(w_in, w_out, nc)

    def _construct(self, w_in, w_out, nc):
        # 1x1, BN, Swish
        self.conv = nn.Conv2d(
            w_in, w_out, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.conv_bn = nn.BatchNorm2d(w_out, eps=cfg.BN.EPS, momentum=cfg.BN.MOM)
        self.conv_swish = Swish()
        # AvgPool
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # Dropout
        if cfg.EN.DROPOUT_RATIO > 0.0:
            self.dropout = nn.Dropout(p=cfg.EN.DROPOUT_RATIO)
        # FC
        self.fc = nn.Linear(w_out, nc, bias=True)

    def forward(self, x):
        x = self.conv_swish(self.conv_bn(self.conv(x)))
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x) if hasattr(self, "dropout") else x
        x = self.fc(x)
        return x


class Swish(nn.Module):
    """Swish activation function: x * sigmoid(x)"""

    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class SE(nn.Module):
    """Squeeze-and-Excitation (SE) block w/ Swish."""

    def __init__(self, w_in, w_se):
        super(SE, self).__init__()
        self._construct(w_in, w_se)

    def _construct(self, w_in, w_se):
        # AvgPool
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # FC, Swish, FC, Sigmoid
        self.f_ex = nn.Sequential(
            nn.Conv2d(w_in, w_se, kernel_size=1, bias=True),
            Swish(),
            nn.Conv2d(w_se, w_in, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.f_ex(self.avg_pool(x))


class MBConv(nn.Module):
    """Mobile inverted bottleneck block w/ SE (MBConv)."""

    def __init__(self, w_in, exp_r, kernel, stride, se_r, w_out):
        super(MBConv, self).__init__()
        self._construct(w_in, exp_r, kernel, stride, se_r, w_out)

    def _construct(self, w_in, exp_r, kernel, stride, se_r, w_out):
        # Expansion ratio is wrt the input width
        self.exp = None
        w_exp = int(w_in * exp_r)
        # Include exp ops only if the exp ratio is different from 1
        if w_exp != w_in:
            # 1x1, BN, Swish
            self.exp = nn.Conv2d(
                w_in, w_exp, kernel_size=1, stride=1, padding=0, bias=False
            )
            self.exp_bn = nn.BatchNorm2d(w_exp, eps=cfg.BN.EPS, momentum=cfg.BN.MOM)
            self.exp_swish = Swish()
        # 3x3 dwise, BN, Swish
        self.dwise = nn.Conv2d(
            w_exp,
            w_exp,
            kernel_size=kernel,
            stride=stride,
            groups=w_exp,
            bias=False,
            # Hacky padding to preserve res  (supports only 3x3 and 5x5)
            padding=(1 if kernel == 3 else 2),
        )
        self.dwise_bn = nn.BatchNorm2d(w_exp, eps=cfg.BN.EPS, momentum=cfg.BN.MOM)
        self.dwise_swish = Swish()
        # Squeeze-and-Excitation (SE)
        w_se = int(w_in * se_r)
        self.se = SE(w_exp, w_se)
        # 1x1, BN
        self.lin_proj = nn.Conv2d(
            w_exp, w_out, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.lin_proj_bn = nn.BatchNorm2d(w_out, eps=cfg.BN.EPS, momentum=cfg.BN.MOM)
        # Skip connection if in and out shapes are the same (MN-V2 style)
        self.has_skip = (stride == 1) and (w_in == w_out)

    def forward(self, x):
        f_x = x
        # Expansion
        if self.exp:
            f_x = self.exp_swish(self.exp_bn(self.exp(f_x)))
        # Depthwise
        f_x = self.dwise_swish(self.dwise_bn(self.dwise(f_x)))
        # SE
        f_x = self.se(f_x)
        # Linear projection
        f_x = self.lin_proj_bn(self.lin_proj(f_x))
        # Skip connection
        if self.has_skip:
            # Drop connect
            if self.training and cfg.EN.DC_RATIO > 0.0:
                f_x = nu.drop_connect(f_x, cfg.EN.DC_RATIO)
            f_x = x + f_x
        return f_x


class EffStage(nn.Module):
    """EfficientNet stage."""

    def __init__(self, w_in, exp_r, kernel, stride, se_r, w_out, d):
        super(EffStage, self).__init__()
        self._construct(w_in, exp_r, kernel, stride, se_r, w_out, d)

    def _construct(self, w_in, exp_r, kernel, stride, se_r, w_out, d):
        # Construct the blocks
        for i in range(d):
            # Stride and input width apply to the first block of the stage
            b_stride = stride if i == 0 else 1
            b_w_in = w_in if i == 0 else w_out
            # Construct the block
            self.add_module(
                "b{}".format(i + 1),
                MBConv(b_w_in, exp_r, kernel, b_stride, se_r, w_out),
            )

    def forward(self, x):
        for block in self.children():
            x = block(x)
        return x


class StemIN(nn.Module):
    """EfficientNet stem for ImageNet."""

    def __init__(self, w_in, w_out):
        super(StemIN, self).__init__()
        self._construct(w_in, w_out)

    def _construct(self, w_in, w_out):
        # 3x3, BN, Swish
        self.conv = nn.Conv2d(
            w_in, w_out, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.bn = nn.BatchNorm2d(w_out, eps=cfg.BN.EPS, momentum=cfg.BN.MOM)
        self.swish = Swish()

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x


class EffNet(nn.Module):
    """EfficientNet model."""

    def __init__(self):
        assert cfg.TRAIN.DATASET in [
            "imagenet"
        ], "Training on {} is not supported".format(cfg.TRAIN.DATASET)
        assert cfg.TEST.DATASET in [
            "imagenet"
        ], "Testing on {} is not supported".format(cfg.TEST.DATASET)
        super(EffNet, self).__init__()
        self._construct(
            stem_w=cfg.EN.STEM_W,
            ds=cfg.EN.DEPTHS,
            ws=cfg.EN.WIDTHS,
            exp_rs=cfg.EN.EXP_RATIOS,
            se_r=cfg.EN.SE_R,
            ss=cfg.EN.STRIDES,
            ks=cfg.EN.KERNELS,
            head_w=cfg.EN.HEAD_W,
            nc=cfg.MODEL.NUM_CLASSES,
        )
        self.apply(nu.init_weights)

    def _construct(self, stem_w, ds, ws, exp_rs, se_r, ss, ks, head_w, nc):
        # Group params by stage
        stage_params = list(zip(ds, ws, exp_rs, ss, ks))
        logger.info("Constructing: EfficientNet-{}".format(stage_params))
        # Construct the stem
        self.stem = StemIN(3, stem_w)
        prev_w = stem_w
        # Construct the stages
        for i, (d, w, exp_r, stride, kernel) in enumerate(stage_params):
            self.add_module(
                "s{}".format(i + 1), EffStage(prev_w, exp_r, kernel, stride, se_r, w, d)
            )
            prev_w = w
        # Construct the head
        self.head = EffHead(prev_w, head_w, nc)

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x

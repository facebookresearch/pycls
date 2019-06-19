#!/usr/bin/env python3

"""Universal network."""

# TODO(ilijar): Option to zero init final BN for each block
# TODO(ilijar): Refactor weight init to reduce duplication
# TODO(ilijar): Shorten BN keys (epsilon -> eps, momentum -> mom)
# TODO(ilijar): Consider creating stems.py and blocks.py

import math
import torch.nn as nn

from pycls.core.config import cfg

import pycls.utils.logging as lu

logger = lu.get_logger(__name__)


def get_stem_fun(stem_type):
    """Retrives the stem function by name."""
    stem_funs = {
        'plain_block': PlainBlock,
        'res_stem_in': ResStemIN
    }
    assert stem_type in stem_funs.keys(), \
        'Stem type \'{}\' not supported'.format(stem_type)
    return stem_funs[stem_type]


def get_block_fun(block_type):
    """Retrieves the block function by name."""
    block_funs = {
        'plain_block': PlainBlock,
        'double_plain_block': DoublePlainBlock,
        'res_basic_block': ResBasicBlock,
        'res_bottleneck_block': ResBottleneckBlock,
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

    def __init__(self, dim_in, dim_out, stride, bot_mul=1.0, num_gs=1):
        assert bot_mul == 1.0 and num_gs == 1, \
            'Plain block does not support bot_mul and num_gs options'
        super(PlainBlock, self).__init__()
        self._construct(dim_in, dim_out, stride)

    def _construct(self, dim_in, dim_out, stride):
        # 3x3, BN, ReLU
        self.conv = nn.Conv2d(
            dim_in, dim_out, kernel_size=3,
            stride=stride, padding=1, bias=False
        )
        self.bn = nn.BatchNorm2d(dim_out, eps=cfg.BN.EPS, momentum=cfg.BN.MOM)
        self.relu = nn.ReLU(cfg.MEM.RELU_INPLACE)

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x


class DoublePlainBlock(nn.Module):
    """Double plain block: [3x3 conv, BN, Relu] x2"""

    def __init__(self, dim_in, dim_out, stride, bot_mul=1.0, num_gs=1):
        assert bot_mul == 1.0 and num_gs == 1, \
            'Dobule plain block does not support bot_mul and num_gs options'
        super(DoublePlainBlock, self).__init__()
        self._construct(dim_in, dim_out, stride)

    def _construct(self, dim_in, dim_out, stride):
        # 3x3, BN, ReLU
        self.a = nn.Conv2d(
            dim_in, dim_out, kernel_size=3,
            stride=stride, padding=1, bias=False
        )
        self.a_bn = nn.BatchNorm2d(dim_out, eps=cfg.BN.EPS, momentum=cfg.BN.MOM)
        self.a_relu = nn.ReLU(inplace=cfg.MEM.RELU_INPLACE)

        # 3x3, BN, ReLU
        self.b = nn.Conv2d(
            dim_out, dim_out, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.b_bn = nn.BatchNorm2d(dim_out, eps=cfg.BN.EPS, momentum=cfg.BN.MOM)
        self.b_relu = nn.ReLU(inplace=cfg.MEM.RELU_INPLACE)

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
        self.a_bn = nn.BatchNorm2d(dim_out, eps=cfg.BN.EPS, momentum=cfg.BN.MOM)
        self.a_relu = nn.ReLU(inplace=cfg.MEM.RELU_INPLACE)

        # 3x3, BN
        self.b = nn.Conv2d(
            dim_out, dim_out, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.b_bn = nn.BatchNorm2d(dim_out, eps=cfg.BN.EPS, momentum=cfg.BN.MOM)

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x


class ResBasicBlock(nn.Module):
    """Residual basic block: x + F(x), F = basic transform"""

    def __init__(self, dim_in, dim_out, stride, bot_mul=1.0, num_gs=1):
        assert bot_mul == 1.0 and num_gs == 1, \
            'Basic transform does not support bot_mul and num_gs options'
        super(ResBasicBlock, self).__init__()
        self._construct(dim_in, dim_out, stride)

    def _add_skip_proj(self, dim_in, dim_out, stride):
        self.proj = nn.Conv2d(
            dim_in, dim_out, kernel_size=1,
            stride=stride, padding=0, bias=False
        )
        self.bn = nn.BatchNorm2d(dim_out, eps=cfg.BN.EPS, momentum=cfg.BN.MOM)

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


class BottleneckTransform(nn.Module):
    """Bottlenect transformation: 1x1, 3x3, 1x1"""

    def __init__(self, dim_in, dim_out, stride, bot_mul, num_gs):
        super(BottleneckTransform, self).__init__()
        self._construct(dim_in, dim_out, stride, bot_mul, num_gs)

    def _construct(self, dim_in, dim_out, stride, bot_mul, num_gs):
        # Compute the bottleneck width
        dim_inner = int(round(dim_out * bot_mul))

        # 1x1, BN, ReLU
        self.a = nn.Conv2d(
            dim_in, dim_inner, kernel_size=1,
            stride=1, padding=0, bias=False
        )
        self.a_bn = nn.BatchNorm2d(
            dim_inner, eps=cfg.BN.EPS, momentum=cfg.BN.MOM)
        self.a_relu = nn.ReLU(inplace=cfg.MEM.RELU_INPLACE)

        # 3x3, BN, ReLU
        self.b = nn.Conv2d(
            dim_inner, dim_inner, kernel_size=3,
            stride=stride, padding=1, groups=num_gs, bias=False
        )
        self.b_bn = nn.BatchNorm2d(
            dim_inner, eps=cfg.BN.EPS, momentum=cfg.BN.MOM
        )
        self.b_relu = nn.ReLU(inplace=cfg.MEM.RELU_INPLACE)

        # 1x1, BN
        self.c = nn.Conv2d(
            dim_inner, dim_out, kernel_size=1,
            stride=1, padding=0, bias=False
        )
        self.c_bn = nn.BatchNorm2d(dim_out, eps=cfg.BN.EPS, momentum=cfg.BN.MOM)
        self.c_bn.final_transform_bn = True

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x


class ResBottleneckBlock(nn.Module):
    """Residual bottleneck block: x + F(x), F = bottleneck transform"""

    def __init__(
        self, dim_in, dim_out, stride, bot_mul=1.0, num_gs=1
    ):
        super(ResBottleneckBlock, self).__init__()
        self._construct(dim_in, dim_out, stride, bot_mul, num_gs)

    def _add_skip_proj(self, dim_in, dim_out, stride):
        self.proj = nn.Conv2d(
            dim_in, dim_out, kernel_size=1,
            stride=stride, padding=0, bias=False
        )
        self.bn = nn.BatchNorm2d(dim_out, eps=cfg.BN.EPS, momentum=cfg.BN.MOM)

    def _construct(self, dim_in, dim_out, stride, bot_mul, num_gs):
        # Use skip connection with projection if dim or res change
        self.proj_block = (dim_in != dim_out) or (stride != 1)
        if self.proj_block:
            self._add_skip_proj(dim_in, dim_out, stride)
        self.f = BottleneckTransform(dim_in, dim_out, stride, bot_mul, num_gs)
        self.relu = nn.ReLU(cfg.MEM.RELU_INPLACE)

    def forward(self, x):
        if self.proj_block:
            x = self.bn(self.proj(x)) + self.f(x)
        else:
            x = x + self.f(x)
        x = self.relu(x)
        return x


class ResStemIN(nn.Module):
    """ResNet stem for ImageNet."""

    def __init__(self, dim_in, dim_out, stride, bot_mul=1.0, num_gs=1):
        assert bot_mul == 1.0 and num_gs == 1, \
            'ResNet IN stem does not support bot_mul and num_gs options'
        assert stride == 4, \
            'Stride of {} not supported for ResNet IN stem'.format(stride)
        super(ResStemIN, self).__init__()
        self._construct(dim_in, dim_out)

    def _construct(self, dim_in, dim_out):
        self.conv = nn.Conv2d(
            dim_in, dim_out, kernel_size=7,
            stride=2, padding=3, bias=False
        )
        self.bn = nn.BatchNorm2d(dim_out, eps=cfg.BN.EPS, momentum=cfg.BN.MOM)
        self.relu = nn.ReLU(cfg.MEM.RELU_INPLACE)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x


class UniStage(nn.Module):
    """UniNet stage."""

    def __init__(
        self, dim_in, dim_out, stride, num_bs, block_fun, bot_mul=1.0, num_gs=1
    ):
        super(UniStage, self).__init__()
        self._construct(
            dim_in, dim_out, stride, num_bs, block_fun, bot_mul, num_gs
        )

    def _construct(
        self, dim_in, dim_out, stride, num_bs, block_fun, bot_mul, num_gs
    ):
        for i in range(num_bs):
            # Stride and dim_in apply to the first block of the stage
            b_stride = stride if i == 0 else 1
            b_dim_in = dim_in if i == 0 else dim_out
            # Construct the block
            self.add_module(
                'b{}'.format(i + 1),
                block_fun(b_dim_in, dim_out, b_stride, bot_mul, num_gs)
            )

    def forward(self, x):
        for block in self.children():
            x = block(x)
        return x


class UniNet(nn.Module):
    """UniNet model."""

    def __init__(self):
        assert cfg.TRAIN.DATASET in ['cifar10', 'imagenet'], \
            'Training UniNet on {} is not supported'.format(cfg.TRAIN.DATASET)
        assert cfg.TEST.DATASET in ['cifar10', 'imagenet'], \
            'Testing UniNet on {} is not supported'.format(cfg.TEST.DATASET)
        assert cfg.TRAIN.DATASET == cfg.TEST.DATASET, \
            'Train and test dataset must be the same for now'
        assert len(cfg.UNINET.DEPTHS) == len(cfg.UNINET.WIDTHS), \
            'Depths and widths must be specified for each stage'
        assert len(cfg.UNINET.DEPTHS) == len(cfg.UNINET.STRIDES), \
            'Depths and strides must be specified for each stage'
        super(UniNet, self).__init__()
        self._construct(
            stem_type=cfg.UNINET.STEM_TYPE,
            block_type=cfg.UNINET.BLOCK_TYPE,
            ds=cfg.UNINET.DEPTHS,
            ws=cfg.UNINET.WIDTHS,
            ss=cfg.UNINET.STRIDES,
            bot_muls=cfg.UNINET.BOT_MULS,
            num_gs=cfg.UNINET.NUM_GS,
            num_classes=cfg.MODEL.NUM_CLASSES
        )
        init_weights(self)

    def _construct(
        self, stem_type, block_type, ds, ws, ss, bot_muls, num_gs, num_classes
    ):
        # Generate dummy bot muls and num gs for models that do not use them
        bot_muls = bot_muls if bot_muls else [1.0 for _d in ds]
        num_gs = num_gs if num_gs else [1 for _d in ds]

        # Group params by stage
        stage_params = list(zip(ds, ws, ss, bot_muls, num_gs))
        logger.info('Constructing: UniNet-{}'.format(stage_params))

        # Construct the backbone
        stem_fun = get_stem_fun(stem_type)
        block_fun = get_block_fun(block_type)
        prev_w = 3

        for i, (d_i, w_i, s_i, bm_i, gs_i) in enumerate(stage_params):
            block_fun_i = stem_fun if i == 0 else block_fun
            self.add_module(
                's{}'.format(i + 1),
                UniStage(prev_w, w_i, s_i, d_i, block_fun_i, bm_i, gs_i)
            )
            prev_w = w_i

        # Construct the head
        self.head = UniHead(dim_in=prev_w, num_classes=num_classes)

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x

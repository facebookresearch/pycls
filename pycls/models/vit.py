#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Vision Transformer (ViT) implementation (with patchify and conv stem).

References:
    https://arxiv.org/abs/2010.11929 (An Image is Worth 16x16 Words)
    https://arxiv.org/abs/2106.14881 (Early Convolutions Help Transformers See Better)
    https://github.com/google-research/vision_transformer
"""

import math

import numpy as np
import torch
from pycls.core.config import cfg
from pycls.models.blocks import (
    MultiheadSelfAttention,
    activation,
    conv2d,
    conv2d_cx,
    layernorm,
    layernorm_cx,
    linear,
    linear_cx,
    norm2d,
    norm2d_cx,
    patchify2d,
    patchify2d_cx,
)
from torch.nn import Module, Parameter, init


class ViTHead(Module):
    """Transformer classifier, an fc layer."""

    def __init__(self, w_in, num_classes):
        super().__init__()
        self.head_fc = linear(w_in, num_classes, bias=True)

    def forward(self, x):
        return self.head_fc(x)

    @staticmethod
    def complexity(cx, w_in, num_classes):
        return linear_cx(cx, w_in, num_classes, bias=True)


class MLPBlock(Module):
    """Transformer MLP block, fc, gelu, fc."""

    def __init__(self, w_in, mlp_d):
        super().__init__()
        self.linear_1 = linear(w_in, mlp_d, bias=True)
        self.af = activation("gelu")
        self.linear_2 = linear(mlp_d, w_in, bias=True)

    def forward(self, x):
        return self.linear_2(self.af(self.linear_1(x)))

    @staticmethod
    def complexity(cx, w_in, mlp_d, seq_len):
        cx = linear_cx(cx, w_in, mlp_d, bias=True, num_locations=seq_len)
        cx = linear_cx(cx, mlp_d, w_in, bias=True, num_locations=seq_len)
        return cx


class ViTEncoderBlock(Module):
    """Transformer encoder block, following https://arxiv.org/abs/2010.11929."""

    def __init__(self, hidden_d, n_heads, mlp_d):
        super().__init__()
        self.ln_1 = layernorm(hidden_d)
        self.self_attention = MultiheadSelfAttention(hidden_d, n_heads)
        self.ln_2 = layernorm(hidden_d)
        self.mlp_block = MLPBlock(hidden_d, mlp_d)

    def forward(self, x):
        x = x + self.self_attention(self.ln_1(x))
        x = x + self.mlp_block(self.ln_2(x))
        return x

    @staticmethod
    def complexity(cx, hidden_d, n_heads, mlp_d, seq_len):
        cx = layernorm_cx(cx, hidden_d)
        cx = MultiheadSelfAttention.complexity(cx, hidden_d, n_heads, seq_len)
        cx = layernorm_cx(cx, hidden_d)
        cx = MLPBlock.complexity(cx, hidden_d, mlp_d, seq_len)
        return cx


class ViTEncoder(Module):
    """Transformer encoder (sequence of ViTEncoderBlocks)."""

    def __init__(self, n_layers, hidden_d, n_heads, mlp_d):
        super(ViTEncoder, self).__init__()
        for i in range(n_layers):
            self.add_module(f"block_{i}", ViTEncoderBlock(hidden_d, n_heads, mlp_d))
        self.ln = layernorm(hidden_d)

    def forward(self, x):
        for block in self.children():
            x = block(x)
        return x

    @staticmethod
    def complexity(cx, n_layers, hidden_d, n_heads, mlp_d, seq_len):
        for _ in range(n_layers):
            cx = ViTEncoderBlock.complexity(cx, hidden_d, n_heads, mlp_d, seq_len)
        cx = layernorm_cx(cx, hidden_d)
        return cx


class ViTStemPatchify(Module):
    """The patchify vision transformer stem as per https://arxiv.org/abs/2010.11929."""

    def __init__(self, w_in, w_out, k):
        super(ViTStemPatchify, self).__init__()
        self.patchify = patchify2d(w_in, w_out, k, bias=True)

    def forward(self, x):
        return self.patchify(x)

    @staticmethod
    def complexity(cx, w_in, w_out, k):
        return patchify2d_cx(cx, w_in, w_out, k, bias=True)


class ViTStemConv(Module):
    """The conv vision transformer stem as per https://arxiv.org/abs/2106.14881."""

    def __init__(self, w_in, ks, ws, ss):
        super(ViTStemConv, self).__init__()
        for i, (k, w_out, stride) in enumerate(zip(ks, ws, ss)):
            if i < len(ks) - 1:
                self.add_module(f"cstem{i}_conv", conv2d(w_in, w_out, 3, stride=stride))
                self.add_module(f"cstem{i}_bn", norm2d(w_out))
                self.add_module(f"cstem{i}_af", activation("relu"))
            else:
                m = conv2d(w_in, w_out, k, stride=stride, bias=True)
                self.add_module("cstem_last_conv", m)
            w_in = w_out

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x

    @staticmethod
    def complexity(cx, w_in, ks, ws, ss):
        for i, (k, w_out, stride) in enumerate(zip(ks, ws, ss)):
            if i < len(ks) - 1:
                cx = conv2d_cx(cx, w_in, w_out, 3, stride=stride)
                cx = norm2d_cx(cx, w_out)
            else:
                cx = conv2d_cx(cx, w_in, w_out, k, stride=stride, bias=True)
            w_in = w_out
        return cx


class ViT(Module):
    """Vision transformer as per https://arxiv.org/abs/2010.11929."""

    @staticmethod
    def get_params():
        return {
            "image_size": cfg.TRAIN.IM_SIZE,
            "patch_size": cfg.VIT.PATCH_SIZE,
            "stem_type": cfg.VIT.STEM_TYPE,
            "c_stem_kernels": cfg.VIT.C_STEM_KERNELS,
            "c_stem_strides": cfg.VIT.C_STEM_STRIDES,
            "c_stem_dims": cfg.VIT.C_STEM_DIMS,
            "n_layers": cfg.VIT.NUM_LAYERS,
            "n_heads": cfg.VIT.NUM_HEADS,
            "hidden_d": cfg.VIT.HIDDEN_DIM,
            "mlp_d": cfg.VIT.MLP_DIM,
            "cls_type": cfg.VIT.CLASSIFIER_TYPE,
            "num_classes": cfg.MODEL.NUM_CLASSES,
        }

    @staticmethod
    def check_params(params):
        p = params
        err_str = "Input shape indivisible by patch size"
        assert p["image_size"] % p["patch_size"] == 0, err_str
        assert p["stem_type"] in ["patchify", "conv"], "Unexpected stem type"
        assert p["cls_type"] in ["token", "pooled"], "Unexpected classifier mode"
        if p["stem_type"] == "conv":
            err_str = "Conv stem layers mismatch"
            assert len(p["c_stem_dims"]) == len(p["c_stem_strides"]), err_str
            assert len(p["c_stem_strides"]) == len(p["c_stem_kernels"]), err_str
            err_str = "Stem strides unequal to patch size"
            assert p["patch_size"] == np.prod(p["c_stem_strides"]), err_str
            err_str = "Stem output dim unequal to hidden dim"
            assert p["c_stem_dims"][-1] == p["hidden_d"], err_str

    def __init__(self, params=None):
        super(ViT, self).__init__()
        p = ViT.get_params() if not params else params
        ViT.check_params(p)
        if p["stem_type"] == "patchify":
            self.stem = ViTStemPatchify(3, p["hidden_d"], p["patch_size"])
        elif p["stem_type"] == "conv":
            ks, ws, ss = p["c_stem_kernels"], p["c_stem_dims"], p["c_stem_strides"]
            self.stem = ViTStemConv(3, ks, ws, ss)
        seq_len = (p["image_size"] // cfg.VIT.PATCH_SIZE) ** 2
        if p["cls_type"] == "token":
            self.class_token = Parameter(torch.zeros(1, 1, p["hidden_d"]))
            seq_len += 1
        else:
            self.class_token = None
        self.pos_embedding = Parameter(torch.zeros(1, seq_len, p["hidden_d"]))
        self.encoder = ViTEncoder(
            p["n_layers"], p["hidden_d"], p["n_heads"], p["mlp_d"]
        )
        self.head = ViTHead(p["hidden_d"], p["num_classes"])
        init_weights_vit(self)

    def forward(self, x):
        # (n, c, h, w) -> (n, hidden_d, n_h, n_w)
        x = self.stem(x)
        # (n, hidden_d, n_h, n_w) -> (n, hidden_d, (n_h * n_w))
        x = x.reshape(x.size(0), x.size(1), -1)
        # (n, hidden_d, (n_h * n_w)) -> (n, (n_h * n_w), hidden_d)
        x = x.permute(0, 2, 1)
        if self.class_token is not None:
            # Expand the class token to the full batch
            class_token = self.class_token.expand(x.size(0), -1, -1)
            x = torch.cat([class_token, x], dim=1)
        x = x + self.pos_embedding
        x = self.encoder(x)
        # `token` or `pooled` features for classification
        x = x[:, 0, :] if self.class_token is not None else x.mean(dim=1)
        return self.head(x)

    @staticmethod
    def complexity(cx, params=None):
        """Computes model complexity. If you alter the model, make sure to update."""
        p = ViT.get_params() if not params else params
        ViT.check_params(p)
        if p["stem_type"] == "patchify":
            cx = ViTStemPatchify.complexity(cx, 3, p["hidden_d"], p["patch_size"])
        elif p["stem_type"] == "conv":
            ks, ws, ss = p["c_stem_kernels"], p["c_stem_dims"], p["c_stem_strides"]
            cx = ViTStemConv.complexity(cx, 3, ks, ws, ss)
        seq_len = (p["image_size"] // cfg.VIT.PATCH_SIZE) ** 2
        if p["cls_type"] == "token":
            seq_len += 1
            cx["params"] += p["hidden_d"]
        # Params of position embeddings
        cx["params"] += seq_len * p["hidden_d"]
        cx = ViTEncoder.complexity(
            cx, p["n_layers"], p["hidden_d"], p["n_heads"], p["mlp_d"], seq_len
        )
        cx = ViTHead.complexity(cx, p["hidden_d"], p["num_classes"])
        return cx


def init_weights_vit(model):
    """Performs ViT weight init."""
    for k, m in model.named_modules():
        if isinstance(m, torch.nn.Conv2d):
            if "patchify" in k:
                # ViT patchify stem init
                fan_in = m.in_channels * m.kernel_size[0] * m.kernel_size[1]
                init.trunc_normal_(m.weight, std=math.sqrt(1.0 / fan_in))
                init.zeros_(m.bias)
            elif "cstem_last" in k:
                # The last 1x1 conv of the conv stem
                init.normal_(m.weight, mean=0.0, std=math.sqrt(2.0 / m.out_channels))
                init.zeros_(m.bias)
            elif "cstem" in k:
                # Use default pytorch init for other conv layers in the C-stem
                pass
            else:
                raise NotImplementedError
        if isinstance(m, torch.nn.Linear):
            if "self_attention" in k:
                if "attn_proj" in k:
                    # Use xavier uniform for attention projection layer
                    init.xavier_uniform_(m.weight, gain=np.sqrt(2.0))
                elif "out_proj" in k:
                    # Use default pytorch init for out projection layer
                    pass
                else:
                    raise NotImplementedError
            elif "mlp_block" in k:
                # MLP block init
                init.xavier_uniform_(m.weight)
                init.normal_(m.bias, std=1e-6)
            elif "head_fc" in k:
                # Head (classifier) init
                init.zeros_(m.weight)
                init.zeros_(m.bias)
            else:
                raise NotImplementedError
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.LayerNorm):
            # Use default pytorch init for norm layers
            pass
    # Pos-embedding init
    init.normal_(model.pos_embedding, mean=0.0, std=0.02)

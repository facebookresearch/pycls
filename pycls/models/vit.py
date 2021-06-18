#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Vision Transformer implementation from https://arxiv.org/abs/2010.11929.

Code modified from
https://github.com/facebookresearch/ClassyVision/blob/master/classy_vision/models/vision_transformer.py

References for the code above:
https://github.com/google-research/vision_transformer
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""

import logging
import math
from collections import OrderedDict
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from pycls.core.config import cfg
from pycls.models.blocks import conv2d_cx, linear_cx, multi_head_attention_cx, norm2d_cx


LayerNorm = partial(nn.LayerNorm, eps=1e-6)


class VisionTransformerHead(nn.Module):
    def __init__(self, in_plane, num_classes):
        super().__init__()
        layers = [("head", nn.Linear(in_plane, num_classes))]
        self.layers = nn.Sequential(OrderedDict(layers))
        self.init_weights()

    def init_weights(self):
        nn.init.zeros_(self.layers.head.weight)
        nn.init.zeros_(self.layers.head.bias)

    def forward(self, x):
        return self.layers(x)

    @staticmethod
    def complexity(cx, w_in, w_out):
        cx = linear_cx(cx, w_in, w_out, bias=True)
        return cx


class MLPBlock(nn.Sequential):
    """Transformer MLP block."""

    def __init__(self, in_dim, mlp_dim, dropout_rate):
        super().__init__()
        self.in_dim = in_dim
        self.mlp_dim = mlp_dim
        self.linear_1 = nn.Linear(in_dim, mlp_dim)
        self.act = nn.GELU()
        self.dropout_1 = nn.Dropout(dropout_rate, inplace=True)
        self.linear_2 = nn.Linear(mlp_dim, in_dim)
        self.dropout_2 = nn.Dropout(dropout_rate, inplace=True)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.linear_1.weight)
        nn.init.xavier_uniform_(self.linear_2.weight)
        nn.init.normal_(self.linear_1.bias, std=1e-6)
        nn.init.normal_(self.linear_2.bias, std=1e-6)

    @staticmethod
    def complexity(cx, in_dim, mlp_dim):
        h, w, flops, params, acts = (
            cx["h"],
            cx["w"],
            cx["flops"],
            cx["params"],
            cx["acts"],
        )
        seq_length = h * w
        flops += seq_length * (in_dim * mlp_dim + mlp_dim) + seq_length * (
            mlp_dim * in_dim + in_dim
        )
        params += (in_dim * mlp_dim + mlp_dim) + (mlp_dim * in_dim + in_dim)
        acts += seq_length * mlp_dim + seq_length * in_dim
        return {"h": h, "w": w, "flops": flops, "params": params, "acts": acts}


class EncoderBlock(nn.Module):
    """Transformer encoder block.

    From @myleott -
    There are at least three common structures.
    1) Attention is all you need had the worst one, where the layernorm came after each
        block and was in the residual path.
    2) BERT improved upon this by moving the layernorm to the beginning of each block
        (and adding an extra layernorm at the end).
    3) There's a further improved version that also moves the layernorm outside of the
        residual path, which is what this implementation does.

    Figure 1 of this paper compares versions 1 and 3:
        https://openreview.net/pdf?id=B1x8anVFPr
    Figure 7 of this paper compares versions 2 and 3 for BERT:
        https://arxiv.org/abs/1909.08053
    """

    def __init__(
        self, num_heads, hidden_dim, mlp_dim, dropout_rate, attention_dropout_rate
    ):
        super().__init__()
        self.ln_1 = LayerNorm(hidden_dim)
        self.self_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=attention_dropout_rate
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.ln_2 = LayerNorm(hidden_dim)
        self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout_rate)

    def forward(self, input):
        x = self.ln_1(input)
        x, _ = self.self_attention(query=x, key=x, value=x, need_weights=False)
        x = self.dropout(x)
        x = input + x

        y = self.ln_2(x)
        y = self.mlp(y)
        z = x + y

        return z

    @staticmethod
    def complexity(cx, seq_length, num_heads, hidden_dim, mlp_dim):
        cx = norm2d_cx(cx, hidden_dim)
        cx = multi_head_attention_cx(cx, seq_length, num_heads, hidden_dim)
        cx = norm2d_cx(cx, hidden_dim)
        cx = MLPBlock.complexity(cx, hidden_dim, mlp_dim)
        return cx


class Encoder(nn.Module):
    """Transformer Encoder."""

    def __init__(
        self,
        seq_length,
        num_layers,
        num_heads,
        hidden_dim,
        mlp_dim,
        dropout_rate,
        attention_dropout_rate,
    ):
        super().__init__()
        self.pos_embedding = nn.Parameter(
            torch.empty(seq_length, 1, hidden_dim).normal_(std=0.02)
        )
        self.dropout = nn.Dropout(dropout_rate)
        layers = []
        for i in range(num_layers):
            layers.append(
                (
                    f"layer_{i}",
                    EncoderBlock(
                        num_heads,
                        hidden_dim,
                        mlp_dim,
                        dropout_rate,
                        attention_dropout_rate,
                    ),
                )
            )
        self.layers = nn.Sequential(OrderedDict(layers))
        self.ln = LayerNorm(hidden_dim)

    def forward(self, x):
        x = x + self.pos_embedding  # should broadcast to the same shape
        return self.ln(self.layers(self.dropout(x)))

    @staticmethod
    def complexity(cx, seq_length, num_layers, num_heads, hidden_dim, mlp_dim):
        # pos_embedding & x = x + self.pos_embedding
        cx["flops"] += seq_length * hidden_dim
        cx["params"] += seq_length * hidden_dim
        cx["acts"] += seq_length * hidden_dim
        for _ in range(num_layers):
            cx = EncoderBlock.complexity(cx, seq_length, num_heads, hidden_dim, mlp_dim)
        # self.ln
        cx = norm2d_cx(cx, hidden_dim)
        return cx


class VisionTransformer(nn.Module):
    """Vision Transformer as per https://arxiv.org/abs/2010.11929."""

    def __init__(self):
        super().__init__()
        self.image_size = cfg.VIT.IMAGE_SIZE
        self.patch_size = cfg.VIT.PATCH_SIZE
        self.hidden_dim = cfg.VIT.HIDDEN_DIM
        self.mlp_dim = cfg.VIT.MLP_DIM
        self.attention_dropout_rate = cfg.VIT.ATTENTION_DROPOUT_RATE
        self.dropout_rate = cfg.VIT.DROPOUT_RATE
        self.stem_type = cfg.VIT.STEM_TYPE
        self.classifier = cfg.VIT.CLASSIFIER
        assert (
            self.image_size % self.patch_size == 0
        ), "Input shape indivisible by patch size"
        assert self.classifier in ["token", "gap"], "Unexpected classifier mode"
        assert self.stem_type in ["patchify", "conv"], "Unexpected stem type"

        input_channels = 3

        # the input
        if self.stem_type == "patchify":
            self.stem = nn.Conv2d(
                input_channels,
                self.hidden_dim,
                kernel_size=self.patch_size,
                stride=self.patch_size,
            )
        elif self.stem_type == "conv":
            assert self.patch_size == np.prod(
                cfg.VIT.STEM_CONV_STRIDES
            ), "Stem strides inequal to patch size"
            layers = []
            for i in range(len(cfg.VIT.STEM_CONV_DIMS)):
                ks = cfg.VIT.STEM_CONV_KSIZES[i]
                in_dim = cfg.VIT.STEM_CONV_DIMS[i - 1] if i > 0 else input_channels
                out_dim = cfg.VIT.STEM_CONV_DIMS[i]
                stride = cfg.VIT.STEM_CONV_STRIDES[i]
                pd = ks // 2
                has_bias = i == len(cfg.VIT.STEM_CONV_DIMS) - 1
                layers.append(
                    nn.Conv2d(
                        in_dim,
                        out_dim,
                        kernel_size=ks,
                        padding=pd,
                        stride=stride,
                        bias=has_bias,
                    )
                )
                if i != len(cfg.VIT.STEM_CONV_DIMS) - 1:
                    layers = layers + [nn.BatchNorm2d(out_dim), nn.ReLU(inplace=True)]
            self.stem = nn.Sequential(*layers)
        else:
            raise NotImplementedError

        seq_length = (self.image_size // self.patch_size) ** 2
        if self.classifier == "token":
            # add a class token
            self.class_token = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
            seq_length += 1

        self.encoder = Encoder(
            seq_length,
            cfg.VIT.NUM_LAYERS,
            cfg.VIT.NUM_HEADS,
            self.hidden_dim,
            self.mlp_dim,
            self.dropout_rate,
            self.attention_dropout_rate,
        )

        self.head = VisionTransformerHead(cfg.VIT.HIDDEN_DIM, cfg.MODEL.NUM_CLASSES)

        self.seq_length = seq_length
        self.init_weights()

    def init_weights(self):
        if self.stem_type == "patchify":
            fan_in = (
                self.stem.in_channels
                * self.stem.kernel_size[0]
                * self.stem.kernel_size[1]
            )
            nn.init.trunc_normal_(self.stem.weight, std=math.sqrt(1 / fan_in))
            nn.init.zeros_(self.stem.bias)
        if self.stem_type == "conv":
            nn.init.normal_(
                self.stem[-1].weight, mean=0.0, std=math.sqrt(2.0 / self.hidden_dim)
            )
            nn.init.zeros_(self.stem[-1].bias)

    def forward(self, x: torch.Tensor):
        assert x.ndim == 4, "Unexpected input shape"
        n, c, h, w = x.shape
        p = self.patch_size
        assert h == w == self.image_size
        n_h = h // p
        n_w = w // p

        # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
        x = self.stem(x)
        # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
        x = x.reshape(n, self.hidden_dim, n_h * n_w)

        # (n, hidden_dim, (n_h * n_w)) -> ((n_h * n_w), n, hidden_dim)
        # the self attention layer expects inputs in the format (S, N, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension
        x = x.permute(2, 0, 1)

        if self.classifier == "token":
            # expand the class token to the full batch
            batch_class_token = self.class_token.expand(-1, n, -1)
            x = torch.cat([batch_class_token, x], dim=0)

        x = self.encoder(x)

        if self.classifier == "token":
            # just return the output for the class token
            x = x[0, :, :]
        else:
            x = x.mean(dim=0)

        return self.head(x)

    def load_state_dict(self, state):
        # shape of pos_embedding is (seq_length, 1, hidden_dim)
        pos_embedding = state["encoder.pos_embedding"]
        seq_length, n, hidden_dim = pos_embedding.shape
        if n != 1:
            raise ValueError(
                f"Unexpected position embedding shape: {pos_embedding.shape}"
            )
        if hidden_dim != self.hidden_dim:
            raise ValueError(
                f"Position embedding hidden_dim incorrect: {hidden_dim}"
                f", expected: {self.hidden_dim}"
            )
        new_seq_length = self.seq_length

        if new_seq_length != seq_length:
            # need to interpolate the weights for the position embedding
            # we do this by reshaping the positions embeddings to a 2d grid, performing
            # an interpolation in the (h, w) space and then reshaping back to a 1d grid
            if self.classifier == "token":
                # the class token embedding shouldn't be interpolated so we split it up
                seq_length -= 1
                new_seq_length -= 1
                pos_embedding_token = pos_embedding[:1, :, :]
                pos_embedding_img = pos_embedding[1:, :, :]
            else:
                pos_embedding_token = pos_embedding[:0, :, :]  # empty data
                pos_embedding_img = pos_embedding
            # (seq_length, 1, hidden_dim) -> (1, hidden_dim, seq_length)
            pos_embedding_img = pos_embedding_img.permute(1, 2, 0)
            seq_length_1d = int(math.sqrt(seq_length))
            assert (
                seq_length_1d * seq_length_1d == seq_length
            ), "seq_length is not a perfect square"

            logging.info(
                "Interpolating the position embeddings from image "
                f"{seq_length_1d * self.patch_size} to size {self.image_size}"
            )

            # (1, hidden_dim, seq_length) -> (1, hidden_dim, seq_l_1d, seq_l_1d)
            pos_embedding_img = pos_embedding_img.reshape(
                1, hidden_dim, seq_length_1d, seq_length_1d
            )
            new_seq_length_1d = self.image_size // self.patch_size

            # use bicubic interpolation - it gives significantly better results in
            # the test `test_resolution_change`
            new_pos_embedding_img = torch.nn.functional.interpolate(
                pos_embedding_img,
                size=new_seq_length_1d,
                mode="bicubic",
                align_corners=True,
            )

            # (1, hidden_dim, new_seq_l_1d, new_seq_l_1d) -> (1, hidden_dim, new_seq_l)
            new_pos_embedding_img = new_pos_embedding_img.reshape(
                1, hidden_dim, new_seq_length
            )
            # (1, hidden_dim, new_seq_length) -> (new_seq_length, 1, hidden_dim)
            new_pos_embedding_img = new_pos_embedding_img.permute(2, 0, 1)
            new_pos_embedding = torch.cat(
                [pos_embedding_token, new_pos_embedding_img], dim=0
            )
            state["encoder.pos_embedding"] = new_pos_embedding
        super().load_state_dict(state)

    @staticmethod
    def complexity(cx):
        """Computes model complexity. If you alter the model, make sure to update."""
        cx["flops"] = 0
        cx["params"] = 0
        cx["acts"] = 0
        cx["h"] = cfg.TRAIN.IM_SIZE
        cx["w"] = cfg.TRAIN.IM_SIZE

        if cfg.VIT.STEM_TYPE == "patchify":
            cx = conv2d_cx(
                cx, 3, cfg.VIT.HIDDEN_DIM, cfg.VIT.PATCH_SIZE, stride=cfg.VIT.PATCH_SIZE
            )
        elif cfg.VIT.STEM_TYPE == "conv":
            for i in range(len(cfg.VIT.STEM_CONV_DIMS)):
                ks = cfg.VIT.STEM_CONV_KSIZES[i]
                in_dim = cfg.VIT.STEM_CONV_DIMS[i - 1] if i > 0 else 3
                out_dim = cfg.VIT.STEM_CONV_DIMS[i]
                stride = cfg.VIT.STEM_CONV_STRIDES[i]
                has_bias = i == len(cfg.VIT.STEM_CONV_DIMS) - 1
                cx = conv2d_cx(cx, in_dim, out_dim, ks, stride=stride, bias=has_bias)
                if i != len(cfg.VIT.STEM_CONV_DIMS) - 1:
                    cx = norm2d_cx(cx, out_dim)
        else:
            raise NotImplementedError

        seq_length = (cfg.VIT.IMAGE_SIZE // cfg.VIT.PATCH_SIZE) ** 2
        if cfg.VIT.CLASSIFIER == "token":
            seq_length += 1
            cx["params"] += cfg.VIT.HIDDEN_DIM

        cx = Encoder.complexity(
            cx,
            seq_length,
            cfg.VIT.NUM_LAYERS,
            cfg.VIT.NUM_HEADS,
            cfg.VIT.HIDDEN_DIM,
            cfg.VIT.MLP_DIM,
        )
        cx = VisionTransformerHead.complexity(
            cx, cfg.VIT.HIDDEN_DIM, cfg.MODEL.NUM_CLASSES
        )

        return cx

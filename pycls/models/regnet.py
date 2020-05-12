#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""RegNet models."""

import numpy as np
from pycls.core.config import cfg
from pycls.models.anynet import AnyNet


def quantize_float(f, q):
    """Converts a float to closest non-zero int divisible by q."""
    return int(round(f / q) * q)


def adjust_ws_gs_comp(ws, bms, gs):
    """Adjusts the compatibility of widths and groups."""
    ws_bot = [int(w * b) for w, b in zip(ws, bms)]
    gs = [min(g, w_bot) for g, w_bot in zip(gs, ws_bot)]
    ws_bot = [quantize_float(w_bot, g) for w_bot, g in zip(ws_bot, gs)]
    ws = [int(w_bot / b) for w_bot, b in zip(ws_bot, bms)]
    return ws, gs


def get_stages_from_blocks(ws, rs):
    """Gets ws/ds of network at each stage from per block values."""
    ts_temp = zip(ws + [0], [0] + ws, rs + [0], [0] + rs)
    ts = [w != wp or r != rp for w, wp, r, rp in ts_temp]
    s_ws = [w for w, t in zip(ws, ts[:-1]) if t]
    s_ds = np.diff([d for d, t in zip(range(len(ts)), ts) if t]).tolist()
    return s_ws, s_ds


def generate_regnet(w_a, w_0, w_m, d, q=8):
    """Generates per block ws from RegNet parameters."""
    assert w_a >= 0 and w_0 > 0 and w_m > 1 and w_0 % q == 0
    ws_cont = np.arange(d) * w_a + w_0
    ks = np.round(np.log(ws_cont / w_0) / np.log(w_m))
    ws = w_0 * np.power(w_m, ks)
    ws = np.round(np.divide(ws, q)) * q
    num_stages, max_stage = len(np.unique(ws)), ks.max() + 1
    ws, ws_cont = ws.astype(int).tolist(), ws_cont.tolist()
    return ws, num_stages, max_stage, ws_cont


class RegNet(AnyNet):
    """RegNet model."""

    @staticmethod
    def get_args():
        """Convert RegNet to AnyNet parameter format."""
        # Generate RegNet ws per block
        w_a, w_0, w_m, d = cfg.REGNET.WA, cfg.REGNET.W0, cfg.REGNET.WM, cfg.REGNET.DEPTH
        ws, num_stages, _, _ = generate_regnet(w_a, w_0, w_m, d)
        # Convert to per stage format
        s_ws, s_ds = get_stages_from_blocks(ws, ws)
        # Use the same gw, bm and ss for each stage
        s_gs = [cfg.REGNET.GROUP_W for _ in range(num_stages)]
        s_bs = [cfg.REGNET.BOT_MUL for _ in range(num_stages)]
        s_ss = [cfg.REGNET.STRIDE for _ in range(num_stages)]
        # Adjust the compatibility of ws and gws
        s_ws, s_gs = adjust_ws_gs_comp(s_ws, s_bs, s_gs)
        # Get AnyNet arguments defining the RegNet
        return {
            "stem_type": cfg.REGNET.STEM_TYPE,
            "stem_w": cfg.REGNET.STEM_W,
            "block_type": cfg.REGNET.BLOCK_TYPE,
            "ds": s_ds,
            "ws": s_ws,
            "ss": s_ss,
            "bms": s_bs,
            "gws": s_gs,
            "se_r": cfg.REGNET.SE_R if cfg.REGNET.SE_ON else None,
            "nc": cfg.MODEL.NUM_CLASSES,
        }

    def __init__(self):
        kwargs = RegNet.get_args()
        super(RegNet, self).__init__(**kwargs)

    @staticmethod
    def complexity(cx, **kwargs):
        """Computes model complexity. If you alter the model, make sure to update."""
        kwargs = RegNet.get_args() if not kwargs else kwargs
        return AnyNet.complexity(cx, **kwargs)

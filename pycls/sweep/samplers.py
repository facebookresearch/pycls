#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Sweep sampling functions."""

import numpy as np
import pycls.core.builders as builders
import pycls.core.net as net
import pycls.models.regnet as regnet
import pycls.sweep.random as rand
from pycls.core.config import cfg
from pycls.sweep.random import validate_rand


def scalar_sampler(sampler):
    """Sampler for scalars in RANGE quantized to QUANTIZE."""
    low, high = sampler.RANGE[0], sampler.RANGE[1]
    rand_fun, q = rand.rand_types[sampler.RAND_TYPE], sampler.QUANTIZE
    return rand_fun(low, high, q)


def value_sampler(sampler):
    """Sampler for uniform sampling from a list of values."""
    rand_index = np.random.randint(len(sampler.VALUES))
    return sampler.VALUES[rand_index]


def list_sampler(sampler):
    """Sampler for a list of n items sampled independently by the item_sampler."""
    item_sampler, n = sampler.ITEM_SAMPLER, sampler.LENGTH
    sampler_function = sampler_types[item_sampler.TYPE]
    return [sampler_function(item_sampler) for _ in range(n)]


def regnet_sampler(sampler):
    """Sampler for main RegNet parameters."""
    d = rand.uniform(*sampler.DEPTH, 1)
    w0 = rand.log_uniform(*sampler.W0, 8)
    wa = rand.log_uniform(*sampler.WA, 0.1)
    wm = rand.log_uniform(*sampler.WM, 0.001)
    gw = rand.power2_or_log_uniform(*sampler.GROUP_W, 8)
    bm = rand.power2_uniform(*sampler.BOT_MUL, 1 / 128)
    params = ["DEPTH", d, "W0", w0, "WA", wa, "WM", wm, "GROUP_W", gw, "BOT_MUL", bm]
    return ["REGNET." + p if i % 2 == 0 else p for i, p in enumerate(params)]


sampler_types = {
    "float_sampler": scalar_sampler,
    "int_sampler": scalar_sampler,
    "value_sampler": value_sampler,
    "list_sampler": list_sampler,
    "regnet_sampler": regnet_sampler,
}


def validate_sampler(param, sampler):
    """Performs various checks on sampler to see if it is valid."""
    if sampler.TYPE in ["int_sampler", "float_sampler"]:
        validate_rand(param, sampler.RAND_TYPE, *sampler.RANGE, sampler.QUANTIZE)
    elif sampler.TYPE == "regnet_sampler":
        assert param == "REGNET", "regnet_sampler can only be used for REGNET"
        validate_rand("REGNET.DEPTH", "uniform", *sampler.DEPTH, 1)
        validate_rand("REGNET.W0", "log_uniform", *sampler.W0, 8)
        validate_rand("REGNET.WA", "log_uniform", *sampler.WA, 0.1)
        validate_rand("REGNET.WM", "log_uniform", *sampler.WM, 0.001)
        validate_rand("REGNET.GROUP_W", "power2_or_log_uniform", *sampler.GROUP_W, 8)
        validate_rand("REGNET.BOT_MUL", "power2_uniform", *sampler.BOT_MUL, 1 / 128)


def is_composite_sampler(sampler_type):
    """Composite samplers return a [key, val, ...] list as opposed to just a val."""
    composite_samplers = ["regnet_sampler"]
    return sampler_type in composite_samplers


def sample_parameters(samplers):
    """Samples params [key, val, ...] list based on the samplers."""
    params = []
    for param, sampler in samplers.items():
        val = sampler_types[sampler.TYPE](sampler)
        is_composite = is_composite_sampler(sampler.TYPE)
        params.extend(val if is_composite else [param, val])
    return params


def check_regnet_constraints(constraints):
    """Checks RegNet specific constraints."""
    if cfg.MODEL.TYPE == "regnet":
        wa, w0, wm, d = cfg.REGNET.WA, cfg.REGNET.W0, cfg.REGNET.WM, cfg.REGNET.DEPTH
        _, _, num_s, max_s, _, _ = regnet.generate_regnet(wa, w0, wm, d, 8)
        num_stages = constraints.REGNET.NUM_STAGES
        if num_s != max_s or not num_stages[0] <= num_s <= num_stages[1]:
            return False
    return True


def check_complexity_constraints(constraints):
    """Checks complexity constraints."""
    cx, valid = None, True
    for p, v in constraints.CX.items():
        p, min_v, max_v = p.lower(), v[0], v[1]
        if min_v != 0 or max_v != 0:
            cx = cx if cx else net.complexity(builders.get_model())
            min_v = cx[p] if min_v == 0 else min_v
            max_v = cx[p] if max_v == 0 else max_v
            valid = valid and (min_v <= cx[p] <= max_v)
    return valid

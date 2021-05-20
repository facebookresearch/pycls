#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


"""Sweep analysis functions."""

import json
from functools import reduce
from operator import getitem

import numpy as np


def load_sweep(sweep_file):
    """Loads sweep data from a file."""
    with open(sweep_file, "r") as f:
        sweep = json.load(f)
    sweep = [data for data in sweep if "test_ema_epoch" in data]
    augment_sweep(sweep)
    return sweep


def augment_sweep(sweep):
    """Augments sweep data with fields useful for analysis."""
    # Augment data with "aug" field
    for data in sweep:
        data["aug"] = {}
    # Augment with "aug.lr_wd" field = log(lr) + log(wd) = log(lr * wd)
    lrs, wds = get_vals(sweep, "lr"), get_vals(sweep, "wd")
    for data, lr, wd in zip(sweep, lrs, wds):
        data["aug"]["lr_wd"] = lr + wd
    # Augment with "aug.done" field
    epoch_ind = get_vals(sweep, "test_epoch.epoch_ind")
    epoch_max = get_vals(sweep, "test_epoch.epoch_max")
    for data, i, m in zip(sweep, epoch_ind, epoch_max):
        data["aug"]["done"] = i[-1] / m[-1]
    # Augment with "ema_gain"
    errors = get_vals(sweep, "test_epoch.min_top1_err")
    errors_ema = get_vals(sweep, "test_ema_epoch.min_top1_err")
    for data, error, error_ema in zip(sweep, errors, errors_ema):
        data["aug"]["ema_gain"] = max(0, min(error) - min(error_ema))


def sort_sweep(sweep, metric, reverse=False):
    """Sorts sweep by any metric (including non scalar metrics)."""
    keys = get_vals(sweep, metric)
    keys = [k if np.isscalar(k) else json.dumps(k, sort_keys=True) for k in keys]
    keys, sweep = zip(*sorted(zip(keys, sweep), key=lambda k: k[0], reverse=reverse))
    return sweep, keys


def describe_sweep(sweep, reverse=False):
    """Generate a string description of sweep."""
    keys = ["error_ema", "error_tst", "done", "log_file", "cfg.DESC"]
    formats = ["ema={:.2f}", "err={:.2f}", "done={:.2f}", "{}", "{}"]
    vals = [get_vals(sweep, key) for key in keys]
    vals[3] = [v.split("/")[-2] for v in vals[3]]
    desc = [" | ".join(formats).format(*val) for val in zip(*vals)]
    desc = [s for _, s in sorted(zip(vals[0], desc), reverse=reverse)]
    return "\n".join(desc)


metrics_info = {
    # Each metric has the form [compound_key, label, transform]
    "error": ["test_ema_epoch.min_top1_err", "", min],
    "error_ema": ["test_ema_epoch.min_top1_err", "", min],
    "error_tst": ["test_epoch.min_top1_err", "", min],
    "done": ["aug.done", "fraction done", None],
    "epochs": ["cfg.OPTIM.MAX_EPOCH", "epochs", None],
    # Complexity metrics
    "flops": ["complexity.flops", "flops (B)", lambda v: v / 1e9],
    "params": ["complexity.params", "params (M)", lambda v: v / 1e6],
    "acts": ["complexity.acts", "activations (M)", lambda v: v / 1e6],
    "memory": ["train_epoch.mem", "memory (GB)", lambda v: max(v) / 1e3],
    "resolution": ["cfg.TRAIN.IM_SIZE", "resolution", None],
    "epoch_fw_bw": ["epoch_times.train_fw_bw_time", "epoch fw_bw time (s)", None],
    "epoch_time": ["train_epoch.time_epoch", "epoch total time (s)", np.mean],
    "batch_size": ["cfg.TRAIN.BATCH_SIZE", "batch size", None],
    # Regnet metrics
    "regnet_depth": ["cfg.REGNET.DEPTH", "depth", None],
    "regnet_w0": ["cfg.REGNET.W0", "w0", None],
    "regnet_wa": ["cfg.REGNET.WA", "wa", None],
    "regnet_wm": ["cfg.REGNET.WM", "wm", None],
    "regnet_gw": ["cfg.REGNET.GROUP_W", "gw", None],
    "regnet_bm": ["cfg.REGNET.BOT_MUL", "bm", None],
    # Anynet metrics
    "anynet_ds": ["cfg.ANYNET.DEPTHS", "ds", None],
    "anynet_ws": ["cfg.ANYNET.WIDTHS", "ws", None],
    "anynet_gs": ["cfg.ANYNET.GROUP_WS", "gs", None],
    "anynet_bs": ["cfg.ANYNET.BOT_MULS", "bs", None],
    "anynet_d": ["cfg.ANYNET.DEPTHS", "d", sum],
    "anynet_w": ["cfg.ANYNET.WIDTHS", "w", max],
    "anynet_g": ["cfg.ANYNET.GROUP_WS", "g", max],
    "anynet_b": ["cfg.ANYNET.BOT_MULS", "b", max],
    # Effnet metrics
    "effnet_ds": ["cfg.EN.DEPTHS", "ds", None],
    "effnet_ws": ["cfg.EN.WIDTHS", "ws", None],
    "effnet_ss": ["cfg.EN.STRIDES", "ss", None],
    "effnet_bs": ["cfg.EN.EXP_RATIOS", "bs", None],
    "effnet_d": ["cfg.EN.DEPTHS", "d", sum],
    "effnet_w": ["cfg.EN.WIDTHS", "w", max],
    # Optimization metrics
    "lr": ["cfg.OPTIM.BASE_LR", r"log$_{10}(lr)$", np.log10],
    "min_lr": ["cfg.OPTIM.MIN_LR", r"min_lr", None],
    "wd": ["cfg.OPTIM.WEIGHT_DECAY", r"log$_{10}(wd)$", np.log10],
    "lr_wd": ["aug.lr_wd", r"log$_{10}(lr \cdot wd)$", None],
    "bn_wd": ["cfg.BN.CUSTOM_WEIGHT_DECAY", r"log$_{10}$(bn_wd)", np.log10],
    "momentum": ["cfg.OPTIM.MOMENTUM", "", None],
    "ema_alpha": ["cfg.OPTIM.EMA_ALPHA", r"log$_{10}$(ema_alpha)", np.log10],
    "ema_beta": ["cfg.OPTIM.EMA_BETA", r"log$_{10}$(ema_beta)", np.log10],
    "ema_update": ["cfg.OPTIM.EMA_UPDATE_PERIOD", r"log$_{10}$(ema_update)", np.log2],
}


def get_info(metric):
    """Returns [compound_key, label, transform] for metric."""
    info = metrics_info[metric] if metric in metrics_info else [metric, metric, None]
    info[1] = info[1] if info[1] else metric
    return info


def get_vals(sweep, metric):
    """Gets values for given metric (transformed if metric transform is specified)."""
    compound_key, _, transform = get_info(metric)
    metric_keys = compound_key.split(".")
    vals = [reduce(getitem, metric_keys, data) for data in sweep]
    vals = [transform(v) for v in vals] if transform else vals
    return vals


def get_filters(sweep, metrics, alpha=5, sample=0.25, b=2500):
    """Use empirical bootstrap to estimate filter ranges per metric for good errors."""
    assert len(sweep), "Sweep cannot be empty."
    errs = np.array(get_vals(sweep, "error"))
    n, b, filters = len(errs), int(b), {}
    percentiles = [alpha / 2, 50, 100 - alpha / 2]
    n_sample = int(sample) if sample > 1 else max(1, int(n * sample))
    samples = [np.random.choice(n, n_sample) for _ in range(b)]
    samples = [s[np.argmin(errs[s])] for s in samples]
    for metric in metrics:
        vals = np.array(get_vals(sweep, metric))
        vals = [vals[s] for s in samples]
        v_min, v_med, v_max = tuple(np.percentile(vals, percentiles))
        filters[metric] = [v_min, v_med, v_max]
    return filters


def apply_filters(sweep, filters):
    """Filter sweep according to dict of filters of form {metric: [min, med, max]}."""
    filters = filters if filters else {}
    for metric, (v_min, _, v_max) in filters.items():
        keep = [v_min <= v <= v_max for v in get_vals(sweep, metric)]
        sweep = [data for k, data in zip(keep, sweep) if k]
    return sweep

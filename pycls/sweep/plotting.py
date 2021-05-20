#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Sweep plotting functions."""

import matplotlib.lines as lines
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pycls.models.regnet as regnet
from pycls.sweep.analysis import get_info, get_vals, sort_sweep


# Global color scheme and fill color
_COLORS, _COLOR_FILL = [], []


def set_plot_style():
    """Sets default plotting styles for all plots."""
    plt.rcParams["figure.figsize"] = [3.0, 2]
    plt.rcParams["axes.linewidth"] = 1
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.4
    plt.rcParams["xtick.bottom"] = False
    plt.rcParams["ytick.left"] = False
    plt.rcParams["legend.edgecolor"] = "0.3"
    plt.rcParams["axes.xmargin"] = 0.025
    plt.rcParams["lines.linewidth"] = 1.25
    plt.rcParams["lines.markersize"] = 5.0
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.titlesize"] = 10
    plt.rcParams["legend.fontsize"] = 8
    plt.rcParams["legend.title_fontsize"] = 8
    plt.rcParams["xtick.labelsize"] = 7
    plt.rcParams["ytick.labelsize"] = 7


def set_colors(colors=None):
    """Sets the global color scheme (colors should be a list of rgb float values)."""
    global _COLORS
    default_colors = [
        [0.000, 0.447, 0.741],
        [0.850, 0.325, 0.098],
        [0.929, 0.694, 0.125],
        [0.494, 0.184, 0.556],
        [0.466, 0.674, 0.188],
        [0.301, 0.745, 0.933],
        [0.635, 0.078, 0.184],
        [0.300, 0.300, 0.300],
        [0.600, 0.600, 0.600],
        [1.000, 0.000, 0.000],
    ]
    colors = default_colors if colors is None else colors
    colors, n = np.array(colors), len(colors)
    err_str = "Invalid colors list: {}".format(colors)
    assert ((colors >= 0) & (colors <= 1)).all() and colors.shape[1] == 3, err_str
    _COLORS = np.tile(colors, (int(np.ceil((10000 / n))), 1)).reshape((-1, 3))


def set_color_fill(color_fill=None):
    """Sets the global color fill (color should be a set of rgb float values)."""
    global _COLOR_FILL
    _COLOR_FILL = [0.000, 0.447, 0.741] if color_fill is None else color_fill


def get_color(ind=(), scale=1, dtype=float):
    """Gets color (or colors) referenced by index (or indices)."""
    return np.ndarray.astype(_COLORS[ind] * scale, dtype)


def fig_make(m_rows, n_cols, flatten, **kwargs):
    """Gets figure for plotting with m x n axes."""
    figsize = plt.rcParams["figure.figsize"]
    figsize = (figsize[0] * n_cols, figsize[1] * m_rows)
    fig, axes = plt.subplots(m_rows, n_cols, figsize=figsize, squeeze=False, **kwargs)
    axes = [ax for axes in axes for ax in axes] if flatten else axes
    return fig, axes


def fig_legend(fig, n_cols, names, colors=None, styles=None, markers=None):
    """Adds legend to figure and tweaks layout (call after fig is done)."""
    n, c, s, m = len(names), colors, styles, markers
    c = c if c else get_color()[:n]
    s = [""] * n if s is None else [s] * n if type(s) == str else s
    m = ["o"] * n if m is None else [m] * n if type(m) == str else m
    n_cols = int(np.ceil(n / np.ceil(n / n_cols)))
    hs = [lines.Line2D([0], [0], color=c, ls=s, marker=m) for c, s, m in zip(c, s, m)]
    fig.legend(hs, names, bbox_to_anchor=(0.5, 1.0), loc="lower center", ncol=n_cols)
    fig.tight_layout(pad=0.3, h_pad=1.08, w_pad=1.08)


def plot_edf(sweeps, names):
    """Plots error EDF for each sweep."""
    m, n = 1, 1
    fig, axes = fig_make(m, n, True)
    for i, sweep in enumerate(sweeps):
        k = len(sweep)
        errs = sorted(get_vals(sweep, "error"))
        edf = np.cumsum(np.ones(k) / k)
        label = "{:3d}|{:.1f}|{:.1f}".format(k, min(errs), np.mean(errs))
        axes[0].plot(errs, edf, "-", alpha=0.8, c=get_color(i), label=label)
    axes[0].legend(loc="lower right", title=" " * 10 + "n|min|mean")
    axes[0].set_xlabel("error")
    axes[0].set_ylabel("cumulative prob.")
    fig_legend(fig, n, names, styles="-", markers="")
    return fig


def plot_values(sweeps, names, metrics, filters):
    """Plots scatter plot of error versus metric for each metric and sweep."""
    m, n, c = len(metrics), len(sweeps), _COLOR_FILL
    fig, axes = fig_make(m, n, False, sharex="row", sharey=True)
    e_min = min(min(get_vals(sweep, "error")) for sweep in sweeps)
    e_max = max(max(get_vals(sweep, "error")) for sweep in sweeps)
    e_min, e_max = e_min - (e_max - e_min) / 5, e_max + (e_max - e_min) / 5
    for i, j in [(i, j) for i in range(m) for j in range(n)]:
        metric, sweep, ax, f = metrics[i], sweeps[j], axes[i][j], filters[j]
        errs = get_vals(sweep, "error")
        vals = get_vals(sweep, metric)
        v_min, v_med, v_max = f[metric]
        f = [float(str("{:.3e}".format(f))) for f in f[metric]]
        l_rng, l_med = "[{}, {}]".format(f[0], f[2]), "best: {}".format(f[1])
        ax.scatter(vals, errs, color=get_color(j), alpha=0.8)
        ax.plot([v_med, v_med], [e_min, e_max], c="k", label=l_med)
        ax.fill_between([v_min, v_max], e_min, e_max, alpha=0.1, color=c, label=l_rng)
        ax.legend(loc="upper left")
        ax.set_ylabel("error" if j == 0 else "")
        ax.set_xlabel(get_info(metric)[1])
    fig_legend(fig, n, names)
    return fig


def plot_values_2d(sweeps, names, metric_pairs):
    """Plots color-coded scatter plot for each metric_pair and sweep."""
    m, n = len(metric_pairs), len(sweeps)
    fig, axes = fig_make(m, n, False, sharex="row", sharey="row")
    for i, j in [(i, j) for i in range(m) for j in range(n)]:
        sweep, ax = sweeps[j], axes[i][j]
        metric_x, metric_y = metric_pairs[i]
        xs = get_vals(sweep, metric_x)
        ys = get_vals(sweep, metric_y)
        errs = get_vals(sweep, "error")
        ranks = (np.argsort(np.argsort(errs)) + 1) / len(errs)
        s = ax.scatter(xs, ys, c=ranks, alpha=0.6)
        ax.set_xlabel(get_info(metric_x)[1], fontsize=12)
        ax.set_ylabel(get_info(metric_y)[1], fontsize=12)
        fig.colorbar(s, ax=ax) if j == n - 1 else ()
    fig_legend(fig, n, names)
    return fig


def plot_trends(sweeps, names, metrics, filters, max_cols=0):
    """Plots metric versus sweep for each metric."""
    n_metrics, xs = len(metrics), range(len(sweeps))
    max_cols = max_cols if max_cols else len(sweeps)
    m = int(np.ceil(n_metrics / max_cols))
    n = min(max_cols, int(np.ceil(n_metrics / m)))
    fig, axes = fig_make(m, n, True, sharex=False, sharey=False)
    [ax.axis("off") for ax in axes[n_metrics::]]
    for ax, metric in zip(axes, metrics):
        # Get values to plot
        vals = [get_vals(sweep, metric) for sweep in sweeps]
        vs_min, vs_max = [min(v) for v in vals], [max(v) for v in vals]
        fs_min, fs_med, fs_max = zip(*[f[metric] for f in filters])
        # Show full range
        ax.plot(xs, vs_min, "-", xs, vs_max, "-", c="0.7")
        ax.fill_between(xs, vs_min, vs_max, alpha=0.05, color=_COLOR_FILL)
        # Show good range
        ax.plot(xs, fs_min, "-", xs, fs_max, "-", c="0.5")
        ax.fill_between(xs, fs_min, fs_max, alpha=0.10, color=_COLOR_FILL)
        # Show best range
        ax.plot(xs, fs_med, "-o", c="k")
        # Show good range with markers
        ax.scatter(xs, fs_min, c=get_color(xs), marker="^", s=80, zorder=10)
        ax.scatter(xs, fs_max, c=get_color(xs), marker="v", s=80, zorder=10)
        # Finalize axis
        ax.set_ylabel(get_info(metric)[1])
        ax.set_xticks([])
        ax.set_xlabel("sweep")
    fig_legend(fig, n, names, markers="D")
    return fig


def plot_curves(sweeps, names, metric, n_curves, reverse=False):
    """Plots metric versus epoch for up to best n_curves jobs per sweep."""
    ms = [min(n_curves, len(sweep)) for sweep in sweeps]
    m, n = max(ms), len(sweeps)
    fig, axes = fig_make(m, n, False, sharex=False, sharey=True)
    sweeps = [sort_sweep(sweep, "error", reverse)[0] for sweep in sweeps]
    xs_trn = [get_vals(sweep, "train_epoch.epoch_ind") for sweep in sweeps]
    xs_tst = [get_vals(sweep, "test_epoch.epoch_ind") for sweep in sweeps]
    xs_ema = [get_vals(sweep, "test_ema_epoch.epoch_ind") for sweep in sweeps]
    xs_max = [get_vals(sweep, "test_ema_epoch.epoch_max") for sweep in sweeps]
    ys_trn = [get_vals(sweep, "train_epoch." + metric) for sweep in sweeps]
    ys_tst = [get_vals(sweep, "test_epoch." + metric) for sweep in sweeps]
    ys_ema = [get_vals(sweep, "test_ema_epoch." + metric) for sweep in sweeps]
    ticks = [1, 2, 4, 8, 16, 32, 64, 100]
    y_min = min(min(y) for y in ys_ema + ys_tst for y in y)
    y_min = ticks[np.argmin(np.asarray(ticks) <= y_min) - 1]
    for i, j in [(i, j) for j in range(n) for i in range(ms[j])]:
        ax, x_max = axes[i][j], xs_max[j][i][-1]
        x_trn, y_trn, e_trn = xs_trn[j][i], ys_trn[j][i], min(ys_trn[j][i])
        x_tst, y_tst, e_tst = xs_tst[j][i], ys_tst[j][i], min(ys_tst[j][i])
        x_ema, y_ema, e_ema = xs_ema[j][i], ys_ema[j][i], min(ys_ema[j][i])
        label, prop = "{} {:5.2f}", {"color": get_color(j), "alpha": 0.8}
        ax.plot(x_trn, y_trn, "--", **prop, label=label.format("trn", e_trn))
        ax.plot(x_tst, y_tst, ":", **prop, label=label.format("tst", e_tst))
        ax.plot(x_ema, y_ema, "-", **prop, label=label.format("ema", e_ema))
        ax.plot([x_ema[0], x_ema[-1]], [e_ema, e_ema], "-", color="k", alpha=0.8)
        xy_good = [(x, y) for x, y in zip(x_ema, y_ema) if y < 1.01 * e_ema]
        ax.scatter(*zip(*xy_good), **prop, s=10)
        ax.scatter([np.argmin(y_ema) + 1], e_ema, **prop)
        ax.legend(loc="upper right")
        ax.set_xlim(right=x_max)
    for i, j in [(i, j) for i in range(m) for j in range(n)]:
        ax = axes[i][j]
        ax.set_xlabel("epoch" if i == m - 1 else "")
        ax.set_ylabel(metric if j == 0 else "")
        ax.set_yscale("log", base=2)
        ax.set_yticks(ticks)
        ax.set_yticklabels(ticks)
        ax.set_yticks([t * np.sqrt(2) for t in ticks], minor=True)
        ax.set_yticklabels([], minor=True)
        ax.set_ylim(bottom=y_min, top=100)
        ax.yaxis.grid(True, which="minor")
    fig_legend(fig, n, names, styles="-", markers="")
    return fig


def plot_models(sweeps, names, n_models, reverse=False):
    """Plots model visualization for up to n_models per sweep."""
    ms = [min(n_models, len(sweep)) for sweep in sweeps]
    m, n = max(ms), len(sweeps)
    fig, axes = fig_make(m, n, False, sharex=True, sharey=True)
    sweeps = [sort_sweep(sweep, "error", reverse)[0] for sweep in sweeps]
    for i, j in [(i, j) for j in range(n) for i in range(ms[j])]:
        ax, sweep, color = axes[i][j], [sweeps[j][i]], get_color(j)
        metrics = ["error", "flops", "params", "acts", "epoch_fw_bw", "resolution"]
        vals = [get_vals(sweep, m)[0] for m in metrics]
        label = "e = {:.2f}%, f = {:.2f}B\n".format(*vals[0:2])
        label += "p = {:.2f}M, a = {:.2f}M\n".format(*vals[2:4])
        label += "t = {0:.0f}s, r = ${1:d} \\times {1:d}$\n".format(*vals[4:6])
        model_type = get_vals(sweep, "cfg.MODEL.TYPE")[0]
        if model_type == "regnet":
            metrics = ["GROUP_W", "BOT_MUL", "WA", "W0", "WM", "DEPTH"]
            vals = [get_vals(sweep, "cfg.REGNET." + m)[0] for m in metrics]
            ws, ds, _, _, _, ws_cont = regnet.generate_regnet(*vals[2:])
            label += "$d_i = {:s}$\n$w_i = {:s}$\n".format(str(ds), str(ws))
            label += "$g={:d}$, $b={:g}$, $w_a={:.1f}$\n".format(*vals[:3])
            label += "$w_0={:d}$, $w_m={:.3f}$".format(*vals[3:5])
            ax.plot(ws_cont, ":", c=color)
        elif model_type == "anynet":
            metrics = ["anynet_ds", "anynet_ws", "anynet_gs", "anynet_bs"]
            ds, ws, gs, bs = [get_vals(sweep, m)[0] for m in metrics]
            label += "$d_i = {:s}$\n$w_i = {:s}$\n".format(str(ds), str(ws))
            label += "$g_i = {:s}$\n$b_i = {:s}$".format(str(gs), str(bs))
        elif model_type == "effnet":
            metrics = ["effnet_ds", "effnet_ws", "effnet_ss", "effnet_bs"]
            ds, ws, ss, bs = [get_vals(sweep, m)[0] for m in metrics]
            label += "$d_i = {:s}$\n$w_i = {:s}$\n".format(str(ds), str(ws))
            label += "$s_i = {:s}$\n$b_i = {:s}$".format(str(ss), str(bs))
        else:
            raise AssertionError("Unknown model type" + model_type)
        ws_all = [w for ws in [[w] * d for d, w in zip(ds, ws)] for w in ws]
        ds_cum = np.cumsum([0] + ds[0:-1])
        ax.plot(ws_all, "o-", c=color, markersize=plt.rcParams["lines.markersize"] - 1)
        ax.plot(ds_cum, ws, "o", c="k", fillstyle="none", label=label)
        ax.legend(loc="lower right", markerscale=0, handletextpad=0, handlelength=0)
    for i, j in [(i, j) for i in range(m) for j in range(n)]:
        ax = axes[i][j]
        ax.set_xlabel("block index" if i == m - 1 else "")
        ax.set_ylabel("width" if j == 0 else "")
        ax.set_yscale("log", base=2)
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    fig_legend(fig, n, names, styles="-")
    return fig


# Set global plot style and colors on import
set_plot_style()
set_colors()
set_color_fill()

#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Analyze results of a sweep."""

import os
import time

import matplotlib.pyplot as plt
import pycls.sweep.analysis as analysis
import pycls.sweep.config as sweep_config
import pycls.sweep.plotting as plotting
from pycls.sweep.config import sweep_cfg
from pycls.sweep.htmlbook import Htmlbook


def sweep_analyze():
    """Analyzes results of a sweep."""
    start_time = time.time()
    analyze_cfg = sweep_cfg.ANALYZE
    sweep_dir = os.path.join(sweep_cfg.ROOT_DIR, sweep_cfg.NAME)
    print("Generating sweepbook for {:s}... ".format(sweep_dir), end="", flush=True)
    # Initialize Htmlbook for results
    h = Htmlbook(sweep_cfg.NAME)
    # Output sweep config
    h.add_section("Config")
    with open(sweep_cfg.SWEEP_CFG_FILE, "r") as f:
        sweep_cfg_raw = f.read()
    h.add_details("sweep_cfg", sweep_cfg_raw)
    h.add_details("sweep_cfg_full", str(sweep_cfg))
    # Load sweep and plot EDF
    names = [sweep_cfg.NAME] + analyze_cfg.EXTRA_SWEEP_NAMES
    files = [os.path.join(sweep_cfg.ROOT_DIR, name, "sweep.json") for name in names]
    sweeps = [analysis.load_sweep(file) for file in files]
    names = [os.path.basename(name) for name in names]
    assert all(len(sweep) for sweep in sweeps), "Loaded sweep cannot be empty."
    h.add_section("EDF")
    h.add_plot(plotting.plot_edf(sweeps, names))
    for sweep, name in zip(sweeps, names):
        h.add_details(name, analysis.describe_sweep(sweep))
    # Pre filter sweep according to pre_filters and plot EDF
    pre_filters = analyze_cfg.PRE_FILTERS
    if pre_filters:
        sweeps = [analysis.apply_filters(sweep, pre_filters) for sweep in sweeps]
        assert all(len(sweep) for sweep in sweeps), "Filtered sweep cannot be empty."
        h.add_section("EDF Filtered")
        h.add_plot(plotting.plot_edf(sweeps, names))
        for sweep, name in zip(sweeps, names):
            h.add_details(name, analysis.describe_sweep(sweep))
    # Split sweep according to split_filters and plot EDF
    split_filters = analyze_cfg.SPLIT_FILTERS
    if split_filters and len(names) == 1:
        names = list(split_filters.keys())
        sweeps = [analysis.apply_filters(sweeps[0], f) for f in split_filters.values()]
        assert all(len(sweep) for sweep in sweeps), "Split sweep cannot be empty."
        h.add_section("EDF Split")
        h.add_plot(plotting.plot_edf(sweeps, names))
        for sweep, name in zip(sweeps, names):
            h.add_details(name, analysis.describe_sweep(sweep))
    # Plot metric scatter plots
    metrics = analyze_cfg.METRICS
    plot_metric_trends = analyze_cfg.PLOT_METRIC_TRENDS and len(sweeps) > 1
    if metrics and (analyze_cfg.PLOT_METRIC_VALUES or plot_metric_trends):
        h.add_section("Metrics")
        filters = [analysis.get_filters(sweep, metrics) for sweep in sweeps]
        if analyze_cfg.PLOT_METRIC_VALUES:
            h.add_plot(plotting.plot_values(sweeps, names, metrics, filters))
        if plot_metric_trends:
            h.add_plot(plotting.plot_trends(sweeps, names, metrics, filters))
    # Plot complexity scatter plots
    complexity = analyze_cfg.COMPLEXITY
    plot_complexity_trends = analyze_cfg.PLOT_COMPLEXITY_TRENDS and len(sweeps) > 1
    if complexity and (analyze_cfg.PLOT_COMPLEXITY_VALUES or plot_complexity_trends):
        h.add_section("Complexity")
        filters = [analysis.get_filters(sweep, complexity) for sweep in sweeps]
        if analyze_cfg.PLOT_COMPLEXITY_VALUES:
            h.add_plot(plotting.plot_values(sweeps, names, complexity, filters))
        if plot_complexity_trends:
            h.add_plot(plotting.plot_trends(sweeps, names, complexity, filters))
    # Plot best/worst error curves
    n = analyze_cfg.PLOT_CURVES_BEST
    if n > 0:
        h.add_section("Best Errors")
        h.add_plot(plotting.plot_curves(sweeps, names, "top1_err", n, False))
    n = analyze_cfg.PLOT_CURVES_WORST
    if n > 0:
        h.add_section("Worst Errors")
        h.add_plot(plotting.plot_curves(sweeps, names, "top1_err", n, True))
    # Plot best/worst models
    n = analyze_cfg.PLOT_MODELS_BEST
    if n > 0:
        h.add_section("Best Models")
        h.add_plot(plotting.plot_models(sweeps, names, n, False))
    n = analyze_cfg.PLOT_MODELS_WORST
    if n > 0:
        h.add_section("Worst Models")
        h.add_plot(plotting.plot_models(sweeps, names, n, True))
    # Output Htmlbook and finalize analysis
    h.to_file(os.path.join(sweep_dir, "analysis.html"))
    plt.close("all")
    print("Done [t={:.1f}s]".format(time.time() - start_time))


def main():
    desc = "Analyze results of a sweep."
    sweep_config.load_cfg_fom_args(desc)
    sweep_cfg.freeze()
    sweep_analyze()


if __name__ == "__main__":
    main()

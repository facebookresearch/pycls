#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Functions for sampling in the closed interval [low, high] quantized by q."""

from decimal import Decimal

import matplotlib.pyplot as plt
import numpy as np


def quantize(f, q, op=np.floor):
    """Quantizes f to be divisible by q and have q's type."""
    quantized = Decimal(op(f / q)) * Decimal(str(q))
    return type(q)(quantized)


def uniform(low, high, q):
    """Samples uniform value from [low, high] quantized to q."""
    # Samples f in [l, h+q) then quantizes f to [l, h] via floor()
    # This is equivalent to sampling f in (l-q, h] then quantizing via ceil()
    f = np.random.uniform(low, high + q)
    return quantize(f, q, np.floor)


def log_uniform(low, high, q):
    """Samples log uniform value from [low, high] quantized to q."""
    # Samples f in (l-q*, h] then quantizes f to [l, h] via ceil(), where q*=min(q,l/2)
    # This is NOT equivalent to sampling f in [l, h-q) then quantizing via floor()
    f = np.exp(-np.random.uniform(-(np.log(high)), -(np.log(low - min(q, low / 2)))))
    return quantize(f, q, np.ceil)


def power2_uniform(low, high, q):
    """Samples uniform powers of 2 from [low, high] quantized to q."""
    # Samples f2 in [l2, h2+1) then quantizes f2 to [l2, h2] via floor()
    f2 = np.floor(np.random.uniform(np.log2(low), np.log2(high) + 1))
    return quantize(2**f2, q)


def power2_or_log_uniform(low, high, q):
    """Samples uniform powers of 2 or values divisible by q from [low, high]."""
    # The overall CDF is log-linear because range in log_uniform is (q/2, high]
    f = type(q)(power2_uniform(low, high, low))
    f = log_uniform(max(low, q), high, min(high, q)) if f >= q else f
    return f


def normal(low, high, q):
    """Samples values from a clipped normal (Gaussian) distribution quantized to q."""
    # mu/sigma are computed from low/high such that ~99.7% of samples are in range
    f, mu, sigma = np.inf, (low + high) / 2, (high - low) / 6
    while not low <= f <= high:
        f = np.random.normal(mu, sigma)
    return quantize(f, q, np.round)


def log_normal(low, high, q):
    """Samples values from a clipped log-normal distribution quantized to q."""
    # mu/sigma are computed from low/high such that ~99.7% of samples are in range
    log_low, log_high = np.log(low), np.log(high)
    f, mu, sigma = np.inf, (log_low + log_high) / 2, (log_high - log_low) / 6
    while not low <= f <= high:
        f = np.random.lognormal(mu, sigma)
    return quantize(f, q, np.round)


rand_types = {
    "uniform": uniform,
    "log_uniform": log_uniform,
    "power2_uniform": power2_uniform,
    "power2_or_log_uniform": power2_or_log_uniform,
    "normal": normal,
    "log_normal": log_normal,
}


def validate_rand(err_str, rand_type, low, high, q):
    """Validate parameters to random number generators."""
    err_msg = "{}: {}(low={}, high={}, q={}) is invalid."
    err_msg = err_msg.format(err_str, rand_type, low, high, q)
    low_q = Decimal(str(low)) % Decimal(str(q)) == 0
    high_q = Decimal(str(high)) % Decimal(str(q)) == 0
    assert type(q) == type(low) == type(high), err_msg
    assert rand_type in rand_types, err_msg
    assert q > 0 and low <= high, err_msg
    assert low > 0 or rand_type in ["uniform", "normal"], err_msg
    assert low_q and high_q or rand_type == "power2_or_log_uniform", err_msg
    if rand_type in ["power2_uniform", "power2_or_log_uniform"]:
        assert all(np.log2(v).is_integer() for v in [low, high, q]), err_msg


def plot_rand_cdf(rand_type, low, high, q, n=10000):
    """Visualizes CDF of rand_fun, resulting CDF should be linear (or log-linear)."""
    validate_rand("plot_rand_cdf", rand_type, low, high, q)
    samples = [rand_types[rand_type](low, high, q) for _ in range(n)]
    unique = list(np.unique(samples))
    assert min(unique) >= low and max(unique) <= high, "Sampled value out of range."
    cdf = np.cumsum(np.histogram(samples, unique + [np.inf])[0]) / len(samples)
    plot_fun = plt.plot if rand_type in ["uniform", "normal"] else plt.semilogx
    plot_fun(unique, cdf, "o-", [low, low], [0, 1], "-k", [high, high], [0, 1], "-k")
    plot_fun([low, high], [cdf[0], cdf[-1]]) if "normal" not in rand_type else ()
    plt.title("{}(low={}, high={}, q={})".format(rand_type, low, high, q))
    plt.show()


def plot_rand_cdfs():
    """Visualize CDFs of selected distributions, for visualization/debugging only."""
    plot_rand_cdf("uniform", -0.5, 0.5, 0.1)
    plot_rand_cdf("power2_uniform", 2, 512, 1)
    plot_rand_cdf("power2_uniform", 0.25, 8.0, 0.25)
    plot_rand_cdf("log_uniform", 1, 32, 1)
    plot_rand_cdf("log_uniform", 0.5, 16.0, 0.5)
    plot_rand_cdf("power2_or_log_uniform", 1.0, 16.0, 1.0)
    plot_rand_cdf("power2_or_log_uniform", 0.25, 4.0, 4.0)
    plot_rand_cdf("power2_or_log_uniform", 1, 128, 4)

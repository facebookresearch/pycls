#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Meters."""

import datetime
from collections import deque

import numpy as np
import pycls.utils.logging as lu
import pycls.utils.metrics as metrics
from pycls.core.config import cfg
from pycls.utils.timer import Timer


def eta_str(eta_td):
    """Converts an eta timedelta to a fixed-width string format."""
    days = eta_td.days
    hrs, rem = divmod(eta_td.seconds, 3600)
    mins, secs = divmod(rem, 60)
    return "{0:02},{1:02}:{2:02}:{3:02}".format(days, hrs, mins, secs)


class ScalarMeter(object):
    """Measures a scalar value (adapted from Detectron)."""

    def __init__(self, window_size):
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0

    def reset(self):
        self.deque.clear()
        self.total = 0.0
        self.count = 0

    def add_value(self, value):
        self.deque.append(value)
        self.count += 1
        self.total += value

    def get_win_median(self):
        return np.median(self.deque)

    def get_win_avg(self):
        return np.mean(self.deque)

    def get_global_avg(self):
        return self.total / self.count


class TrainMeter(object):
    """Measures training stats."""

    def __init__(self, epoch_iters):
        self.epoch_iters = epoch_iters
        self.max_iter = cfg.OPTIM.MAX_EPOCH * epoch_iters
        self.iter_timer = Timer()
        self.loss = ScalarMeter(cfg.LOG_PERIOD)
        self.loss_total = 0.0
        self.lr = None
        # Current minibatch errors (smoothed over a window)
        self.mb_top1_err = ScalarMeter(cfg.LOG_PERIOD)
        self.mb_top5_err = ScalarMeter(cfg.LOG_PERIOD)
        # Number of misclassified examples
        self.num_top1_mis = 0
        self.num_top5_mis = 0
        self.num_samples = 0

    def reset(self, timer=False):
        if timer:
            self.iter_timer.reset()
        self.loss.reset()
        self.loss_total = 0.0
        self.lr = None
        self.mb_top1_err.reset()
        self.mb_top5_err.reset()
        self.num_top1_mis = 0
        self.num_top5_mis = 0
        self.num_samples = 0

    def iter_tic(self):
        self.iter_timer.tic()

    def iter_toc(self):
        self.iter_timer.toc()

    def update_stats(self, top1_err, top5_err, loss, lr, mb_size):
        # Current minibatch stats
        self.mb_top1_err.add_value(top1_err)
        self.mb_top5_err.add_value(top5_err)
        self.loss.add_value(loss)
        self.lr = lr
        # Aggregate stats
        self.num_top1_mis += top1_err * mb_size
        self.num_top5_mis += top5_err * mb_size
        self.loss_total += loss * mb_size
        self.num_samples += mb_size

    def get_iter_stats(self, cur_epoch, cur_iter):
        eta_sec = self.iter_timer.average_time * (
            self.max_iter - (cur_epoch * self.epoch_iters + cur_iter + 1)
        )
        eta_td = datetime.timedelta(seconds=int(eta_sec))
        mem_usage = metrics.gpu_mem_usage()
        stats = {
            "_type": "train_iter",
            "epoch": "{}/{}".format(cur_epoch + 1, cfg.OPTIM.MAX_EPOCH),
            "iter": "{}/{}".format(cur_iter + 1, self.epoch_iters),
            "time_avg": self.iter_timer.average_time,
            "time_diff": self.iter_timer.diff,
            "eta": eta_str(eta_td),
            "top1_err": self.mb_top1_err.get_win_median(),
            "top5_err": self.mb_top5_err.get_win_median(),
            "loss": self.loss.get_win_median(),
            "lr": self.lr,
            "mem": int(np.ceil(mem_usage)),
        }
        return stats

    def log_iter_stats(self, cur_epoch, cur_iter):
        if (cur_iter + 1) % cfg.LOG_PERIOD != 0:
            return
        stats = self.get_iter_stats(cur_epoch, cur_iter)
        lu.log_json_stats(stats)

    def get_epoch_stats(self, cur_epoch):
        eta_sec = self.iter_timer.average_time * (
            self.max_iter - (cur_epoch + 1) * self.epoch_iters
        )
        eta_td = datetime.timedelta(seconds=int(eta_sec))
        mem_usage = metrics.gpu_mem_usage()
        top1_err = self.num_top1_mis / self.num_samples
        top5_err = self.num_top5_mis / self.num_samples
        avg_loss = self.loss_total / self.num_samples
        stats = {
            "_type": "train_epoch",
            "epoch": "{}/{}".format(cur_epoch + 1, cfg.OPTIM.MAX_EPOCH),
            "time_avg": self.iter_timer.average_time,
            "eta": eta_str(eta_td),
            "top1_err": top1_err,
            "top5_err": top5_err,
            "loss": avg_loss,
            "lr": self.lr,
            "mem": int(np.ceil(mem_usage)),
        }
        return stats

    def log_epoch_stats(self, cur_epoch):
        stats = self.get_epoch_stats(cur_epoch)
        lu.log_json_stats(stats)


class TestMeter(object):
    """Measures testing stats."""

    def __init__(self, max_iter):
        self.max_iter = max_iter
        self.iter_timer = Timer()
        # Current minibatch errors (smoothed over a window)
        self.mb_top1_err = ScalarMeter(cfg.LOG_PERIOD)
        self.mb_top5_err = ScalarMeter(cfg.LOG_PERIOD)
        # Min errors (over the full test set)
        self.min_top1_err = 100.0
        self.min_top5_err = 100.0
        # Number of misclassified examples
        self.num_top1_mis = 0
        self.num_top5_mis = 0
        self.num_samples = 0

    def reset(self, min_errs=False):
        if min_errs:
            self.min_top1_err = 100.0
            self.min_top5_err = 100.0
        self.iter_timer.reset()
        self.mb_top1_err.reset()
        self.mb_top5_err.reset()
        self.num_top1_mis = 0
        self.num_top5_mis = 0
        self.num_samples = 0

    def iter_tic(self):
        self.iter_timer.tic()

    def iter_toc(self):
        self.iter_timer.toc()

    def update_stats(self, top1_err, top5_err, mb_size):
        self.mb_top1_err.add_value(top1_err)
        self.mb_top5_err.add_value(top5_err)
        self.num_top1_mis += top1_err * mb_size
        self.num_top5_mis += top5_err * mb_size
        self.num_samples += mb_size

    def get_iter_stats(self, cur_epoch, cur_iter):
        mem_usage = metrics.gpu_mem_usage()
        iter_stats = {
            "_type": "test_iter",
            "epoch": "{}/{}".format(cur_epoch + 1, cfg.OPTIM.MAX_EPOCH),
            "iter": "{}/{}".format(cur_iter + 1, self.max_iter),
            "time_avg": self.iter_timer.average_time,
            "time_diff": self.iter_timer.diff,
            "top1_err": self.mb_top1_err.get_win_median(),
            "top5_err": self.mb_top5_err.get_win_median(),
            "mem": int(np.ceil(mem_usage)),
        }
        return iter_stats

    def log_iter_stats(self, cur_epoch, cur_iter):
        if (cur_iter + 1) % cfg.LOG_PERIOD != 0:
            return
        stats = self.get_iter_stats(cur_epoch, cur_iter)
        lu.log_json_stats(stats)

    def get_epoch_stats(self, cur_epoch):
        top1_err = self.num_top1_mis / self.num_samples
        top5_err = self.num_top5_mis / self.num_samples
        self.min_top1_err = min(self.min_top1_err, top1_err)
        self.min_top5_err = min(self.min_top5_err, top5_err)
        mem_usage = metrics.gpu_mem_usage()
        stats = {
            "_type": "test_epoch",
            "epoch": "{}/{}".format(cur_epoch + 1, cfg.OPTIM.MAX_EPOCH),
            "time_avg": self.iter_timer.average_time,
            "top1_err": top1_err,
            "top5_err": top5_err,
            "min_top1_err": self.min_top1_err,
            "min_top5_err": self.min_top5_err,
            "mem": int(np.ceil(mem_usage)),
        }
        return stats

    def log_epoch_stats(self, cur_epoch):
        stats = self.get_epoch_stats(cur_epoch)
        lu.log_json_stats(stats)

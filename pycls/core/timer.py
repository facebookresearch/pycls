#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Timer."""

import time


class Timer(object):
    """A simple timer (adapted from Detectron)."""

    def __init__(self):
        self.total_time = None
        self.calls = None
        self.start_time = None
        self.diff = None
        self.average_time = None
        self.reset()

    def tic(self):
        # using time.time as time.clock does not normalize for multithreading
        self.start_time = time.time()

    def toc(self):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls

    def reset(self):
        self.total_time = 0.0
        self.calls = 0
        self.start_time = 0.0
        self.diff = 0.0
        self.average_time = 0.0

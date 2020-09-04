#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Units test for all models in model zoo."""

import datetime
import json
import os
import shutil
import tempfile
import unittest

import pycls.core.builders as builders
import pycls.core.distributed as dist
import pycls.core.logging as logging
import pycls.core.net as net
import pycls.core.trainer as trainer
import pycls.models.model_zoo as model_zoo
from parameterized import parameterized
from pycls.core.config import cfg, reset_cfg


# Location of pycls directory
_PYCLS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")

# If True run selected tests
_RUN_COMPLEXITY_TESTS = True
_RUN_ERROR_TESTS = False
_RUN_TIMING_TESTS = False


def test_complexity(key):
    """Measure the complexity of a single model."""
    reset_cfg()
    cfg_file = os.path.join(_PYCLS_DIR, key)
    cfg.merge_from_file(cfg_file)
    return net.complexity(builders.get_model())


def test_timing(key):
    """Measure the timing of a single model."""
    reset_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(key))
    cfg.PREC_TIME.WARMUP_ITER, cfg.PREC_TIME.NUM_ITER = 5, 50
    cfg.OUT_DIR, cfg.LOG_DEST = tempfile.mkdtemp(), "file"
    dist.multi_proc_run(num_proc=cfg.NUM_GPUS, fun=trainer.time_model)
    log_file = os.path.join(cfg.OUT_DIR, "stdout.log")
    data = logging.sort_log_data(logging.load_log_data(log_file))["iter_times"]
    shutil.rmtree(cfg.OUT_DIR)
    return data


def test_error(key):
    """Measure the error of a single model."""
    reset_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(key))
    cfg.TEST.WEIGHTS = model_zoo.get_weights_file(key)
    cfg.OUT_DIR, cfg.LOG_DEST = tempfile.mkdtemp(), "file"
    dist.multi_proc_run(num_proc=cfg.NUM_GPUS, fun=trainer.test_model)
    log_file = os.path.join(cfg.OUT_DIR, "stdout.log")
    data = logging.sort_log_data(logging.load_log_data(log_file))["test_epoch"]
    data = {"top1_err": data["top1_err"][-1], "top5_err": data["top5_err"][-1]}
    shutil.rmtree(cfg.OUT_DIR)
    return data


def generate_complexity_tests():
    """Generate complexity tests for every model in the configs directory."""
    configs_dir = os.path.join(_PYCLS_DIR, "configs")
    keys = [os.path.join(r, f) for r, _, fs in os.walk(configs_dir) for f in fs]
    keys = [os.path.relpath(k, _PYCLS_DIR) for k in keys if ".yaml" in k]
    generate_tests("complexity", test_complexity, keys)


def generate_timing_tests():
    """Generate timing tests for every model in the model zoo."""
    keys = model_zoo.get_model_list()
    generate_tests("timing", test_timing, keys)


def generate_error_tests():
    """Generate error tests for every model in the model zoo."""
    keys = model_zoo.get_model_list()
    generate_tests("error", test_error, keys)


def generate_tests(test_name, test_fun, keys):
    """Generate and store all the unit tests."""
    data = load_test_data(test_name)
    keys = sorted(k for k in keys if k not in data)
    for key in keys:
        data[key] = test_fun(key)
        print("data['{}'] = {}".format(key, data[key]))
        save_test_data(data, test_name)


def save_test_data(data, test_name):
    """Save the data file for a given set of tests."""
    filename = os.path.join(_PYCLS_DIR, "dev/model_{}.json".format(test_name))
    with open(filename, "w") as file:
        data["date-created"] = str(datetime.datetime.now())
        json.dump(data, file, sort_keys=True, indent=4)


def load_test_data(test_name):
    """Load the data file for a given set of tests."""
    filename = os.path.join(_PYCLS_DIR, "dev/model_{}.json".format(test_name))
    if not os.path.exists(filename):
        return {}
    with open(filename, "r") as f:
        return json.load(f)


def parse_tests(data):
    """Parse the data file in a format useful for the unit tests."""
    return [[f, data[f]] for f in data if f != "date-created"]


class TestComplexity(unittest.TestCase):
    """Generates unit tests for complexity."""

    @parameterized.expand(parse_tests(load_test_data("complexity")), skip_on_empty=True)
    @unittest.skipIf(not _RUN_COMPLEXITY_TESTS, "Skipping complexity tests")
    def test(self, key, out_expected):
        print("Testing complexity of: {}".format(key))
        out = test_complexity(key)
        self.assertEqual(out, out_expected)


class TestError(unittest.TestCase):
    """Generates unit tests for error."""

    @parameterized.expand(parse_tests(load_test_data("error")), skip_on_empty=True)
    @unittest.skipIf(not _RUN_ERROR_TESTS, "Skipping error tests")
    def test(self, key, out_expected):
        print("\nTesting error of: {}".format(key))
        out = test_error(key)
        print("expected = {}".format(out_expected))
        print("measured = {}".format(out))
        for k in out.keys():
            self.assertAlmostEqual(out[k], out_expected[k], 2)


class TestTiming(unittest.TestCase):
    """Generates unit tests for timing."""

    @parameterized.expand(parse_tests(load_test_data("timing")), skip_on_empty=True)
    @unittest.skipIf(not _RUN_TIMING_TESTS, "Skipping timing tests")
    def test(self, key, out_expected):
        print("\nTesting timing of: {}".format(key))
        out = test_timing(key)
        print("expected = {}".format(out_expected))
        print("measured = {}".format(out))
        for k in out.keys():
            self.assertLessEqual(out[k] / out_expected[k], 1.05)
            self.assertLessEqual(out_expected[k] / out[k], 1.05)


if __name__ == "__main__":
    # generate_complexity_tests()
    # generate_timing_tests()
    # generate_error_tests()
    unittest.main()

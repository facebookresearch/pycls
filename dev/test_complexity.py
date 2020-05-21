#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Complexity test for all models in configs."""

import datetime
import json
import os
import unittest

import pycls.core.builders as builders
import pycls.core.net as net
from parameterized import parameterized
from pycls.core.config import cfg


# Location of pycls directory
_PYCLS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")

# Location of all configs
_CONFIGS_DIR = os.path.join(_PYCLS_DIR, "configs")

# Location to store complexity of all models in configs
_COMPLEXITY_FILE = os.path.join(_PYCLS_DIR, "dev/complexity.json")


def dump_complexity():
    """Measure the complexity of every model in the configs/ directory."""
    complexity = {"date-created": str(datetime.datetime.now())}
    cfg_files = [os.path.join(r, f) for r, _, fs in os.walk(_CONFIGS_DIR) for f in fs]
    cfg_files = sorted(f for f in cfg_files if ".yaml" in f)
    for cfg_file in cfg_files:
        cfg_init = cfg.clone()
        cfg.merge_from_file(cfg_file)
        cfg_file_rel = os.path.relpath(cfg_file, _PYCLS_DIR)
        complexity[cfg_file_rel] = net.complexity(builders.get_model())
        cfg.merge_from_other_cfg(cfg_init)
    with open(_COMPLEXITY_FILE, "w") as file:
        json.dump(complexity, file, sort_keys=True, indent=4)


def load_complexity():
    """Load the complexity file in a format useful for TestModelComplexity."""
    with open(_COMPLEXITY_FILE, "r") as f:
        complexity = json.load(f)
    complexity.pop("date-created")
    return [[f, complexity[f]] for f in complexity]


class TestModelComplexity(unittest.TestCase):
    """Test the complexity of every model in the configs/ directory."""

    @parameterized.expand(load_complexity())
    def test_complexity(self, cfg_file_rel, cx_expected):
        """Test complexity of a single model with the specified config."""
        cfg_init = cfg.clone()
        cfg_file = os.path.join(_PYCLS_DIR, cfg_file_rel)
        cfg.merge_from_file(cfg_file)
        cx = net.complexity(builders.get_model())
        cfg.merge_from_other_cfg(cfg_init)
        self.assertEqual(cx_expected, cx)


if __name__ == "__main__":
    unittest.main()

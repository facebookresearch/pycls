#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Scale a model using scaling strategies from "Fast and Accurate Model Scaling".

For reference on scaling strategies, see: https://arxiv.org/abs/2103.06877.
For example usage, see GETTING_STARTED, MODEL SCALING section.
For implementation details, see pycls/models/scaler.py.

This function takes a config as input, scaled the model in the config, and saves the
config for the scaled model back to disk (to OUT_DIR/CFG_DEST). The typical params in
the config that need to specified when scaling a model are MODEL.SCALING_FACTOR and
MODEL.SCALING_TYPE. For the SCALING_TYPE, "d1_w8_g8_r1" gives the fast compound scaling
and is the likely best default option. For further details, see pycls/models/scaler.py.
"""

import pycls.core.builders as builders
import pycls.core.config as config
import pycls.core.net as net
import pycls.models.scaler as scaler


def main():
    config.load_cfg_fom_args("Scale a model.")
    config.assert_and_infer_cfg()
    cx_orig = net.complexity(builders.get_model())
    scaler.scale_model()
    cx_scaled = net.complexity(builders.get_model())
    cfg_file = config.dump_cfg()
    print("Scaled config dumped to:", cfg_file)
    print("Original model complexity:", cx_orig)
    print("Scaled model complexity:", cx_scaled)


if __name__ == "__main__":
    main()

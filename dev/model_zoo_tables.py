#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Generate all MODEL_ZOO.md tables."""

import json
import os

import pycls.core.builders as builders
import pycls.core.net as net
import pycls.models.model_zoo as model_zoo
from pycls.core.config import cfg, reset_cfg


# Location of pycls directory
_PYCLS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")

# Template for tables for each model family
_TABLE_TEMPLATE = [
    "### {model_family} Models",
    "",
    "<table><tbody>",
    "<!-- START {model_family} TABLE -->",
    "<!-- TABLE HEADER -->",
    '<th valign="bottom">model</th>',
    '<th valign="bottom">flops<br/>(B)</th>',
    '<th valign="bottom">params<br/>(M)</th>',
    '<th valign="bottom">acts<br/>(M)</th>',
    '<th valign="bottom">batch<br/>size</th>',
    '<th valign="bottom">infer<br/>(ms)</th>',
    '<th valign="bottom">train<br/>(hr)</th>',
    '<th valign="bottom">error<br/>(top-1)</th>',
    '<th valign="bottom">model id</th>',
    '<th valign="bottom">download</th>',
    "<!-- TABLE BODY -->",
    "{table_rows}",
    "<!-- END {model_family} TABLE -->",
    "</tbody></table>\n",
]


def get_model_data(name, timings, errors):
    """Get model data for a single model."""
    # Load model config
    reset_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(name))
    config_url, _, model_id, _, weight_url_full = model_zoo.get_model_info(name)
    # Get model complexity
    cx = net.complexity(builders.get_model())
    # Inference time is measured in ms with a reference batch_size and num_gpus
    batch_size, num_gpus = 64, 1
    reference = batch_size / cfg.TEST.BATCH_SIZE * cfg.NUM_GPUS / num_gpus
    infer_time = timings[name]["test_fw_time"] * reference * 1000
    # Training time is measured in hours for 100 epochs over the ImageNet train set
    iterations = 1281167 / cfg.TRAIN.BATCH_SIZE * 100
    train_time = timings[name]["train_fw_bw_time"] * iterations / 3600
    # Gather all data about the model
    return {
        "config_url": "configs/" + config_url,
        "flops": round(cx["flops"] / 1e9, 1),
        "params": round(cx["params"] / 1e6, 1),
        "acts": round(cx["acts"] / 1e6, 1),
        "batch_size": cfg.TRAIN.BATCH_SIZE,
        "infer_time": round(infer_time),
        "train_time": round(train_time, 1),
        "error": round(errors[name]["top1_err"], 1),
        "model_id": model_id,
        "weight_url": weight_url_full,
    }


def model_zoo_table_row(name, timings, errors):
    """Make a single row for the MODEL_ZOO.md table."""
    data = get_model_data(name, timings, errors)
    out = "<!-- ROW {} -->\n<tr>\n".format(name)
    template = '<td align="left"><a href="{}">{}</a></td>\n'
    out += template.format(data["config_url"], name)
    template = '<td align="center">{}</td>\n'
    for key in list(data.keys())[1:-1]:
        out += template.format(data[key])
    template = '<td align="center"><a href="{}">model</a></td>\n</tr>'
    out += template.format(data["weight_url"], name)
    return out


def model_zoo_table(model_family):
    """Make MODEL_ZOO.md table for a given model family."""
    filename = _PYCLS_DIR + "/dev/model_{}.json"
    with open(filename.format("timing"), "r") as f:
        timings = json.load(f)
    with open(filename.format("error"), "r") as f:
        errors = json.load(f)
    names = [n for n in model_zoo.get_model_list() if model_family in n]
    table_rows = "\n".join(model_zoo_table_row(n, timings, errors) for n in names)
    table_template = "\n".join(_TABLE_TEMPLATE)
    return table_template.format(model_family=model_family, table_rows=table_rows)


def model_zoo_tables():
    """Make MODEL_ZOO.md tables for all model family."""
    model_families = ["RegNetX", "RegNetY", "ResNet", "ResNeXt", "EfficientNet"]
    out = [model_zoo_table(model_family) for model_family in model_families]
    return "\n".join(out)


if __name__ == "__main__":
    print(model_zoo_tables())

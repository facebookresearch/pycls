#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Functions that handle saving and loading of checkpoints."""

import os

import pycls.core.distributed as dist
import torch
from pycls.core.config import cfg
from pycls.core.io import pathmgr
from pycls.core.net import unwrap_model


# Common prefix for checkpoint file names
_NAME_PREFIX = "model_epoch_"

# Checkpoints directory name
_DIR_NAME = "checkpoints"


def get_checkpoint_dir():
    """Retrieves the location for storing checkpoints."""
    return os.path.join(cfg.OUT_DIR, _DIR_NAME)


def get_checkpoint(epoch):
    """Retrieves the path to a checkpoint file."""
    name = "{}{:04d}.pyth".format(_NAME_PREFIX, epoch)
    return os.path.join(get_checkpoint_dir(), name)


def get_checkpoint_best():
    """Retrieves the path to the best checkpoint file."""
    return os.path.join(cfg.OUT_DIR, "model.pyth")


def get_last_checkpoint():
    """Retrieves the most recent checkpoint (highest epoch number)."""
    checkpoint_dir = get_checkpoint_dir()
    checkpoints = [f for f in pathmgr.ls(checkpoint_dir) if _NAME_PREFIX in f]
    last_checkpoint_name = sorted(checkpoints)[-1]
    return os.path.join(checkpoint_dir, last_checkpoint_name)


def has_checkpoint():
    """Determines if there are checkpoints available."""
    checkpoint_dir = get_checkpoint_dir()
    if not pathmgr.exists(checkpoint_dir):
        return False
    return any(_NAME_PREFIX in f for f in pathmgr.ls(checkpoint_dir))


def save_checkpoint(model, model_ema, optimizer, epoch, test_err, ema_err):
    """Saves a checkpoint and also the best weights so far in a best checkpoint."""

    # Ensure that the checkpoint dir exists
    pathmgr.mkdirs(get_checkpoint_dir())
    # Record the state
    checkpoint = {
        "epoch": epoch,
        "test_err": test_err,
        "ema_err": ema_err,
        "model_state": unwrap_model(model).state_dict(),
        "ema_state": unwrap_model(model_ema).state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "cfg": cfg.dump(),
    }

    # Write the checkpoint
    checkpoint_file = get_checkpoint(epoch + 1)
    # Save checkpoints only from the main process
    if not dist.is_main_proc():
        return

    with pathmgr.open(checkpoint_file, "wb") as f:
        torch.save(checkpoint, f)
    # Store the best model and model_ema weights so far
    if not pathmgr.exists(get_checkpoint_best()):
        with pathmgr.open(get_checkpoint_best(), "wb") as f:
            torch.save(checkpoint, f)
    else:
        with open(get_checkpoint_best(), "rb") as f:
            best = torch.load(f, map_location="cpu")
        # Select the best model weights and the best model_ema weights
        if test_err < best["test_err"] or ema_err < best["ema_err"]:
            if test_err < best["test_err"]:
                best["model_state"] = checkpoint["model_state"]
                best["test_err"] = test_err
            if ema_err < best["ema_err"]:
                best["ema_state"] = checkpoint["ema_state"]
                best["ema_err"] = ema_err
            with pathmgr.open(get_checkpoint_best(), "wb") as f:
                torch.save(best, f)
    return checkpoint_file


def load_checkpoint(checkpoint_file, model, model_ema=None, optimizer=None):
    """
    Loads a checkpoint selectively based on the input options.

    Each checkpoint contains both the model and model_ema weights (except checkpoints
    created by old versions of the code). If both the model and model_weights are
    requested, both sets of weights are loaded. If only the model weights are requested
    (that is if model_ema=None), the *better* set of weights is selected to be loaded
    (according to the lesser of test_err and ema_err, also stored in the checkpoint).

    The code is backward compatible with checkpoints that do not store the ema weights.
    """
    err_str = "Checkpoint '{}' not found"
    assert pathmgr.exists(checkpoint_file), err_str.format(checkpoint_file)
    with pathmgr.open(checkpoint_file, "rb") as f:
        checkpoint = torch.load(f, map_location="cpu")
    # Get test_err and ema_err (with backward compatibility)
    test_err = checkpoint["test_err"] if "test_err" in checkpoint else 100
    ema_err = checkpoint["ema_err"] if "ema_err" in checkpoint else 100
    # Load model and optionally model_ema weights (with backward compatibility)
    ema_state = "ema_state" if "ema_state" in checkpoint else "model_state"
    if model_ema:
        unwrap_model(model).load_state_dict(checkpoint["model_state"])
        unwrap_model(model_ema).load_state_dict(checkpoint[ema_state])
    else:
        best_state = "model_state" if test_err <= ema_err else ema_state
        unwrap_model(model).load_state_dict(checkpoint[best_state])
    # Load optimizer if requested
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    return checkpoint["epoch"], test_err, ema_err


def delete_checkpoints(checkpoint_dir=None, keep="all"):
    """Deletes unneeded checkpoints, keep can be "all", "last", or "none"."""
    assert keep in ["all", "last", "none"], "Invalid keep setting: {}".format(keep)
    checkpoint_dir = checkpoint_dir if checkpoint_dir else get_checkpoint_dir()
    if keep == "all" or not pathmgr.exists(checkpoint_dir):
        return 0
    checkpoints = [f for f in pathmgr.ls(checkpoint_dir) if _NAME_PREFIX in f]
    checkpoints = sorted(checkpoints)[:-1] if keep == "last" else checkpoints
    for checkpoint in checkpoints:
        pathmgr.rm(os.path.join(checkpoint_dir, checkpoint))
    return len(checkpoints)

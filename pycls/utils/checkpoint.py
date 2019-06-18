#!/usr/bin/env python3

"""Functions that handle saving and loading of checkpoints."""

# TODO(ilijar): Resolve memory issues with loading ddp checkpoints
# TODO(ilijar): Get rid of the ddp wrapper when saving checkpoints

import os

import torch

from pycls.core.config import cfg

import pycls.utils.distributed as du
import pycls.utils.logging as logging

logger = logging.get_logger(__name__)

# Common prefix for checkpoint file names
_NAME_PREFIX = 'model_epoch_'

# Checkpoints directory name
_DIR_NAME = 'checkpoints'


def make_checkpoint_dir():
    """Creates the checkpoint directory (if not present already)."""
    checkpoint_dir = os.path.join(cfg.OUT_DIR, _DIR_NAME)
    # Create the checkpoint dir from the master process
    if du.is_master_proc():
        os.makedirs(checkpoint_dir, exist_ok=True)
    return checkpoint_dir


def is_checkpoint_epoch(cur_epoch):
    """Determines if a checkpoint should be saved on current epoch."""
    return (cur_epoch + 1) % cfg.TRAIN.CHECKPOINT_PERIOD == 0


def save_checkpoint(checkpoint_dir, model, optimizer, epoch):
    """Saves a checkpoint."""
    assert os.path.exists(checkpoint_dir), \
        'Checkpoint dir \'{}\' not found'.format(checkpoint_dir)
    # Save checkpoints only from the master process
    if not du.is_master_proc():
        return
    # Record the state
    checkpoint = {
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'cfg': cfg.dump()
    }
    # Write the checkpoint
    file_name = '{}{:04d}.pyth'.format(_NAME_PREFIX, epoch + 1)
    checkpoint_file = os.path.join(checkpoint_dir, file_name)
    torch.save(checkpoint, checkpoint_file)
    logger.info('Wrote checkpoint to: {}'.format(checkpoint_file))


def has_checkpoint(checkpoint_dir):
    """Determines if the given directory contains a checkpoint."""
    assert os.path.exists(checkpoint_dir), \
        'Checkpoint dir \'{}\' not found'.format(checkpoint_dir)
    return any(_NAME_PREFIX in f for f in os.listdir(checkpoint_dir))


def get_last_checkpoint(checkpoint_dir):
    """Retrieves the most recent checkpoint (highest epoch number)."""
    assert os.path.exists(checkpoint_dir), \
        'Checkpoint dir \'{}\' not found'.format(checkpoint_dir)
    # Checkpoint file names are in lexicographic order
    checkpoints = [f for f in os.listdir(checkpoint_dir) if _NAME_PREFIX in f]
    last_checkpoint_name = sorted(checkpoints)[-1]
    last_checkpoint_file = os.path.join(checkpoint_dir, last_checkpoint_name)
    return last_checkpoint_file


def load_checkpoint(checkpoint_file, model, optimizer=None):
    """Loads the checkpoint from the given file."""
    assert os.path.exists(checkpoint_file), \
        'Checkpoint \'{}\' not found'.format(checkpoint_file)
    checkpoint = torch.load(checkpoint_file)
    epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['model_state'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state'])
    logger.info('Loaded checkpoint from: {}'.format(checkpoint_file))
    return epoch

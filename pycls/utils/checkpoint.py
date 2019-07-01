#!/usr/bin/env python3

"""Functions that handle saving and loading of checkpoints."""

# TODO(ilijar): Resolve memory issues with loading ddp checkpoints
# TODO(ilijar): Get rid of the ddp wrapper when saving checkpoints

import os
import torch

from pycls.config import cfg

import pycls.utils.distributed as du


# Common prefix for checkpoint file names
_NAME_PREFIX = 'model_epoch_'

# Checkpoints directory name
_DIR_NAME = 'checkpoints'


def get_checkpoint_dir():
    """Get location for storing checkpoints."""
    return os.path.join(cfg.OUT_DIR, _DIR_NAME)


def get_checkpoint(epoch):
    """Get the full path to a checkpoint file."""
    name = '{}{:04d}.pyth'.format(_NAME_PREFIX, epoch)
    return os.path.join(get_checkpoint_dir(), name)


def get_checkpoint_last():
    d = get_checkpoint_dir()
    names = os.listdir(d) if os.path.exists(d) else []
    names = [f for f in names if _NAME_PREFIX in f]
    assert len(names), 'No checkpoints found in \'{}\'.'.format(d)
    name = sorted(names)[-1]
    return os.path.join(d, name)


def has_checkpoint():
    """Determines if the given directory contains a checkpoint."""
    d = get_checkpoint_dir()
    files = os.listdir(d) if os.path.exists(d) else []
    return any(_NAME_PREFIX in f for f in files)


def is_checkpoint_epoch(cur_epoch):
    """Determines if a checkpoint should be saved on current epoch."""
    return (cur_epoch + 1) % cfg.TRAIN.CHECKPOINT_PERIOD == 0


def save_checkpoint(model, optimizer, epoch):
    """Saves a checkpoint."""
    # Save checkpoints only from the master process
    if not du.is_master_proc():
        return
    os.makedirs(get_checkpoint_dir(), exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'cfg': cfg.dump()
    }
    checkpoint_file = get_checkpoint(epoch + 1)
    torch.save(checkpoint, checkpoint_file)
    return checkpoint_file


def load_checkpoint(checkpoint_file, model, optimizer=None):
    """Loads the checkpoint from the given file."""
    assert os.path.exists(checkpoint_file), \
        'Checkpoint \'{}\' not found'.format(checkpoint_file)
    checkpoint = torch.load(checkpoint_file)
    epoch = checkpoint['epoch']
    # Either the checkpoint or the current model uses DistributedDataParallel, 
    # but the other isn't
    if hasattr(model, 'module'):
        if not next(iter(checkpoint['model_state'])).startswith('module'):
            state = checkpoint['model_state']
            checkpoint['model_state'] = {'module.' + k: state[k]for k in state}
    else:
        if next(iter(checkpoint['model_state'])).startswith('module'):
            state = checkpoint['model_state']
            checkpoint['model_state'] = {k[len('module.'):]: state[k]for k in state}
    model.load_state_dict(checkpoint['model_state'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state'])
    return epoch
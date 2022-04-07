#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Data loader."""

import os

import torch
from pycls.core.config import cfg
from pycls.datasets.cifar10 import Cifar10
from pycls.datasets.imagenet import ImageNet, ImageNetFFCV
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler


try:
    from ffcv.loader import Loader, OrderOption
except ImportError:
    import pycls.core.logging as logging

    logger = logging.get_logger(__name__)
    logger.info("ffcv.loader failed to import")

# Supported data loaders
FFCV = "ffcv"

# Supported datasets
_DATASETS = {"cifar10": Cifar10, "imagenet": ImageNet}
_FFCV_DATASETS = {"imagenet": ImageNetFFCV}

# Default data directory (/path/pycls/pycls/datasets/data)
_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# Relative data paths to default data directory
_PATHS = {"cifar10": "cifar10", "imagenet": "imagenet"}


def _construct_loader(dataset_name, split, batch_size, shuffle, drop_last):
    """Constructs the data loader for the given dataset."""
    err_str = "Dataset '{}' not supported".format(dataset_name)
    assert dataset_name in _DATASETS and dataset_name in _PATHS, err_str
    # Retrieve the data path for the dataset
    data_path = os.path.join(_DATA_DIR, _PATHS[dataset_name])
    # Construct the dataset
    dataset = _DATASETS[dataset_name](data_path, split)
    # Create a sampler for multi-process training
    sampler = DistributedSampler(dataset) if cfg.NUM_GPUS > 1 else None
    # Create a loader
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(False if sampler else shuffle),
        sampler=sampler,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
        pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
        drop_last=drop_last,
    )
    return loader


def _construct_loader_ffcv(dataset_name, split, batch_size, shuffle, drop_last):
    """Constructs the data loader via ffcv for the given dataset."""
    err_str = "Dataset '{}' not supported".format(dataset_name)
    assert dataset_name in _DATASETS and dataset_name in _PATHS, err_str
    # Retrieve the data path for the dataset
    data_path = os.path.join(_DATA_DIR, "..", "ffcv", _PATHS[dataset_name])
    # Construct the dataset
    dataset = _FFCV_DATASETS[dataset_name](data_path, split)
    # Create a loader
    dataset.construct_ffcv()
    loader = Loader(
        dataset.split_path,
        batch_size=batch_size,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
        order=OrderOption.RANDOM if shuffle else OrderOption.SEQUENTIAL,
        os_cache=dataset.os_cache,
        drop_last=drop_last,
        pipelines=dataset.pipelines,
        distributed=dataset.distributed,
    )
    return loader


def construct_train_loader():
    """Train loader wrapper."""
    if cfg.DATA_LOADER.MODE == "ffcv":
        return _construct_loader_ffcv(
            dataset_name=cfg.TRAIN.DATASET,
            split=cfg.TRAIN.SPLIT,
            batch_size=int(cfg.TRAIN.BATCH_SIZE / cfg.NUM_GPUS),
            shuffle=True,
            drop_last=True,
        )
    return _construct_loader(
        dataset_name=cfg.TRAIN.DATASET,
        split=cfg.TRAIN.SPLIT,
        batch_size=int(cfg.TRAIN.BATCH_SIZE / cfg.NUM_GPUS),
        shuffle=True,
        drop_last=True,
    )


def construct_test_loader():
    """Test loader wrapper."""
    if cfg.DATA_LOADER.MODE == "ffcv":
        return _construct_loader_ffcv(
            dataset_name=cfg.TEST.DATASET,
            split=cfg.TEST.SPLIT,
            batch_size=int(cfg.TEST.BATCH_SIZE / cfg.NUM_GPUS),
            shuffle=False,
            drop_last=False,
        )
    return _construct_loader(
        dataset_name=cfg.TEST.DATASET,
        split=cfg.TEST.SPLIT,
        batch_size=int(cfg.TEST.BATCH_SIZE / cfg.NUM_GPUS),
        shuffle=False,
        drop_last=False,
    )


def shuffle(loader, cur_epoch):
    """ "Shuffles the data."""
    err_str = "Sampler type '{}' not supported".format(type(loader.sampler))
    assert isinstance(loader.sampler, (RandomSampler, DistributedSampler)), err_str
    # RandomSampler handles shuffling automatically
    if isinstance(loader.sampler, DistributedSampler):
        # DistributedSampler shuffles data based on epoch
        loader.sampler.set_epoch(cur_epoch)

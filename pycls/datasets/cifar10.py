#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""CIFAR10 dataset."""

import os
import pickle

import numpy as np
import pycls.core.logging as logging
import torch.utils.data
from iopath.common.file_io import g_pathmgr
from pycls.core.config import cfg


logger = logging.get_logger(__name__)

# Per-channel mean and standard deviation values on CIFAR
_MEAN = [125.3, 123.0, 113.9]
_STD = [63.0, 62.1, 66.7]


class Cifar10(torch.utils.data.Dataset):
    """CIFAR-10 dataset."""

    def __init__(self, data_path, split):
        assert g_pathmgr.exists(data_path), "Data path '{}' not found".format(data_path)
        splits = ["train", "test"]
        assert split in splits, "Split '{}' not supported for cifar".format(split)
        logger.info("Constructing CIFAR-10 {}...".format(split))
        self._im_size = cfg.TRAIN.IM_SIZE
        self._data_path, self._split = data_path, split
        self._inputs, self._labels = self._load_data()

    def _load_data(self):
        """Loads data into memory."""
        logger.info("{} data path: {}".format(self._split, self._data_path))
        # Compute data batch names
        if self._split == "train":
            batch_names = ["data_batch_{}".format(i) for i in range(1, 6)]
        else:
            batch_names = ["test_batch"]
        # Load data batches
        inputs, labels = [], []
        for batch_name in batch_names:
            batch_path = os.path.join(self._data_path, batch_name)
            with g_pathmgr.open(batch_path, "rb") as f:
                data = pickle.load(f, encoding="bytes")
            inputs.append(data[b"data"])
            labels += data[b"labels"]
        # Combine and reshape the inputs
        inputs = np.vstack(inputs).astype(np.float32)
        inputs = inputs.reshape((-1, 3, self._im_size, self._im_size))
        return inputs, labels

    def _prepare_im(self, im):
        """Prepares the image for network input."""
        for i in range(3):
            # Perform per-channel normalization on CHW image
            im[i] = (im[i] - _MEAN[i]) / _STD[i]
        if self._split == "train":
            # Randomly flip and crop center patch from CHW image
            size = self._im_size
            im = im[:, :, ::-1] if np.random.uniform() < 0.5 else im
            im = np.pad(im, ((0, 0), (4, 4), (4, 4)), mode="constant")
            y = np.random.randint(0, im.shape[1] - size)
            x = np.random.randint(0, im.shape[2] - size)
            im = im[:, y : (y + size), x : (x + size)]
        return im

    def __getitem__(self, index):
        im, label = self._inputs[index, ...].copy(), self._labels[index]
        im = self._prepare_im(im)
        return im, label

    def __len__(self):
        return self._inputs.shape[0]

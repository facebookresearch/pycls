#!/usr/bin/env python3

"""CIFAR10 dataset."""

import pickle
import numpy as np
import os

import torch
import torch.utils.data

import pycls.datasets.transforms as transforms
import pycls.utils.logging as logging

logger = logging.get_logger(__name__)

# Data location
_GFSAI_DIR = '/mnt/vol/gfsai-flash-east/ai-group/users/ilijar'
_DATA_DIR = _GFSAI_DIR + '/cifar/cifar-10-batches-py'

# Pixel mean and std values in RGB order
_MEAN = [125.3, 123.0, 113.9]
_STD = [63.0, 62.1, 66.7]


class Cifar10(torch.utils.data.Dataset):
    """CIFAR10 dataset."""

    def __init__(self, split):
        assert split in ['train', 'test'], \
            'Split \'{}\' not supported for cifar'.format(split)
        self._split = split
        # Load the data in memory
        #   inputs - (split_size, 3, 32, 32) ndarray
        #   labels - split_size list
        self._inputs, self._labels = self._load_data()

    def _load_batch(self, batch_path):
        with open(batch_path, 'rb') as f:
            d = pickle.load(f, encoding='bytes')
        return d[b'data'], d[b'labels']

    def _load_data(self):
        logger.info(
            'Loading cifar10 {} split from: {}'.format(self._split, _DATA_DIR)
        )
        # Determine data batch names
        if self._split == 'train':
            batch_names = ['data_batch_{}'.format(i) for i in range(1, 6)]
        else:
            batch_names = ['test_batch']
        # Load data batches
        inputs, labels = [], []
        for batch_name in batch_names:
            batch_path = os.path.join(_DATA_DIR, batch_name)
            inputs_batch, labels_batch = self._load_batch(batch_path)
            inputs.append(inputs_batch)
            labels += labels_batch
        # Combine and reshape the inputs
        inputs = np.vstack(inputs).astype(np.float32)
        inputs = inputs.reshape((-1, 3, 32, 32))
        return inputs, labels

    def transform_image(self, image):
        image = transforms.color_normalization(image, _MEAN, _STD)
        if self._split == 'train':
            image = transforms.horizontal_flip(image=image, prob=0.5)
            image = transforms.random_crop(image=image, size=32, pad_size=4)
        return image

    def __getitem__(self, index):
        image, label = self._inputs[index, ...], self._labels[index]
        image = self.transform_image(image)
        return image, label

    def __len__(self):
        return self._inputs.shape[0]

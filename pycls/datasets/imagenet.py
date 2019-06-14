#!/usr/bin/env python3

"""ImageNet dataset."""

import cv2
import numpy as np
import os

import torch
import torch.utils.data

import pycls.datasets.transforms as transforms
import pycls.utils.logging as logging

logger = logging.get_logger(__name__)

# Data location
_LOCAL_DIR = '/data/local/packages/'
_DATA_DIR = _LOCAL_DIR + 'ai-group.imagenet-full-size/prod/imagenet_full_size'

# Pixel mean and std values in BGR order
_MEAN = [0.406, 0.456, 0.485]
_STD = [0.225, 0.224, 0.229]

# Eig vals and vecs of the cov mat
_EIG_VALS = [0.2175, 0.0188, 0.0045]
_EIG_VECS = np.array([
    [-0.5675, 0.7192, 0.4009],
    [-0.5808, -0.0045, -0.8140],
    [-0.5836, -0.6948, 0.4203]
])


class ImageNet(torch.utils.data.Dataset):
    """ImageNet dataset."""

    def __init__(self, split):
        assert split in ['train', 'val'], \
            'Split \'{}\' not supported for ImageNet'.format(split)
        logger.info('Constructing ImageNet {}...'.format(split))
        self._split = split
        self._construct_imdb()

    def _construct_imdb(self):
        """Constructs the imdb."""
        data_dir = os.path.join(_DATA_DIR, self._split)
        assert os.path.exists(data_dir), '{} dir not found'.format(data_dir)

        # Map ImageNet class ids to contiguous ids
        self._class_ids = os.listdir(data_dir)
        self._class_id_cont_id = {v: i for i, v in enumerate(self._class_ids)}

        # Construct the image db
        self._imdb = []
        for class_id in self._class_ids:
            cont_id = self._class_id_cont_id[class_id]
            im_dir = os.path.join(data_dir, class_id)
            for im_name in os.listdir(im_dir):
                self._imdb.append({
                    'im_path': os.path.join(im_dir, im_name),
                    'class': cont_id,
                })

        logger.info('Number of images: {}'.format(len(self._imdb)))
        logger.info('Number of classes: {}'.format(len(self._class_ids)))

    def _prepare_im(self, im):
        if self._split == 'train':
            # Scale and aspect ratio
            im = transforms.random_sized_crop(
                image=im, size=224, area_frac=0.08
            )
            # Horizontal flip
            im = transforms.horizontal_flip(image=im, prob=0.5, order='HWC')
        else:
            # Scale and center crop
            im = transforms.scale(256, im)
            im = transforms.center_crop(224, im)
        # HWC -> CHW
        im = transforms.HWC2CHW(im)
        # [0, 255] -> [0, 1]
        im = im / 255.0
        # PCA jitter
        if self._split == 'train':
            im = transforms.lighting(im, 0.1, _EIG_VALS, _EIG_VECS)
        # Color normalization
        im = transforms.color_normalization(im, _MEAN, _STD)
        return im

    def __getitem__(self, index):
        # Load the image
        im = cv2.imread(self._imdb[index]['im_path'])
        im = im.astype(np.float32, copy=False)
        # Prepare the image for training / testing
        im = self._prepare_im(im)
        # Retrieve the label
        label = self._imdb[index]['class']
        return im, label

    def __len__(self):
        return len(self._imdb)

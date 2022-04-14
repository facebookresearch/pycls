#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""ImageNet dataset."""

import os
import re

import cv2
import numpy as np
import pycls.core.logging as logging
import pycls.datasets.transforms as transforms
import torch.utils.data
from pycls.core.config import cfg


try:
    # FFCV imports
    import torchvision as tv
    from ffcv.fields.basics import IntDecoder
    from ffcv.fields.rgb_image import CenterCropRGBImageDecoder
    from ffcv.fields.rgb_image import RandomResizedCropRGBImageDecoder
    from ffcv.transforms import (
        ToTensor,
        ToDevice,
        Squeeze,
        NormalizeImage,
        RandomHorizontalFlip,
        ToTorchImage,
        Convert,
    )
except ImportError:
    logging.get_logger(__name__).info("ffcv failed to import")

logger = logging.get_logger(__name__)

# Per-channel mean and standard deviation values on ImageNet (in RGB order)
# https://github.com/facebookarchive/fb.resnet.torch/blob/master/datasets/imagenet.lua
_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]

# Constants for lighting normalization on ImageNet (in RGB order)
# https://github.com/facebookarchive/fb.resnet.torch/blob/master/datasets/imagenet.lua
_EIG_VALS = [[0.2175, 0.0188, 0.0045]]
_EIG_VECS = [
    [-0.5675, 0.7192, 0.4009],
    [-0.5808, -0.0045, -0.8140],
    [-0.5836, -0.6948, 0.4203],
]


class ImageNet(torch.utils.data.Dataset):
    """ImageNet dataset."""

    def __init__(self, data_path, split):
        assert os.path.exists(data_path), "Data path '{}' not found".format(data_path)
        splits = ["train", "val"]
        assert split in splits, "Split '{}' not supported for ImageNet".format(split)
        logger.info("Constructing ImageNet {}...".format(split))
        self._data_path, self._split = data_path, split
        self._construct_imdb()

    def _construct_imdb(self):
        """Constructs the imdb."""
        # Compile the split data path
        split_path = os.path.join(self._data_path, self._split)
        logger.info("{} data path: {}".format(self._split, split_path))
        # Images are stored per class in subdirs (format: n<number>)
        split_files = os.listdir(split_path)
        self._class_ids = sorted(f for f in split_files if re.match(r"^n[0-9]+$", f))
        # Map ImageNet class ids to contiguous ids
        self._class_id_cont_id = {v: i for i, v in enumerate(self._class_ids)}
        # Construct the image db
        self._imdb = []
        for class_id in self._class_ids:
            cont_id = self._class_id_cont_id[class_id]
            im_dir = os.path.join(split_path, class_id)
            for im_name in os.listdir(im_dir):
                im_path = os.path.join(im_dir, im_name)
                self._imdb.append({"im_path": im_path, "class": cont_id})
        logger.info("Number of images: {}".format(len(self._imdb)))
        logger.info("Number of classes: {}".format(len(self._class_ids)))

    def _prepare_im(self, im):
        """Prepares the image for network input (HWC/BGR/int -> CHW/BGR/float)."""
        # Convert HWC/BGR/int to HWC/RGB/float format for applying transforms
        im = im[:, :, ::-1].astype(np.float32) / 255
        # Train and test setups differ
        train_size, test_size = cfg.TRAIN.IM_SIZE, cfg.TEST.IM_SIZE
        if self._split == "train":
            # For training use random_sized_crop, horizontal_flip, augment, lighting
            im = transforms.random_sized_crop(im, train_size)
            im = transforms.horizontal_flip(im, prob=0.5)
            im = transforms.augment(im, cfg.TRAIN.AUGMENT)
            im = transforms.lighting(im, cfg.TRAIN.PCA_STD, _EIG_VALS, _EIG_VECS)
        else:
            # For testing use scale and center crop
            im = transforms.scale_and_center_crop(im, test_size, train_size)
        # For training and testing use color normalization
        im = transforms.color_norm(im, _MEAN, _STD)
        # Convert HWC/RGB/float to CHW/BGR/float format
        im = np.ascontiguousarray(im[:, :, ::-1].transpose([2, 0, 1]))
        return im

    def __getitem__(self, index):
        # Load the image
        im = cv2.imread(self._imdb[index]["im_path"])
        # Prepare the image for training / testing
        im = self._prepare_im(im)
        # Retrieve the label
        label = self._imdb[index]["class"]
        return im, label

    def __len__(self):
        return len(self._imdb)


class ImageNetFFCV:
    """ImageNet FFCV dataset."""

    def __init__(self, data_path, split):
        splits = ["train", "val"]
        assert split in splits, "Split '{}' not supported for ImageNet".format(split)
        logger.info("Constructing ImageNet FFCV {}...".format(split))
        self._data_path, self._split = data_path, split

    def construct_ffcv(self):
        imagenet_mean = (np.array(_MEAN) * 255).astype(np.uint8)
        imagenet_std = (np.array(_STD) * 255).astype(np.uint8)
        imagenet_mean_float = np.array(_MEAN) * 255
        imagenet_std_float = np.array(_STD) * 255
        default_crop_ratio = 224 / 256

        # Leverages the operating system for caching purposes.
        # This is beneficial when there is enough memory to cache the dataset
        # and/or when multiple processes on the same machine training using the same dataset.
        self.os_cache = True
        # For distributed training (multiple GPUs).
        # Emulates the behavior of DistributedSampler from PyTorch.
        self.distributed = True
        self.split_path = os.path.join(self._data_path, self._split + ".ffcv")
        cur_device = torch.cuda.current_device()

        if self._split == "train":
            res = cfg.TRAIN.IM_SIZE
            decoder = RandomResizedCropRGBImageDecoder((res, res))
            image_pipeline = [
                decoder,
                RandomHorizontalFlip(),
                ToTensor(),
                ToDevice(cur_device),
            ]
            label_pipeline = [
                IntDecoder(),
                ToTensor(),
                Squeeze(),
                ToDevice(cur_device, non_blocking=True),
            ]
        else:
            res_tuple = (cfg.TRAIN.IM_SIZE, cfg.TRAIN.IM_SIZE)
            cropper = CenterCropRGBImageDecoder(res_tuple, ratio=default_crop_ratio)
            image_pipeline = [cropper, ToTensor(), ToDevice(cur_device)]

            label_pipeline = [
                IntDecoder(),
                ToTensor(),
                Squeeze(),
                ToDevice(cur_device, non_blocking=True),
            ]
        logger.info("Data Path: {}".format(self.split_path))
        # Use torchvision APIs for augmentation
        if cfg.TRAIN.AUGMENT == "":
            image_pipeline.append(ToTorchImage())
            image_pipeline.append(
                NormalizeImage(imagenet_mean, imagenet_std, np.float32)
            )
        else:
            image_pipeline.append(ToTorchImage(channels_last=False))
            params = str(cfg.TRAIN.AUGMENT).split("_")
            if params[0] == "AutoAugment":
                image_pipeline.append(tv.transforms.AutoAugment())
            elif params[0] == "RandAugment":
                names = {"N": "num_ops", "M": "magnitude", "P": "prob"}
                keys = [names[p[0]] for p in params[1:]]
                vals = [float(p[1:]) for p in params[1:]]
                num_ops = int(vals[keys.index("num_ops")])
                magnitude = int(vals[keys.index("magnitude")] * 10)
                logger.warn(
                    "Ignoring probability parameter as it is not supported by torchvision augmentation..."
                )
                image_pipeline.append(
                    tv.transforms.RandAugment(num_ops=num_ops, magnitude=magnitude)
                )
            image_pipeline.append(Convert(torch.float32))
            image_pipeline.append(
                tv.transforms.Normalize(imagenet_mean_float, imagenet_std_float)
            )
        # Pipeline for each data field
        self.pipelines = {"image": image_pipeline, "label": label_pipeline}

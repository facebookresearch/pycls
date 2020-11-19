#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Image transformations on HWC float images with RGB channel order."""

from math import ceil, sqrt

import cv2
import numpy as np
from PIL import Image
from pycls.datasets.augment import make_augment


def scale_and_center_crop(im, scale_size, crop_size):
    """Performs scaling and center cropping (used for testing)."""
    h, w = im.shape[:2]
    if w < h and w != scale_size:
        w, h = scale_size, int(h / w * scale_size)
        im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
    elif h <= w and h != scale_size:
        w, h = int(w / h * scale_size), scale_size
        im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
    x = ceil((w - crop_size) / 2)
    y = ceil((h - crop_size) / 2)
    return im[y : (y + crop_size), x : (x + crop_size), :]


def random_sized_crop(im, size, area_frac=0.08, max_iter=10):
    """Performs Inception-style cropping (used for training)."""
    h, w = im.shape[:2]
    area = h * w
    for _ in range(max_iter):
        target_area = np.random.uniform(area_frac, 1.0) * area
        aspect_ratio = np.random.uniform(3.0 / 4.0, 4.0 / 3.0)
        w_crop = round(sqrt(target_area * aspect_ratio))
        h_crop = round(sqrt(target_area / aspect_ratio))
        if np.random.uniform() < 0.5:
            w_crop, h_crop = h_crop, w_crop
        if h_crop <= h and w_crop <= w:
            y = 0 if h_crop == h else np.random.randint(0, h - h_crop)
            x = 0 if w_crop == w else np.random.randint(0, w - w_crop)
            im = im[y : (y + h_crop), x : (x + w_crop), :]
            return cv2.resize(im, (size, size), interpolation=cv2.INTER_LINEAR)
    return scale_and_center_crop(im, size, size)


def horizontal_flip(im, prob=0.5):
    """Performs horizontal flip (used for training)."""
    return im[:, ::-1, :] if np.random.uniform() < prob else im


def augment(im, augment_str):
    """Augments image (used for training)."""
    if augment_str:
        im = Image.fromarray((im * 255).astype(np.uint8))
        im = make_augment(augment_str)(im)
        im = np.asarray(im).astype(np.float32) / 255
    return im


def lighting(im, alpha_std, eig_val, eig_vec):
    """Performs AlexNet-style PCA jitter (used for training)."""
    alpha = np.random.normal(0, alpha_std, size=(1, 3))
    alpha = np.repeat(alpha, 3, axis=0)
    eig_val = np.repeat(np.array(eig_val), 3, axis=0)
    rgb = np.sum(np.array(eig_vec) * alpha * eig_val, axis=1)
    for i in range(3):
        im[:, :, i] = im[:, :, i] + rgb[i]
    return im


def color_norm(im, mean, std):
    """Performs per-channel normalization (used for training and testing)."""
    for i in range(3):
        im[:, :, i] = (im[:, :, i] - mean[i]) / std[i]
    return im

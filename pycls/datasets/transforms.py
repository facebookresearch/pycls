#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Image transformations."""

import math

import cv2
import numpy as np


def color_norm(im, mean, std):
    """Performs per-channel normalization (CHW format)."""
    for i in range(im.shape[0]):
        im[i] = im[i] - mean[i]
        im[i] = im[i] / std[i]
    return im


def zero_pad(im, pad_size):
    """Performs zero padding (CHW format)."""
    pad_width = ((0, 0), (pad_size, pad_size), (pad_size, pad_size))
    return np.pad(im, pad_width, mode="constant")


def horizontal_flip(im, p, order="CHW"):
    """Performs horizontal flip (CHW or HWC format)."""
    assert order in ["CHW", "HWC"]
    if np.random.uniform() < p:
        if order == "CHW":
            im = im[:, :, ::-1]
        else:
            im = im[:, ::-1, :]
    return im


def random_crop(im, size, pad_size=0):
    """Performs random crop (CHW format)."""
    if pad_size > 0:
        im = zero_pad(im=im, pad_size=pad_size)
    h, w = im.shape[1:]
    y = np.random.randint(0, h - size)
    x = np.random.randint(0, w - size)
    im_crop = im[:, y : (y + size), x : (x + size)]
    assert im_crop.shape[1:] == (size, size)
    return im_crop


def scale(size, im):
    """Performs scaling (HWC format)."""
    h, w = im.shape[:2]
    if (w <= h and w == size) or (h <= w and h == size):
        return im
    h_new, w_new = size, size
    if w < h:
        h_new = int(math.floor((float(h) / w) * size))
    else:
        w_new = int(math.floor((float(w) / h) * size))
    im = cv2.resize(im, (w_new, h_new), interpolation=cv2.INTER_LINEAR)
    return im.astype(np.float32)


def center_crop(size, im):
    """Performs center cropping (HWC format)."""
    h, w = im.shape[:2]
    y = int(math.ceil((h - size) / 2))
    x = int(math.ceil((w - size) / 2))
    im_crop = im[y : (y + size), x : (x + size), :]
    assert im_crop.shape[:2] == (size, size)
    return im_crop


def random_sized_crop(im, size, area_frac=0.08, max_iter=10):
    """Performs Inception-style cropping (HWC format)."""
    h, w = im.shape[:2]
    area = h * w
    for _ in range(max_iter):
        target_area = np.random.uniform(area_frac, 1.0) * area
        aspect_ratio = np.random.uniform(3.0 / 4.0, 4.0 / 3.0)
        w_crop = int(round(math.sqrt(float(target_area) * aspect_ratio)))
        h_crop = int(round(math.sqrt(float(target_area) / aspect_ratio)))
        if np.random.uniform() < 0.5:
            w_crop, h_crop = h_crop, w_crop
        if h_crop <= h and w_crop <= w:
            y = 0 if h_crop == h else np.random.randint(0, h - h_crop)
            x = 0 if w_crop == w else np.random.randint(0, w - w_crop)
            im_crop = im[y : (y + h_crop), x : (x + w_crop), :]
            assert im_crop.shape[:2] == (h_crop, w_crop)
            im_crop = cv2.resize(im_crop, (size, size), interpolation=cv2.INTER_LINEAR)
            return im_crop.astype(np.float32)
    return center_crop(size, scale(size, im))


def lighting(im, alpha_std, eig_val, eig_vec):
    """Performs AlexNet-style PCA jitter (CHW format)."""
    if alpha_std == 0:
        return im
    alpha = np.random.normal(0, alpha_std, size=(1, 3))
    alpha = np.repeat(alpha, 3, axis=0)
    eig_val = np.repeat(eig_val, 3, axis=0)
    rgb = np.sum(eig_vec * alpha * eig_val, axis=1)
    for i in range(im.shape[0]):
        im[i] = im[i] + rgb[2 - i]
    return im

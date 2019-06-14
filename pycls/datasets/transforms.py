#!/usr/bin/env python3

"""Image transformations."""

# TODO(ilijar): consider moving to utils

import cv2
import math
import numpy as np


def CHW2HWC(image):
    return image.transpose([1, 2, 0])


def HWC2CHW(image):
    return image.transpose([2, 0, 1])


def color_normalization(image, mean, std):
    """Expects image in CHW format."""
    assert len(mean) == image.shape[0]
    assert len(std) == image.shape[0]
    for i in range(image.shape[0]):
        image[i] = image[i] - mean[i]
        image[i] = image[i] / std[i]
    return image


def zero_pad(image, pad_size, order='CHW'):
    assert order in ['CHW', 'HWC']
    if order == 'CHW':
        pad_width = ((0, 0), (pad_size, pad_size), (pad_size, pad_size))
    else:
        pad_width = ((pad_size, pad_size), (pad_size, pad_size), (0, 0))
    return np.pad(image, pad_width, mode='constant')


def horizontal_flip(image, prob, order='CHW'):
    assert order in ['CHW', 'HWC']
    if np.random.uniform() < prob:
        if order == 'CHW':
            image = image[:, :, ::-1]
        else:
            image = image[:, ::-1, :]
    return image


def random_crop(image, size, pad_size=0, order='CHW'):
    # TODO(ilijar): Refactor
    assert order in ['CHW', 'HWC']
    if pad_size > 0:
        image = zero_pad(image=image, pad_size=pad_size, order=order)
    if order == 'CHW':
        if image.shape[1] == size and image.shape[2] == size:
            return image
        height = image.shape[1]
        width = image.shape[2]
        y_offset = 0
        if height > size:
            y_offset = int(np.random.randint(0, height - size))
        x_offset = 0
        if width > size:
            x_offset = int(np.random.randint(0, width - size))
        cropped = image[:, y_offset:y_offset + size, x_offset:x_offset + size]
        assert cropped.shape[1] == size, "Image not cropped properly"
        assert cropped.shape[2] == size, "Image not cropped properly"
    else:
        if image.shape[0] == size and image.shape[1] == size:
            return image
        height = image.shape[0]
        width = image.shape[1]
        y_offset = 0
        if height > size:
            y_offset = int(np.random.randint(0, height - size))
        x_offset = 0
        if width > size:
            x_offset = int(np.random.randint(0, width - size))
        cropped = image[y_offset:y_offset + size, x_offset:x_offset + size, :]
        assert cropped.shape[0] == size, "Image not cropped properly"
        assert cropped.shape[1] == size, "Image not cropped properly"
    return cropped


def scale(size, image):
    # TODO(ilijar): Refactor
    height = image.shape[0]
    width = image.shape[1]
    if ((width <= height and width == size) or
            (height <= width and height == size)):
        return image
    new_width = size
    new_height = size
    if width < height:
        new_height = int(math.floor((float(height) / width) * size))
    else:
        new_width = int(math.floor((float(width) / height) * size))
    img = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_LINEAR
    )
    return img.astype(np.float32)


def center_crop(size, image):
    # TODO(ilijar): Refactor
    height = image.shape[0]
    width = image.shape[1]
    y_offset = int(math.ceil((height - size) / 2))
    x_offset = int(math.ceil((width - size) / 2))
    cropped = image[y_offset:y_offset + size, x_offset:x_offset + size, :]
    assert cropped.shape[0] == size, "Image height not cropped properly"
    assert cropped.shape[1] == size, "Image width not cropped properly"
    return cropped


def random_sized_crop(image, size, area_frac=0.08):
    # TODO(ilijar): Refactor
    for _ in range(0, 10):
        height = image.shape[0]
        width = image.shape[1]
        area = height * width
        target_area = np.random.uniform(area_frac, 1.0) * area
        aspect_ratio = np.random.uniform(3.0 / 4.0, 4.0 / 3.0)
        w = int(round(math.sqrt(float(target_area) * aspect_ratio)))
        h = int(round(math.sqrt(float(target_area) / aspect_ratio)))
        if np.random.uniform() < 0.5:
            w, h = h, w
        if h <= height and w <= width:
            if height == h:
                y_offset = 0
            else:
                y_offset = np.random.randint(0, height - h)
            if width == w:
                x_offset = 0
            else:
                x_offset = np.random.randint(0, width - w)
            y_offset = int(y_offset)
            x_offset = int(x_offset)
            cropped = image[y_offset:y_offset + h, x_offset:x_offset + w, :]
            assert cropped.shape[0] == h and cropped.shape[1] == w, \
                "Wrong crop size"
            cropped = cv2.resize(
                cropped,
                (size, size),
                interpolation=cv2.INTER_LINEAR
            )
            return cropped.astype(np.float32)
    return center_crop(size, scale(size, image))


def lighting(img, alphastd, eigval, eigvec):
    # TODO(ilijar): Refactor
    if alphastd == 0:
        return img
    # generate alpha1, alpha2, alpha3
    alpha = np.random.normal(0, alphastd, size=(1, 3))
    eig_vec = np.array(eigvec)
    eig_val = np.reshape(eigval, (1, 3))
    rgb = np.sum(
        eig_vec * np.repeat(alpha, 3, axis=0) * np.repeat(eig_val, 3, axis=0),
        axis=1
    )
    for idx in range(img.shape[0]):
        img[idx] = img[idx] + rgb[2 - idx]
    return img

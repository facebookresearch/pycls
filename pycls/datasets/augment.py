#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Lightweight and simple implementation of AutoAugment and RandAugment.

AutoAugment - https://arxiv.org/abs/1805.09501
RandAugment - https://arxiv.org/abs/1909.13719

http://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/autoaugment.py
Note that the official implementation varies substantially from the papers :-(

Our AutoAugment policy should be fairly identical to the official AutoAugment policy.
The main difference is we set POSTERIZE_MIN = 1, which avoids degenerate (all 0) images.
Our RandAugment policy differs, and uses transforms that increase in intensity with
increasing magnitude. This allows for a more natural control of the magnitude. That is,
setting magnitude = 0 results in ops that leaves the image unchanged, if possible.
We also set the range of the magnitude to be 0 to 1 to avoid setting a "max level".

Our implementation is inspired by and uses policies that are the similar to those in:
https://github.com/rwightman/pytorch-image-models/blob/master/timm/data/auto_augment.py
Specifically our implementation can be *numerically identical* as the implementation in
timm if using timm's "v0" policy for AutoAugment and "inc" transforms for RandAugment
and if we set POSTERIZE_MIN = 0 (although as noted our default is POSTERIZE_MIN = 1).
Note that magnitude in our code ranges from 0 to 1 (compared to 0 to 10 in timm).

Specifically, given the same seeds, the functions from timm:
    out_auto = auto_augment_transform("v0", {"interpolation": 2})(im)
    out_rand = rand_augment_transform("rand-inc1-n2-m05", {"interpolation": 2})(im)
Are numerically equivalent to:
    POSTERIZE_MIN = 0
    out_auto = auto_augment(im)
    out_rand = rand_augment(im, prob=0.5, n_ops=2, magnitude=0.5)
Tested as of 10/07/2020. Can alter corresponding params for both and should match.

Finally, the ops and augmentations can be visualized as follows:
    from PIL import Image
    import pycls.datasets.augment as augment
    im = Image.open("scratch.jpg")
    im_ops = augment.visualize_ops(im)
    im_rand = augment.visualize_aug(im, augment=augment.rand_augment, magnitude=0.5)
    im_auto = augment.visualize_aug(im, augment=augment.auto_augment)
    im_ops.show()
    im_auto.show()
    im_rand.show()
"""

import random

import numpy as np
from PIL import Image, ImageEnhance, ImageOps


# Minimum value for posterize (0 in EfficientNet implementation)
POSTERIZE_MIN = 1

# Parameters for affine warping and rotation
WARP_PARAMS = {"fillcolor": (128, 128, 128), "resample": Image.BILINEAR}


def affine_warp(im, data):
    """Applies affine transform to image."""
    return im.transform(im.size, Image.AFFINE, data, **WARP_PARAMS)


OP_FUNCTIONS = {
    # Each op takes an image x and a level v and returns an augmented image.
    "auto_contrast": lambda x, _: ImageOps.autocontrast(x),
    "equalize": lambda x, _: ImageOps.equalize(x),
    "invert": lambda x, _: ImageOps.invert(x),
    "rotate": lambda x, v: x.rotate(v, **WARP_PARAMS),
    "posterize": lambda x, v: ImageOps.posterize(x, max(POSTERIZE_MIN, int(v))),
    "posterize_inc": lambda x, v: ImageOps.posterize(x, max(POSTERIZE_MIN, 4 - int(v))),
    "solarize": lambda x, v: x.point(lambda i: i if i < int(v) else 255 - i),
    "solarize_inc": lambda x, v: x.point(lambda i: i if i < 256 - v else 255 - i),
    "solarize_add": lambda x, v: x.point(lambda i: min(255, v + i) if i < 128 else i),
    "color": lambda x, v: ImageEnhance.Color(x).enhance(v),
    "contrast": lambda x, v: ImageEnhance.Contrast(x).enhance(v),
    "brightness": lambda x, v: ImageEnhance.Brightness(x).enhance(v),
    "sharpness": lambda x, v: ImageEnhance.Sharpness(x).enhance(v),
    "color_inc": lambda x, v: ImageEnhance.Color(x).enhance(1 + v),
    "contrast_inc": lambda x, v: ImageEnhance.Contrast(x).enhance(1 + v),
    "brightness_inc": lambda x, v: ImageEnhance.Brightness(x).enhance(1 + v),
    "sharpness_inc": lambda x, v: ImageEnhance.Sharpness(x).enhance(1 + v),
    "shear_x": lambda x, v: affine_warp(x, (1, v, 0, 0, 1, 0)),
    "shear_y": lambda x, v: affine_warp(x, (1, 0, 0, v, 1, 0)),
    "trans_x": lambda x, v: affine_warp(x, (1, 0, v * x.size[0], 0, 1, 0)),
    "trans_y": lambda x, v: affine_warp(x, (1, 0, 0, 0, 1, v * x.size[1])),
}


OP_RANGES = {
    # Ranges for each op in the form of a (min, max, negate).
    "auto_contrast": (0, 1, False),
    "equalize": (0, 1, False),
    "invert": (0, 1, False),
    "rotate": (0.0, 30.0, True),
    "posterize": (0, 4, False),
    "posterize_inc": (0, 4, False),
    "solarize": (0, 256, False),
    "solarize_inc": (0, 256, False),
    "solarize_add": (0, 110, False),
    "color": (0.1, 1.9, False),
    "contrast": (0.1, 1.9, False),
    "brightness": (0.1, 1.9, False),
    "sharpness": (0.1, 1.9, False),
    "color_inc": (0, 0.9, True),
    "contrast_inc": (0, 0.9, True),
    "brightness_inc": (0, 0.9, True),
    "sharpness_inc": (0, 0.9, True),
    "shear_x": (0.0, 0.3, True),
    "shear_y": (0.0, 0.3, True),
    "trans_x": (0.0, 0.45, True),
    "trans_y": (0.0, 0.45, True),
}


AUTOAUG_POLICY = [
    # AutoAugment "policy_v0" in form of (op, prob, magnitude), where magnitude <= 1.
    [("equalize", 0.8, 0.1), ("shear_y", 0.8, 0.4)],
    [("color", 0.4, 0.9), ("equalize", 0.6, 0.3)],
    [("color", 0.4, 0.1), ("rotate", 0.6, 0.8)],
    [("solarize", 0.8, 0.3), ("equalize", 0.4, 0.7)],
    [("solarize", 0.4, 0.2), ("solarize", 0.6, 0.2)],
    [("color", 0.2, 0.0), ("equalize", 0.8, 0.8)],
    [("equalize", 0.4, 0.8), ("solarize_add", 0.8, 0.3)],
    [("shear_x", 0.2, 0.9), ("rotate", 0.6, 0.8)],
    [("color", 0.6, 0.1), ("equalize", 1.0, 0.2)],
    [("invert", 0.4, 0.9), ("rotate", 0.6, 0.0)],
    [("equalize", 1.0, 0.9), ("shear_y", 0.6, 0.3)],
    [("color", 0.4, 0.7), ("equalize", 0.6, 0.0)],
    [("posterize", 0.4, 0.6), ("auto_contrast", 0.4, 0.7)],
    [("solarize", 0.6, 0.8), ("color", 0.6, 0.9)],
    [("solarize", 0.2, 0.4), ("rotate", 0.8, 0.9)],
    [("rotate", 1.0, 0.7), ("trans_y", 0.8, 0.9)],
    [("shear_x", 0.0, 0.0), ("solarize", 0.8, 0.4)],
    [("shear_y", 0.8, 0.0), ("color", 0.6, 0.4)],
    [("color", 1.0, 0.0), ("rotate", 0.6, 0.2)],
    [("equalize", 0.8, 0.4), ("equalize", 0.0, 0.8)],
    [("equalize", 1.0, 0.4), ("auto_contrast", 0.6, 0.2)],
    [("shear_y", 0.4, 0.7), ("solarize_add", 0.6, 0.7)],
    [("posterize", 0.8, 0.2), ("solarize", 0.6, 1.0)],
    [("solarize", 0.6, 0.8), ("equalize", 0.6, 0.1)],
    [("color", 0.8, 0.6), ("rotate", 0.4, 0.5)],
]


RANDAUG_OPS = [
    # RandAugment list of operations using "increasing" transforms.
    "auto_contrast",
    "equalize",
    "invert",
    "rotate",
    "posterize_inc",
    "solarize_inc",
    "solarize_add",
    "color_inc",
    "contrast_inc",
    "brightness_inc",
    "sharpness_inc",
    "shear_x",
    "shear_y",
    "trans_x",
    "trans_y",
]


def apply_op(im, op, prob, magnitude):
    """Apply the selected op to image with given probability and magnitude."""
    # The magnitude is converted to an absolute value v for an op (some ops use -v or v)
    assert 0 <= magnitude <= 1
    assert op in OP_RANGES and op in OP_FUNCTIONS, "unknown op " + op
    if prob < 1 and random.random() > prob:
        return im
    min_v, max_v, negate = OP_RANGES[op]
    v = magnitude * (max_v - min_v) + min_v
    v = -v if negate and random.random() > 0.5 else v
    return OP_FUNCTIONS[op](im, v)


def rand_augment(im, magnitude, ops=None, n_ops=2, prob=1.0):
    """Applies random augmentation to an image."""
    ops = ops if ops else RANDAUG_OPS
    for op in np.random.choice(ops, int(n_ops)):
        im = apply_op(im, op, prob, magnitude)
    return im


def auto_augment(im, policy=None):
    """Apply auto augmentation to an image."""
    policy = policy if policy else AUTOAUG_POLICY
    for op, prob, magnitude in random.choice(policy):
        im = apply_op(im, op, prob, magnitude)
    return im


def make_augment(augment_str):
    """Generate augmentation function from separated parameter string.
    The parameter string augment_str may be either "AutoAugment" or "RandAugment".
    Undocumented use allows for specifying extra params, e.g. "RandAugment_N2_M0.5"."""
    params = augment_str.split("_")
    names = {"N": "n_ops", "M": "magnitude", "P": "prob"}
    assert params[0] in ["RandAugment", "AutoAugment"]
    assert all(p[0] in names for p in params[1:])
    keys = [names[p[0]] for p in params[1:]]
    vals = [float(p[1:]) for p in params[1:]]
    augment = rand_augment if params[0] == "RandAugment" else auto_augment
    return lambda im: augment(im, **dict(zip(keys, vals)))


def visualize_ops(im, ops=None, num_steps=10):
    """Visualize ops by applying each op by varying amounts."""
    ops = ops if ops else RANDAUG_OPS
    w, h, magnitudes = im.size[0], im.size[1], np.linspace(0, 1, num_steps)
    output = Image.new("RGB", (w * num_steps, h * len(ops)))
    for i, op in enumerate(ops):
        for j, m in enumerate(magnitudes):
            out = apply_op(im, op, prob=1.0, magnitude=m)
            output.paste(out, (j * w, i * h))
    return output


def visualize_aug(im, augment=rand_augment, num_trials=10, **kwargs):
    """Visualize augmentation by applying random augmentations."""
    w, h = im.size[0], im.size[1]
    output = Image.new("RGB", (w * num_trials, h * num_trials))
    for i in range(num_trials):
        for j in range(num_trials):
            output.paste(augment(im, **kwargs), (j * w, i * h))
    return output

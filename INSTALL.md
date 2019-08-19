# Installing PyCls

This document covers how to install Pycls and its dependencies.

- For general information about PyCls, please see [`README.md`](README.md)

**Requirements:**

- NVIDIA GPU, Linux, Python3
- PyTorch, various Python packages; Instructions for installing these dependencies are found below

**Notes:**

- PyCls currently does not support running on CPU; a GPU system is required
- PyCls has been tested with CUDA 9.2 and cuDNN 7.1

## PyTorch

To install PyTorch with CUDA support, follow the [installation instructions](https://pytorch.org/get-started/locally/) from the [PyTorch website](https://pytorch.org).

## PyCls

Clone the PyCls repository:

```
# PYCLS=/path/to/clone/pycls
git clone https://github.com/facebookresearch/pycls $PYCLS
```

Install Python dependencies:

```
pip install -r $PYCLS/requirements.txt
```

Set up Python modules:

```
cd $PYCLS && make
```

## Datasets

PyCls finds datasets via symlinks from `pycls/datasets/data` to the actual locations where the dataset images and annotations are stored. For instructions on how to create symlinks for CIFAR and ImageNet, please see [`DATA.md`](pycls/DATA.md).

## Using PyCls

Please see [`GETTING_STARTED.md`](GETTING_STARTED.md) for basic tutorials covering training and evaluation with PyCls.

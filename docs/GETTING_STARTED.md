# Getting Started

This document provides basic instructions for training and evaluation using **pycls**.

- For general information about **pycls**, please see [`README.md`](../README.md)
- For installation instructions, please see [`INSTALL.md`](INSTALL.md)

## Training Models

Training on CIFAR with 1 GPU: 

```
python tools/train_net.py \
    --cfg configs/cifar/resnet/R-56_nds_1gpu.yaml \
    OUT_DIR /tmp
```

Training on ImageNet with 1 GPU:

```
python tools/train_net.py \
    --cfg configs/imagenet/resnet/R-50-1x64d_step_1gpu.yaml \
    OUT_DIR /tmp
```

Training on ImageNet with 2 GPUs:

```
python tools/train_net.py \
    --cfg configs/imagenet/resnet/R-50-1x64d_step_2gpu.yaml \
    OUT_DIR /tmp
```

## Finetuning Models

Finetuning on ImageNet with 1 GPU:

```
python tools/train_net.py \
    --cfg configs/imagenet/resnet/R-50-1x64d_step_1gpu.yaml \
    TRAIN.WEIGHTS /path/to/weights/file \
    OUT_DIR /tmp
```

## Evaluating Models

Evaluation on ImageNet with 1 GPU:

```
python tools/test_net.py \
    --cfg configs/imagenet/resnet/R-50-1x64d_step_1gpu.yaml \
    TEST.WEIGHTS /path/to/weights/file \
    OUT_DIR /tmp
```

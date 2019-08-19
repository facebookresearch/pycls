# Using PyCls

This document provides basic instructions for training and evaluation using PyCls.

- For general information about PyCls, please see [`README.md`](README.md)
- For installation instructions, please see [`INSTALL.md`](INSTALL.md)

## Training a Model with PyCls

Training on CIFAR with 1 GPU: 

```
python tools/train_net.py \
    --cfg configs/08_2019_baselines/cifar10/R-56_bs128_1gpu.yaml \
    OUT_DIR /tmp
```

Training on ImageNet with 1 GPU:

```
python tools/train_net.py \
    --cfg configs/08_2019_baselines/imagenet/R-50-1x64d_bs32_1gpu.yaml \
    OUT_DIR /tmp
```

Training on ImageNet with 2 GPUs:

```
python tools/train_net.py \
    --cfg configs/08_2019_baselines/imagenet/R-50-1x64d_bs64_2gpu.yaml \
    OUT_DIR /tmp
```

## Finetuning Models

Finetuning on ImageNet with 1 GPU:

```
python tools/train_net.py \
    --cfg configs/08_2019_baselines/imagenet/R-50-1x64d_bs32_1gpu.yaml \
    TRAIN.WEIGHTS /path/to/weights/file \
    OUT_DIR /tmp
```

## Evaluation of Pretrained Models

Evaluation on ImageNet with 1 GPU:

```
python tools/test_net.py \
    --cfg configs/08_2019_baselines/imagenet/R-50-1x64d_bs32_1gpu.yaml \
    TEST.WEIGHTS /path/to/weights/file \
    OUT_DIR /tmp
```

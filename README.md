# [wip] OSS version of `pycls`

## Installation instructions

Clone the repo:

```
# PYCLS=/path/to/clone/pycls
git clone https://github.com/facebookresearch/pycls $PYCLS
```

Set up Python modules:

```
cd $PYCLS && make
```

Debugging cifar10 models locally using 1 GPU:

```
python tools/train_net.py \
    --cfg configs/baselines/cifar10/vgg11_bs128_1gpu.yaml \
    TRAIN.AUTO_RESUME False \
    OUT_DIR /tmp
```

Debugging IN-1k models locally using 1 GPU:

```
python tools/train_net.py \
    --cfg configs/baselines/imagenet/R-50-1x64d_bs32_1gpu.yaml \
    TRAIN.AUTO_RESUME False \
    OUT_DIR /tmp
```

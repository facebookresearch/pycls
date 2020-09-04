# Getting Started

**pycls** can be used as a [library](#library-usage) (e.g. import pretrained models) or as a [framework](#framework-usage) (e.g. modify for your needs). This document provides brief installation instructions and basic usage examples for both use cases.

**Notes:**

- **pycls** has been tested with PyTorch 1.6, CUDA 9.2 and cuDNN 7.1
- **pycls** currently does not support running on CPU; a GPU system is required

## Library Usage

Install the package:

```
pip install pycls
```

Load a pretrained model:

```
model = pycls.models.regnetx("400MF", pretrained=True)
```

Create a model with the number of classes altered:

```
model = pycls.models.regnety("4.0GF", pretrained=False, cfg_list=("MODEL.NUM_CLASSES", 100))
```

Please see the [`MODEL_ZOO.md`](../MODEL_ZOO.md) for the available pretrained models.

## Framework Usage

Clone the repository:

```
git clone https://github.com/facebookresearch/pycls
```

Install dependencies:

```
pip install -r requirements.txt
```

Set up modules:

```
python setup.py develop --user
```

Please see [`DATA.md`](DATA.md) for the instructions on setting up datasets.

### Evaluation

RegNetX-400MF on ImageNet with 8 GPUs:

```
python tools/test_net.py \
    --cfg configs/dds_baselines/regnetx/RegNetX-400MF_dds_8gpu.yaml \
    TEST.WEIGHTS https://dl.fbaipublicfiles.com/pycls/dds_baselines/160905967/RegNetX-400MF_dds_8gpu.pyth \
    OUT_DIR /tmp
```

### Training

RegNetX-400MF on ImageNet with 8 GPUs:

```
python tools/train_net.py \
    --cfg configs/dds_baselines/regnetx/RegNetX-400MF_dds_8gpu.yaml \
    OUT_DIR /tmp
```

### Finetuning

RegNetX-400MF on ImageNet with 8 GPUs:

```
python tools/train_net.py \
    --cfg configs/dds_baselines/regnetx/RegNetX-400MF_dds_8gpu.yaml \
    TRAIN.WEIGHTS https://dl.fbaipublicfiles.com/pycls/dds_baselines/160905967/RegNetX-400MF_dds_8gpu.pyth \
    OUT_DIR /tmp
```

### Timing

RegNetX-400MF with 1 GPU:

```
python tools/time_net.py
    --cfg configs/dds_baselines/regnetx/RegNetX-400MF_dds_8gpu.yaml \
    NUM_GPUS 1 \
    TRAIN.BATCH_SIZE 64 \
    TEST.BATCH_SIZE 64 \
    PREC_TIME.WARMUP_ITER 5 \
    PREC_TIME.NUM_ITER 50
```

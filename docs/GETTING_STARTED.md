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

Set all the files in ./tools to be executable by the user:

```
chmod 744 ./tools/*.py
```

Set up modules:

```
python setup.py develop --user
```

Please see [`DATA.md`](DATA.md) for the instructions on setting up datasets.

The examples below use a config for RegNetX-400MF on ImageNet with 8 GPUs.

### Model Info

```
./tools/run_net.py --mode info \
    --cfg configs/dds_baselines/regnetx/RegNetX-400MF_dds_8gpu.yaml
```

### Model Evaluation

```
./tools/run_net.py --mode test \
    --cfg configs/dds_baselines/regnetx/RegNetX-400MF_dds_8gpu.yaml \
    TEST.WEIGHTS https://dl.fbaipublicfiles.com/pycls/dds_baselines/160905967/RegNetX-400MF_dds_8gpu.pyth \
    OUT_DIR /tmp
```

### Model Evaluation (multi-node)

```
 ./tools/run_net.py --mode test \
    --cfg configs/dds_baselines/regnetx/RegNetX-400MF_dds_8gpu.yaml \
    TEST.WEIGHTS https://dl.fbaipublicfiles.com/pycls/dds_baselines/160905967/RegNetX-400MF_dds_8gpu.pyth \
    OUT_DIR test/ LOG_DEST file LAUNCH.MODE slurm LAUNCH.PARTITION devlab NUM_GPUS 16 LAUNCH.NAME pycls_eval_test
```

### Model Training

```
./tools/run_net.py --mode train \
    --cfg configs/dds_baselines/regnetx/RegNetX-400MF_dds_8gpu.yaml \
    OUT_DIR /tmp
```

### Model Finetuning

```
./tools/run_net.py --mode train \
    --cfg configs/dds_baselines/regnetx/RegNetX-400MF_dds_8gpu.yaml \
    TRAIN.WEIGHTS https://dl.fbaipublicfiles.com/pycls/dds_baselines/160905967/RegNetX-400MF_dds_8gpu.pyth \
    OUT_DIR /tmp
```

### Model Timing

```
./tools/run_net.py --mode time \
    --cfg configs/dds_baselines/regnetx/RegNetX-400MF_dds_8gpu.yaml \
    NUM_GPUS 1 \
    TRAIN.BATCH_SIZE 64 \
    TEST.BATCH_SIZE 64 \
    PREC_TIME.WARMUP_ITER 5 \
    PREC_TIME.NUM_ITER 50
```

### Model Scaling

Scale a RegNetY-4GF by 4x using fast compound scaling (see https://arxiv.org/abs/2103.06877):

```
./tools/run_net.py --mode scale \
    --cfg configs/dds_baselines/regnety/RegNetY-4.0GF_dds_8gpu.yaml \
    OUT_DIR ./ \
    CFG_DEST "RegNetY-4.0GF_dds_8gpu_scaled.yaml" \
    MODEL.SCALING_FACTOR 4.0 \
    MODEL.SCALING_TYPE "d1_w8_g8_r1"
```

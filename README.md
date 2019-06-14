# PyTorch classification code

## Local development

Go to fbcode:

```
cd ~/fbsource/fbcode
```

Build with Buck:

```
buck build @mode/dev-nosan -c 'python.native_link_strategy=separate' experimental/deeplearning/ilijar/pycls/tools/...
```

Train on cifar10 with 1 GPU:

```
buck-out/gen/experimental/deeplearning/ilijar/pycls/tools/train_net.par --cfg experimental/deeplearning/ilijar/pycls/configs/baselines/cifar10/resnet56_bs128_1gpu.yaml OUTPUT_DIR /tmp
```

Train on IN-1k with 1 GPU:

```
buck-out/gen/experimental/deeplearning/ilijar/pycls/tools/train_net.par --cfg experimental/deeplearning/ilijar/pycls/configs/baselines/in1k/R-50-1x64d_bs32_1gpu.yaml OUTPUT_DIR /tmp
```

## Running jobs on the cluster

Go to pycls:

```
cd ~/fbsource/fbcode/experimental/deeplearning/ilijar/pycls
```

Train on cifar10 with 1 GPU:

```
GPU=1 CPU=8 MEM=32 ./cluster/launch.sh configs/baselines/cifar10/resnet56_bs128_1gpu.yaml <run_name>
```

Train on IN-1k with 8 GPUs:

```
./cluster/launch.sh configs/baselines/in1k/R-50-1x64d_bs32_1gpu.yaml <run_name>
```

## Plotting the training curves

See https://fburl.com/unpnkfxv

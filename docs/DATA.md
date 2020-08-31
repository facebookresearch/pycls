# Setting Up Data Paths

**pycls** finds datasets via symlinks from `pycls/datasets/data` to the actual locations where the dataset images and labels are stored. The instructions on how to create symlinks for ImageNet and CIFAR are given below.

Expected datasets structure for ImageNet:

```
imagenet
|_ train
|  |_ n01440764
|  |_ ...
|  |_ n15075141
|_ val
|  |_ n01440764
|  |_ ...
|  |_ n15075141
|_ ...
```

Expected datasets structure for CIFAR-10:

```
cifar10
|_ data_batch_1
|_ data_batch_2
|_ data_batch_3
|_ data_batch_4
|_ data_batch_5
|_ test_batch
|_ ...
```

Create a directory containing symlinks:

```
mkdir -p /path/pycls/pycls/datasets/data
```

Symlink ImageNet:

```
ln -s /path/imagenet /path/pycls/pycls/datasets/data/imagenet
```

Symlink CIFAR-10:

```
ln -s /path/cifar10 /path/pycls/pycls/datasets/data/cifar10
```

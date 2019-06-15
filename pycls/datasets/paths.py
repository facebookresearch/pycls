#!/usr/bin/env python3

"""Dataset paths."""

# Common data path prefixes
_LOCAL = '/data/local/packages'
_GFSAI = '/mnt/vol/gfsai-bistro-east/ai-group/users/ilijar'

# Data paths
_paths = {
    'cifar10': _GFSAI + '/cifar/cifar-10-batches-py',
    'imagenet': _LOCAL + '/ai-group.imagenet-full-size/prod/imagenet_full_size'
}


def contains(dataset_name):
    """Determines if the dataset path is present."""
    return dataset_name in _paths.keys()


def get_data_path(dataset_name):
    """Retrieves data path for the dataset."""
    return _paths[dataset_name]


def set_data_path(dataset_name, data_path):
    """Sets data path for the dataset."""
    _paths[dataset_name] = data_path

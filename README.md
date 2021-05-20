# pycls

**pycls** is an image classification codebase, written in [PyTorch](https://pytorch.org/). It was originally developed for the [On Network Design Spaces for Visual Recognition](https://arxiv.org/abs/1905.13214) project. **pycls** has since matured and been adopted by a number of [projects](#projects) at Facebook AI Research.

<div align="center">
  <img src="docs/regnetx_nets.png" width="550px" />
  <p align="center"><b>pycls</b> provides a large set of baseline models across a wide range of flop regimes.</p>
</div>

## Introduction

The goal of **pycls** is to provide a simple and flexible codebase for image classification. It is designed to support rapid implementation and evaluation of research ideas. **pycls** also provides a large collection of baseline results ([Model Zoo](MODEL_ZOO.md)).  The codebase supports efficient single-machine multi-gpu training, powered by the PyTorch distributed package, and provides implementations of standard models including [ResNet](https://arxiv.org/abs/1512.03385), [ResNeXt](https://arxiv.org/abs/1611.05431), [EfficientNet](https://arxiv.org/abs/1905.11946), and [RegNet](https://arxiv.org/abs/2003.13678).

## Using pycls

Please see [`GETTING_STARTED`](docs/GETTING_STARTED.md) for brief installation instructions and basic usage examples.

## Model Zoo

We provide a large set of baseline results and pretrained models available for download in the **pycls** [Model Zoo](MODEL_ZOO.md); including the simple, fast, and effective [RegNet](https://arxiv.org/abs/2003.13678) models that we hope can serve as solid baselines across a wide range of flop regimes.

## Sweep Code

The pycls codebase now provides powerful support for studying *design spaces* and more generally *population statistics* of models as introduced in [On Network Design Spaces for Visual Recognition](https://arxiv.org/abs/1905.13214) and [Designing Network Design Spaces](https://arxiv.org/abs/2003.13678). This idea is that instead of planning a single pycls job (e.g., testing a specific model configuration), one can study the behavior of an entire population of models. This allows for quite powerful and succinct experimental design, and elevates the study of individual model behavior to the study of the behavior of model populations. Please see [`SWEEP_INFO`](docs/SWEEP_INFO.md) for details.

## Projects

A number of projects at FAIR have been built on top of **pycls**:

- [On Network Design Spaces for Visual Recognition](https://arxiv.org/abs/1905.13214)
- [Exploring Randomly Wired Neural Networks for Image Recognition](https://arxiv.org/abs/1904.01569)
- [Designing Network Design Spaces](https://arxiv.org/abs/2003.13678)
- [Fast and Accurate Model Scaling](https://arxiv.org/abs/2103.06877)
- [Are Labels Necessary for Neural Architecture Search?](https://arxiv.org/abs/2003.12056)
- [PySlowFast Video Understanding Codebase](https://github.com/facebookresearch/SlowFast)

If you are using **pycls** in your research and would like to include your project here, please let us know or send a PR.

## Citing pycls

If you find **pycls** helpful in your research or refer to the baseline results in the [Model Zoo](MODEL_ZOO.md), please consider citing an appropriate subset of the following papers:

```
@InProceedings{Radosavovic2019,
  title = {On Network Design Spaces for Visual Recognition},
  author = {Ilija Radosavovic and Justin Johnson and Saining Xie Wan-Yen Lo and Piotr Doll{\'a}r},
  booktitle = {ICCV},
  year = {2019}
}

@InProceedings{Radosavovic2020,
  title = {Designing Network Design Spaces},
  author = {Ilija Radosavovic and Raj Prateek Kosaraju and Ross Girshick and Kaiming He and Piotr Doll{\'a}r},
  booktitle = {CVPR},
  year = {2020}
}

@InProceedings{Dollar2021,
  title = {Fast and Accurate Model Scaling},
  author = {Piotr Doll{\'a}r and Mannat Singh and Ross Girshick},
  booktitle = {CVPR},
  year = {2021}
}
```

## License

**pycls** is released under the MIT license. Please see the [`LICENSE`](LICENSE) file for more information.

## Contributing

We actively welcome your pull requests! Please see [`CONTRIBUTING.md`](docs/CONTRIBUTING.md) and [`CODE_OF_CONDUCT.md`](docs/CODE_OF_CONDUCT.md) for more info.

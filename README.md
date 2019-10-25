# pycls

**pycl** is an image classification codebase, written in [PyTorch](https://pytorch.org/). The codebase was originally developed for a project that led to the [On Network Design Spaces for Visual Recognition](https://arxiv.org/abs/1905.13214) work. **pycls** has since matured into a general image classification codebase that has been adopted by a number representation learning projects at Facebook AI Research.

## Introduction

The goal of **pycls** is to provide a high-quality, high-performance codebase for image classification <i>research</i>. It is designed to be <i>simple</i> and <i>flexible</i> in order to support rapid implementation and evaluation of research ideas.

The codebase implements efficient single-machine multi-gpu training, powered by PyTorch distributed package. **pycls** includes implementations of standard baseline models ([ResNet](https://arxiv.org/abs/1512.03385), [ResNeXt](https://arxiv.org/abs/1611.05431), [EfficientNet](https://arxiv.org/abs/1905.11946)) and generic modeling functionality that can be useful for experimenting with network design. Additional models can be easily implemented.

## Installation

Please see [`INSTALL.md`](INSTALL.md) for installation instructions.

## Getting Started

After installation, please see [`GETTING_STARTED.md`](GETTING_STARTED.md) for basic instructions on training and evaluation with **pycls**.

## Model Zoo

Coming soon!

## Citing pycls

If you use **pycls** in your research, please use the following BibTex entry

```
@InProceedings{Radosavovic2019,
  title = {On Network Design Spaces for Visual Recognition},
  author = {Radosavovic, Ilija and Johnson, Justin and Xie, Saining and Lo, Wan-Yen and Doll{\'a}r, Piotr},
  booktitle = {ICCV},
  year = {2019},
}
```

## License

**pycls** is released under the MIT license. See the [LICENSE](LICENSE) file for more information.

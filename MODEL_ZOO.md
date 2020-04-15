# Model Zoo

## Introduction

This file documents a collection of baselines trained with **pycls**, primarily for the [Designing Network Design Spaces](https://arxiv.org/abs/2003.13678) paper. All configurations for these baselines are located in the `configs/dds_baselines` directory. The tables below provide results and useful statistics about training and inference. Links to the pretrained models are provided as well. The following experimental and training settings are used for all of the training and inference runs.

### Experimental Settings

- All baselines were run on [Big Basin](https://code.facebook.com/posts/1835166200089399/introducing-big-basin) servers with 8 NVIDIA Tesla V100 GPUs (16GB GPU memory).
- All baselines were run using PyTorch 1.6, CUDA 9.2, and cuDNN 7.6.
- Inference times are reported for 64 images on 1 GPU for all models.
- Training times are reported for 100 epochs on 8 GPUs with the batch size listed.
- The reported errors are averaged across 5 reruns for robust estimates.
- The provided checkpoints are from the runs with errors closest to the average.
- All models and results below are on the ImageNet-1k dataset.
- The *model id* column is provided for ease of reference.

### Training Settings

Our primary goal is to provide simple and strong baselines that are easy to reproduce. For all models, we use our basic training settings without any training enhancements (e.g., DropOut, DropConnect, AutoAugment, EMA, etc.) or testing enhancements (e.g., multi-crop, multi-scale, flipping, etc.); please see our [Designing Network Design Spaces](https://arxiv.org/abs/2003.13678) paper for more information.

- We use SGD with mometum of 0.9, a half-period cosine schedule, and train for 100 epochs.
- For ResNet/ResNeXt/RegNet, we use a *reference* learning rate of 0.1 and a weight decay of 5e-5 (see [Figure 21](https://arxiv.org/abs/2003.13678)).
- For EfficientNet, we use a *reference* learning rate of 0.2 and a weight decay of 1e-5 (see [Figure 22](https://arxiv.org/abs/2003.13678)).
- The actual learning rate for each model is computed as (batch-size / 128) * reference-lr.
- For training, we use aspect ratio, flipping, PCA, and per-channel mean and SD normalization.
- At test time, we rescale images to (256 / 224) * train-res and take the center crop of train-res.
- For ResNet/ResNeXt/RegNet, we use the image size of 224x224 for training.
- For EfficientNet, the training image size varies following the original paper.

For 8 GPU training, we apply 5 epoch gradual warmup, following the [ImageNet in 1 Hour](https://arxiv.org/abs/1706.02677) paper. Note that the learning rate scaling rule described above is similar to the one from the [ImageNet in 1 Hour](https://arxiv.org/abs/1706.02677) paper but the number of images per GPU varies among models. To understand how the configs are adjusted, please see the examples in the `configs/lr_scaling` directory.

## Baselines

### RegNetX Models

<table><tbody>
<!-- START RegNetX TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">model</th>
<th valign="bottom">flops</br>(B)</th>
<th valign="bottom">params</br>(M)</th>
<th valign="bottom">acts</br>(M)</th>
<th valign="bottom">batch</br>size</th>
<th valign="bottom">infer</br>(ms)</th>
<th valign="bottom">train</br>(hr)</th>
<th valign="bottom">error</br>(top-1)</th>
<th valign="bottom">model id</th>
<th valign="bottom">download</th>
<!-- TABLE BODY -->
<!-- ROW RegNetX-200MF -->
<tr>
<td align="left"><a href="configs/dds_baselines/regnetx/RegNetX-200MF_dds_8gpu.yaml">RegNetX-200MF</a></td>
<td align="center">0.2</td>
<td align="center">2.7</td>
<td align="center">2.2</td>
<td align="center">1024</td>
<td align="center">10</td>
<td align="center">2.8</td>
<td align="center">31.1</td>
<td align="center">000000000</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/pycls/dds_baselines/<...>/model_epoch_0100.pyth">model</a></td>
</tr>
<!-- ROW RegNetX-400MF -->
<tr>
<td align="left"><a href="configs/dds_baselines/regnetx/RegNetX-400MF_dds_8gpu.yaml">RegNetX-400MF</a></td>
<td align="center">0.4</td>
<td align="center">5.2</td>
<td align="center">3.1</td>
<td align="center">1024</td>
<td align="center">15</td>
<td align="center">3.9</td>
<td align="center">27.3</td>
<td align="center">000000000</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/pycls/dds_baselines/<...>/model_epoch_0100.pyth">model</a></td>
</tr>
<!-- ROW RegNetX-600MF -->
<tr>
<td align="left"><a href="configs/dds_baselines/regnetx/RegNetX-600MF_dds_8gpu.yaml">RegNetX-600MF</a></td>
<td align="center">0.6</td>
<td align="center">6.2</td>
<td align="center">4.0</td>
<td align="center">1024</td>
<td align="center">17</td>
<td align="center">4.4</td>
<td align="center">25.9</td>
<td align="center">000000000</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/pycls/dds_baselines/<...>/model_epoch_0100.pyth">model</a></td>
</tr>
<!-- ROW RegNetX-800MF -->
<tr>
<td align="left"><a href="configs/dds_baselines/regnetx/RegNetX-800MF_dds_8gpu.yaml">RegNetX-800MF</a></td>
<td align="center">0.8</td>
<td align="center">7.3</td>
<td align="center">5.1</td>
<td align="center">1024</td>
<td align="center">21</td>
<td align="center">5.7</td>
<td align="center">24.8</td>
<td align="center">000000000</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/pycls/dds_baselines/<...>/model_epoch_0100.pyth">model</a></td>
</tr>
<!-- ROW RegNetX-1.6GF -->
<tr>
<td align="left"><a href="configs/dds_baselines/regnetx/RegNetX-1.6GF_dds_8gpu.yaml">RegNetX-1.6GF</a></td>
<td align="center">1.6</td>
<td align="center">9.2</td>
<td align="center">7.9</td>
<td align="center">1024</td>
<td align="center">33</td>
<td align="center">8.7</td>
<td align="center">23.0</td>
<td align="center">000000000</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/pycls/dds_baselines/<...>/model_epoch_0100.pyth">model</a></td>
</tr>
<!-- ROW RegNetX-3.2GF -->
<tr>
<td align="left"><a href="configs/dds_baselines/regnetx/RegNetX-3.2GF_dds_8gpu.yaml">RegNetX-3.2GF</a></td>
<td align="center">3.2</td>
<td align="center">15.3</td>
<td align="center">11.4</td>
<td align="center">512</td>
<td align="center">57</td>
<td align="center">14.3</td>
<td align="center">21.7</td>
<td align="center">000000000</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/pycls/dds_baselines/<...>/model_epoch_0100.pyth">model</a></td>
</tr>
<!-- ROW RegNetX-4.0GF -->
<tr>
<td align="left"><a href="configs/dds_baselines/regnetx/RegNetX-4.0GF_dds_8gpu.yaml">RegNetX-4.0GF</a></td>
<td align="center">4.0</td>
<td align="center">22.1</td>
<td align="center">12.2</td>
<td align="center">512</td>
<td align="center">69</td>
<td align="center">17.1</td>
<td align="center">21.4</td>
<td align="center">000000000</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/pycls/dds_baselines/<...>/model_epoch_0100.pyth">model</a></td>
</tr>
<!-- ROW RegNetX-6.4GF -->
<tr>
<td align="left"><a href="configs/dds_baselines/regnetx/RegNetX-6.4GF_dds_8gpu.yaml">RegNetX-6.4GF</a></td>
<td align="center">6.5</td>
<td align="center">26.2</td>
<td align="center">16.4</td>
<td align="center">512</td>
<td align="center">92</td>
<td align="center">23.5</td>
<td align="center">20.8</td>
<td align="center">000000000</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/pycls/dds_baselines/<...>/model_epoch_0100.pyth">model</a></td>
</tr>
<!-- ROW RegNetX-8.0GF -->
<tr>
<td align="left"><a href="configs/dds_baselines/regnetx/RegNetX-8.0GF_dds_8gpu.yaml">RegNetX-8.0GF</a></td>
<td align="center">8.0</td>
<td align="center">39.6</td>
<td align="center">14.1</td>
<td align="center">512</td>
<td align="center">94</td>
<td align="center">22.6</td>
<td align="center">20.7</td>
<td align="center">000000000</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/pycls/dds_baselines/<...>/model_epoch_0100.pyth">model</a></td>
</tr>
<!-- ROW RegNetX-12GF -->
<tr>
<td align="left"><a href="configs/dds_baselines/regnetx/RegNetX-12GF_dds_8gpu.yaml">RegNetX-12GF</a></td>
<td align="center">12.1</td>
<td align="center">46.1</td>
<td align="center">21.4</td>
<td align="center">512</td>
<td align="center">137</td>
<td align="center">32.9</td>
<td align="center">20.3</td>
<td align="center">000000000</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/pycls/dds_baselines/<...>/model_epoch_0100.pyth">model</a></td>
</tr>
<!-- ROW RegNetX-16GF -->
<tr>
<td align="left"><a href="configs/dds_baselines/regnetx/RegNetX-16GF_dds_8gpu.yaml">RegNetX-16GF</a></td>
<td align="center">15.9</td>
<td align="center">54.3</td>
<td align="center">25.5</td>
<td align="center">512</td>
<td align="center">168</td>
<td align="center">39.7</td>
<td align="center">20.0</td>
<td align="center">000000000</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/pycls/dds_baselines/<...>/model_epoch_0100.pyth">model</a></td>
</tr>
<!-- ROW RegNetX-32GF -->
<tr>
<td align="left"><a href="configs/dds_baselines/regnetx/RegNetX-32GF_dds_8gpu.yaml">RegNetX-32GF</a></td>
<td align="center">31.7</td>
<td align="center">107.8</td>
<td align="center">36.3</td>
<td align="center">256</td>
<td align="center">318</td>
<td align="center">76.9</td>
<td align="center">19.5</td>
<td align="center">000000000</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/pycls/dds_baselines/<...>/model_epoch_0100.pyth">model</a></td>
</tr>
<!-- END RegNetX TABLE -->
</tbody></table>

### RegNetY Models

<table><tbody>
<!-- START RegNetY TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">model</th>
<th valign="bottom">flops</br>(B)</th>
<th valign="bottom">params</br>(M)</th>
<th valign="bottom">acts</br>(M)</th>
<th valign="bottom">batch</br>size</th>
<th valign="bottom">infer</br>(ms)</th>
<th valign="bottom">train</br>(hr)</th>
<th valign="bottom">error</br>(top-1)</th>
<th valign="bottom">model id</th>
<th valign="bottom">download</th>
<!-- TABLE BODY -->
<!-- ROW RegNetY-200MF -->
<tr>
<td align="left"><a href="configs/dds_baselines/regnety/RegNetY-200MF_dds_8gpu.yaml">RegNetY-200MF</a></td>
<td align="center">0.2</td>
<td align="center">3.2</td>
<td align="center">2.2</td>
<td align="center">1024</td>
<td align="center">11</td>
<td align="center">3.1</td>
<td align="center">29.6</td>
<td align="center">000000000</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/pycls/dds_baselines/<...>/model_epoch_0100.pyth">model</a></td>
</tr>
<!-- ROW RegNetY-400MF -->
<tr>
<td align="left"><a href="configs/dds_baselines/regnety/RegNetY-400MF_dds_8gpu.yaml">RegNetY-400MF</a></td>
<td align="center">0.4</td>
<td align="center">4.3</td>
<td align="center">3.9</td>
<td align="center">1024</td>
<td align="center">19</td>
<td align="center">5.1</td>
<td align="center">25.9</td>
<td align="center">000000000</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/pycls/dds_baselines/<...>/model_epoch_0100.pyth">model</a></td>
</tr>
<!-- ROW RegNetY-600MF -->
<tr>
<td align="left"><a href="configs/dds_baselines/regnety/RegNetY-600MF_dds_8gpu.yaml">RegNetY-600MF</a></td>
<td align="center">0.6</td>
<td align="center">6.1</td>
<td align="center">4.3</td>
<td align="center">1024</td>
<td align="center">19</td>
<td align="center">5.2</td>
<td align="center">24.5</td>
<td align="center">000000000</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/pycls/dds_baselines/<...>/model_epoch_0100.pyth">model</a></td>
</tr>
<!-- ROW RegNetY-800MF -->
<tr>
<td align="left"><a href="configs/dds_baselines/regnety/RegNetY-800MF_dds_8gpu.yaml">RegNetY-800MF</a></td>
<td align="center">0.8</td>
<td align="center">6.3</td>
<td align="center">5.2</td>
<td align="center">1024</td>
<td align="center">22</td>
<td align="center">6.0</td>
<td align="center">23.7</td>
<td align="center">000000000</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/pycls/dds_baselines/<...>/model_epoch_0100.pyth">model</a></td>
</tr>
<!-- ROW RegNetY-1.6GF -->
<tr>
<td align="left"><a href="configs/dds_baselines/regnety/RegNetY-1.6GF_dds_8gpu.yaml">RegNetY-1.6GF</a></td>
<td align="center">1.6</td>
<td align="center">11.2</td>
<td align="center">8.0</td>
<td align="center">1024</td>
<td align="center">39</td>
<td align="center">10.1</td>
<td align="center">22.0</td>
<td align="center">000000000</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/pycls/dds_baselines/<...>/model_epoch_0100.pyth">model</a></td>
</tr>
<!-- ROW RegNetY-3.2GF -->
<tr>
<td align="left"><a href="configs/dds_baselines/regnety/RegNetY-3.2GF_dds_8gpu.yaml">RegNetY-3.2GF</a></td>
<td align="center">3.2</td>
<td align="center">19.4</td>
<td align="center">11.3</td>
<td align="center">512</td>
<td align="center">67</td>
<td align="center">16.5</td>
<td align="center">21.0</td>
<td align="center">000000000</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/pycls/dds_baselines/<...>/model_epoch_0100.pyth">model</a></td>
</tr>
<!-- ROW RegNetY-4.0GF -->
<tr>
<td align="left"><a href="configs/dds_baselines/regnety/RegNetY-4.0GF_dds_8gpu.yaml">RegNetY-4.0GF</a></td>
<td align="center">4.0</td>
<td align="center">20.6</td>
<td align="center">12.3</td>
<td align="center">512</td>
<td align="center">68</td>
<td align="center">16.8</td>
<td align="center">20.6</td>
<td align="center">000000000</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/pycls/dds_baselines/<...>/model_epoch_0100.pyth">model</a></td>
</tr>
<!-- ROW RegNetY-6.4GF -->
<tr>
<td align="left"><a href="configs/dds_baselines/regnety/RegNetY-6.4GF_dds_8gpu.yaml">RegNetY-6.4GF</a></td>
<td align="center">6.4</td>
<td align="center">30.6</td>
<td align="center">16.4</td>
<td align="center">512</td>
<td align="center">104</td>
<td align="center">26.1</td>
<td align="center">20.1</td>
<td align="center">000000000</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/pycls/dds_baselines/<...>/model_epoch_0100.pyth">model</a></td>
</tr>
<!-- ROW RegNetY-8.0GF -->
<tr>
<td align="left"><a href="configs/dds_baselines/regnety/RegNetY-8.0GF_dds_8gpu.yaml">RegNetY-8.0GF</a></td>
<td align="center">8.0</td>
<td align="center">39.2</td>
<td align="center">18.0</td>
<td align="center">512</td>
<td align="center">113</td>
<td align="center">28.1</td>
<td align="center">20.1</td>
<td align="center">000000000</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/pycls/dds_baselines/<...>/model_epoch_0100.pyth">model</a></td>
</tr>
<!-- ROW RegNetY-12GF -->
<tr>
<td align="left"><a href="configs/dds_baselines/regnety/RegNetY-12GF_dds_8gpu.yaml">RegNetY-12GF</a></td>
<td align="center">12.1</td>
<td align="center">51.8</td>
<td align="center">21.4</td>
<td align="center">512</td>
<td align="center">150</td>
<td align="center">36.0</td>
<td align="center">19.7</td>
<td align="center">000000000</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/pycls/dds_baselines/<...>/model_epoch_0100.pyth">model</a></td>
</tr>
<!-- ROW RegNetY-16GF -->
<tr>
<td align="left"><a href="configs/dds_baselines/regnety/RegNetY-16GF_dds_8gpu.yaml">RegNetY-16GF</a></td>
<td align="center">15.9</td>
<td align="center">83.6</td>
<td align="center">23.0</td>
<td align="center">512</td>
<td align="center">189</td>
<td align="center">45.6</td>
<td align="center">19.6</td>
<td align="center">000000000</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/pycls/dds_baselines/<...>/model_epoch_0100.pyth">model</a></td>
</tr>
<!-- ROW RegNetY-32GF -->
<tr>
<td align="left"><a href="configs/dds_baselines/regnety/RegNetY-32GF_dds_8gpu.yaml">RegNetY-32GF</a></td>
<td align="center">32.3</td>
<td align="center">145.0</td>
<td align="center">30.3</td>
<td align="center">256</td>
<td align="center">319</td>
<td align="center">76.0</td>
<td align="center">19.0</td>
<td align="center">000000000</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/pycls/dds_baselines/<...>/model_epoch_0100.pyth">model</a></td>
</tr>
<!-- END RegNetY TABLE -->
</tbody></table>

### ResNet Models

<table><tbody>
<!-- START ResNet TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">model</th>
<th valign="bottom">flops</br>(B)</th>
<th valign="bottom">params</br>(M)</th>
<th valign="bottom">acts</br>(M)</th>
<th valign="bottom">batch</br>size</th>
<th valign="bottom">infer</br>(ms)</th>
<th valign="bottom">train</br>(hr)</th>
<th valign="bottom">error</br>(top-1)</th>
<th valign="bottom">model id</th>
<th valign="bottom">download</th>
<!-- TABLE BODY -->
<!-- ROW ResNet-50 -->
<tr>
<td align="left"><a href="configs/dds_baselines/resnet/R-50-1x64d_dds_8gpu.yaml">ResNet-50</a></td>
<td align="center">4.1</td>
<td align="center">22.6</td>
<td align="center">11.1</td>
<td align="center">256</td>
<td align="center">53</td>
<td align="center">12.2</td>
<td align="center">23.2</td>
<td align="center">000000000</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/pycls/dds_baselines/<...>/model_epoch_0100.pyth">model</a></td>
</tr>
<!-- ROW ResNet-101 -->
<tr>
<td align="left"><a href="configs/dds_baselines/resnet/R-101-1x64d_dds_8gpu.yaml">ResNet-101</a></td>
<td align="center">7.8</td>
<td align="center">44.6</td>
<td align="center">16.2</td>
<td align="center">256</td>
<td align="center">90</td>
<td align="center">20.4</td>
<td align="center">21.4</td>
<td align="center">000000000</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/pycls/dds_baselines/<...>/model_epoch_0100.pyth">model</a></td>
</tr>
<!-- ROW ResNet-152 -->
<tr>
<td align="left"><a href="configs/dds_baselines/resnet/R-152-1x64d_dds_8gpu.yaml">ResNet-152</a></td>
<td align="center">11.5</td>
<td align="center">60.2</td>
<td align="center">22.6</td>
<td align="center">256</td>
<td align="center">130</td>
<td align="center">29.2</td>
<td align="center">20.9</td>
<td align="center">000000000</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/pycls/dds_baselines/<...>/model_epoch_0100.pyth">model</a></td>
</tr>
<!-- END ResNet TABLE -->
</tbody></table>

### ResNeXt Models

<table><tbody>
<!-- START ResNeXt TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">model</th>
<th valign="bottom">flops</br>(B)</th>
<th valign="bottom">params</br>(M)</th>
<th valign="bottom">acts</br>(M)</th>
<th valign="bottom">batch</br>size</th>
<th valign="bottom">infer</br>(ms)</th>
<th valign="bottom">train</br>(hr)</th>
<th valign="bottom">error</br>(top-1)</th>
<th valign="bottom">model id</th>
<th valign="bottom">download</th>
<!-- TABLE BODY -->
<!-- ROW ResNeXt-50 -->
<tr>
<td align="left"><a href="configs/dds_baselines/resnext/X-50-32x4d_dds_8gpu.yaml">ResNeXt-50</a></td>
<td align="center">4.2</td>
<td align="center">25.0</td>
<td align="center">14.4</td>
<td align="center">256</td>
<td align="center">78</td>
<td align="center">18.0</td>
<td align="center">21.9</td>
<td align="center">000000000</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/pycls/dds_baselines/<...>/model_epoch_0100.pyth">model</a></td>
</tr>
<!-- ROW ResNeXt-101 -->
<tr>
<td align="left"><a href="configs/dds_baselines/resnext/X-101-32x4d_dds_8gpu.yaml">ResNeXt-101</a></td>
<td align="center">8.0</td>
<td align="center">44.2</td>
<td align="center">21.2</td>
<td align="center">256</td>
<td align="center">137</td>
<td align="center">31.8</td>
<td align="center">20.7</td>
<td align="center">000000000</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/pycls/dds_baselines/<...>/model_epoch_0100.pyth">model</a></td>
</tr>
<!-- ROW ResNeXt-152 -->
<tr>
<td align="left"><a href="configs/dds_baselines/resnext/X-152-32x4d_dds_8gpu.yaml">ResNeXt-152</a></td>
<td align="center">11.7</td>
<td align="center">60.0</td>
<td align="center">29.7</td>
<td align="center">256</td>
<td align="center">197</td>
<td align="center">45.7</td>
<td align="center">20.4</td>
<td align="center">000000000</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/pycls/dds_baselines/<...>/model_epoch_0100.pyth">model</a></td>
</tr>
<!-- END ResNeXt TABLE -->
</tbody></table>

### EfficientNet Models

<table><tbody>
<!-- START EfficientNet TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">model</th>
<th valign="bottom">flops</br>(B)</th>
<th valign="bottom">params</br>(M)</th>
<th valign="bottom">acts</br>(M)</th>
<th valign="bottom">batch</br>size</th>
<th valign="bottom">infer</br>(ms)</th>
<th valign="bottom">train</br>(hr)</th>
<th valign="bottom">error</br>(top-1)</th>
<th valign="bottom">model id</th>
<th valign="bottom">download</th>
<!-- TABLE BODY -->
<!-- ROW EfficientNet-B0 -->
<tr>
<td align="left"><a href="configs/dds_baselines/effnet/EN-B0_dds_8gpu.yaml">EfficientNet-B0</a></td>
<td align="center">0.4</td>
<td align="center">5.3</td>
<td align="center">6.7</td>
<td align="center">256</td>
<td align="center">34</td>
<td align="center">11.7</td>
<td align="center">24.9</td>
<td align="center">000000000</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/pycls/dds_baselines/<...>/model_epoch_0100.pyth">model</a></td>
</tr>
<!-- ROW EfficientNet-B1 -->
<tr>
<td align="left"><a href="configs/dds_baselines/effnet/EN-B1_dds_8gpu.yaml">EfficientNet-B1</a></td>
<td align="center">0.7</td>
<td align="center">7.8</td>
<td align="center">10.9</td>
<td align="center">256</td>
<td align="center">52</td>
<td align="center">15.6</td>
<td align="center">24.1</td>
<td align="center">000000000</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/pycls/dds_baselines/<...>/model_epoch_0100.pyth">model</a></td>
</tr>
<!-- ROW EfficientNet-B2 -->
<tr>
<td align="left"><a href="configs/dds_baselines/effnet/EN-B2_dds_8gpu.yaml">EfficientNet-B2</a></td>
<td align="center">1.0</td>
<td align="center">9.2</td>
<td align="center">13.8</td>
<td align="center">256</td>
<td align="center">68</td>
<td align="center">18.4</td>
<td align="center">23.4</td>
<td align="center">000000000</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/pycls/dds_baselines/<...>/model_epoch_0100.pyth">model</a></td>
</tr>
<!-- ROW EfficientNet-B3 -->
<tr>
<td align="left"><a href="configs/dds_baselines/effnet/EN-B3_dds_8gpu.yaml">EfficientNet-B3</a></td>
<td align="center">1.8</td>
<td align="center">12.0</td>
<td align="center">23.8</td>
<td align="center">256</td>
<td align="center">114</td>
<td align="center">32.1</td>
<td align="center">22.5</td>
<td align="center">000000000</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/pycls/dds_baselines/<...>/model_epoch_0100.pyth">model</a></td>
</tr>
<!-- ROW EfficientNet-B4 -->
<tr>
<td align="left"><a href="configs/dds_baselines/effnet/EN-B4_dds_8gpu.yaml">EfficientNet-B4</a></td>
<td align="center">4.2</td>
<td align="center">19.0</td>
<td align="center">48.5</td>
<td align="center">128</td>
<td align="center">240</td>
<td align="center">65.1</td>
<td align="center">21.2</td>
<td align="center">000000000</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/pycls/dds_baselines/<...>/model_epoch_0100.pyth">model</a></td>
</tr>
<!-- ROW EfficientNet-B5 -->
<tr>
<td align="left"><a href="configs/dds_baselines/effnet/EN-B5_dds_8gpu.yaml">EfficientNet-B5</a></td>
<td align="center">9.9</td>
<td align="center">30.0</td>
<td align="center">98.9</td>
<td align="center">64</td>
<td align="center">504</td>
<td align="center">135.1</td>
<td align="center">21.5</td>
<td align="center">000000000</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/pycls/dds_baselines/<...>/model_epoch_0100.pyth">model</a></td>
</tr>
<!-- END EfficientNet TABLE -->
</tbody></table>

# TF-Vision Model Garden

## Introduction

TF-Vision modeling library for computer vision provides a collection of
baselines and checkpoints for image classification, object detection, and
segmentation.

## Image Classification

### ImageNet Baselines

#### ResNet models trained with vanilla settings

* Models are trained from scratch with batch size 4096 and 1.6 initial learning
  rate.
* Linear warmup is applied for the first 5 epochs.
* Models trained with l2 weight regularization and ReLU activation.

| Model        | Resolution    | Epochs  |  Top-1  |  Top-5  | Download |
| ------------ |:-------------:|--------:|--------:|--------:|---------:|
| ResNet-50    | 224x224       |    90    | 76.1 | 92.9 | [config](https://github.com/tensorflow/models/blob/master/official/vision/beta/configs/experiments/image_classification/imagenet_resnet50_tpu.yaml) |
| ResNet-50    | 224x224       |    200   | 77.1 | 93.5 | [config](https://github.com/tensorflow/models/blob/master/official/vision/beta/configs/experiments/image_classification/imagenet_resnet50_tpu.yaml) |
| ResNet-101   | 224x224       |    200   | 78.3 | 94.2 | [config](https://github.com/tensorflow/models/blob/master/official/vision/beta/configs/experiments/image_classification/imagenet_resnet101_tpu.yaml) |
| ResNet-152   | 224x224       |    200   | 78.7 | 94.3 | [config](https://github.com/tensorflow/models/blob/master/official/vision/beta/configs/experiments/image_classification/imagenet_resnet152_tpu.yaml) |

#### ResNet-RS models trained with various settings

We support state-of-the-art [ResNet-RS](https://arxiv.org/abs/2103.07579) image
classification models with features:

* ResNet-RS architectural changes and Swish activation. (Note that ResNet-RS
  adopts ReLU activation in the paper.)
* Regularization methods including Random Augment, 4e-5 weight decay, stochastic
depth, label smoothing and dropout.
* New training methods including a 350-epoch schedule, cosine learning rate and
  EMA.
* Configs are in this [directory](https://github.com/tensorflow/models/blob/master/official/vision/beta/configs/experiments/image_classification).

| Model     | Resolution | Params (M) | Top-1 | Top-5 | Download |
| --------- | :--------: | ---------: | ----: | ----: | --------:|
| ResNet-RS-50 | 160x160    | 35.7    | 79.1  | 94.5  | [config](https://github.com/tensorflow/models/blob/master/official/vision/beta/configs/experiments/image_classification/imagenet_resnetrs50_i160.yaml) \| [ckpt](https://storage.cloud.google.com/tf_model_garden/vision/resnet-rs/resnet-rs-50-i160.tar.gz) |
| ResNet-RS-101 | 160x160    | 63.7    | 80.2  | 94.9  | [config](https://github.com/tensorflow/models/blob/master/official/vision/beta/configs/experiments/image_classification/imagenet_resnetrs101_i160.yaml) \| [ckpt](https://storage.cloud.google.com/tf_model_garden/vision/resnet-rs/resnet-rs-101-i160.tar.gz) |
| ResNet-RS-101 | 192x192    | 63.7    | 81.3  | 95.6  | [config](https://github.com/tensorflow/models/blob/master/official/vision/beta/configs/experiments/image_classification/imagenet_resnetrs101_i192.yaml) \| [ckpt](https://storage.cloud.google.com/tf_model_garden/vision/resnet-rs/resnet-rs-101-i192.tar.gz) |
| ResNet-RS-152 | 192x192    | 86.8    | 81.9  | 95.8  | [config](https://github.com/tensorflow/models/blob/master/official/vision/beta/configs/experiments/image_classification/imagenet_resnetrs152_i192.yaml) \| [ckpt](https://storage.cloud.google.com/tf_model_garden/vision/resnet-rs/resnet-rs-152-i192.tar.gz) |
| ResNet-RS-152 | 224x224    | 86.8    | 82.5  | 96.1  | [config](https://github.com/tensorflow/models/blob/master/official/vision/beta/configs/experiments/image_classification/imagenet_resnetrs152_i224.yaml) \| [ckpt](https://storage.cloud.google.com/tf_model_garden/vision/resnet-rs/resnet-rs-152-i224.tar.gz) |
| ResNet-RS-152 | 256x256    | 86.8    | 83.1  | 96.3  | [config](https://github.com/tensorflow/models/blob/master/official/vision/beta/configs/experiments/image_classification/imagenet_resnetrs152_i256.yaml) \| [ckpt](https://storage.cloud.google.com/tf_model_garden/vision/resnet-rs/resnet-rs-152-i256.tar.gz) |
| ResNet-RS-200 | 256x256    | 93.4    | 83.5  | 96.6  | [config](https://github.com/tensorflow/models/blob/master/official/vision/beta/configs/experiments/image_classification/imagenet_resnetrs200_i256.yaml) \| [ckpt](https://storage.cloud.google.com/tf_model_garden/vision/resnet-rs/resnet-rs-200-i256.tar.gz) |
| ResNet-RS-270 | 256x256    | 130.1    | 83.6  | 96.6  | [config](https://github.com/tensorflow/models/blob/master/official/vision/beta/configs/experiments/image_classification/imagenet_resnetrs270_i256.yaml) \| [ckpt](https://storage.cloud.google.com/tf_model_garden/vision/resnet-rs/resnet-rs-270-i256.tar.gz) |
| ResNet-RS-350 | 256x256    |  164.3   | 83.7  | 96.7  | [config](https://github.com/tensorflow/models/blob/master/official/vision/beta/configs/experiments/image_classification/imagenet_resnetrs350_i256.yaml) \| [ckpt](https://storage.cloud.google.com/tf_model_garden/vision/resnet-rs/resnet-rs-350-i256.tar.gz) |
| ResNet-RS-350 | 320x320    | 164.3   | 84.2  | 96.9  | [config](https://github.com/tensorflow/models/blob/master/official/vision/beta/configs/experiments/image_classification/imagenet_resnetrs420_i256.yaml) \| [ckpt](https://storage.cloud.google.com/tf_model_garden/vision/resnet-rs/resnet-rs-350-i320.tar.gz) |

## Object Detection and Instance Segmentation

### Common Settings and Notes

* We provide models adopting [ResNet-FPN](https://arxiv.org/abs/1612.03144) and
  [SpineNet](https://arxiv.org/abs/1912.05027) backbones  based on detection frameworks:
  * [RetinaNet](https://arxiv.org/abs/1708.02002) and [RetinaNet-RS](https://arxiv.org/abs/2107.00057)
  * [Mask R-CNN](https://arxiv.org/abs/1703.06870)
  * [Cascade RCNN](https://arxiv.org/abs/1712.00726) and [Cascade RCNN-RS](https://arxiv.org/abs/2107.00057)

* Models are all trained on COCO train2017 and evaluated on COCO val2017.
* Training details:
  * Models finetuned from ImageNet pretrained checkpoints adopt the 12 or 36
    epochs schedule. Models trained from scratch adopt the 350 epochs schedule.
  * The default training data augmentation implements horizontal flipping and
    scale jittering with a random scale between [0.5, 2.0].
  * Unless noted, all models are trained with l2 weight regularization and ReLU
    activation.
  * We use batch size 256 and stepwise learning rate that decays at the last 30
    and 10 epoch.
  * We use square image as input by resizing the long side of an image to the
    target size then padding the short side with zeros.

### COCO Object Detection Baselines

#### RetinaNet (ImageNet pretrained)

| Backbone     | Resolution    | Epochs  | FLOPs (B)     | Params (M) | Box AP | Download |
| ------------ |:-------------:| -------:|--------------:|-----------:|-------:|---------:|
| R50-FPN      | 640x640       |    12   | 97.0 | 34.0 | 34.3 | config|
| R50-FPN      | 640x640       |    72   | 97.0 | 34.0 | 36.8 | config \| [ckpt](https://storage.cloud.google.com/tf_model_garden/vision/retinanet/retinanet-resnet50fpn.tar.gz) |

#### RetinaNet (Trained from scratch) with training features including:

* Stochastic depth with drop rate 0.2.
* Swish activation.

| Backbone     | Resolution    | Epochs  | FLOPs (B)     | Params (M) |  Box AP | Download |
| ------------ |:-------------:| -------:|--------------:|-----------:|--------:|---------:|
| SpineNet-49  | 640x640       |    500    | 85.4| 28.5 | 44.2 | [config](https://github.com/tensorflow/models/blob/master/official/vision/beta/configs/experiments/retinanet/coco_spinenet49_tpu.yaml) \| [TB.dev](https://tensorboard.dev/experiment/n2UN83TkTdyKZn3slCWulg/#scalars&_smoothingWeight=0)|
| SpineNet-96  | 1024x1024     |    500    | 265.4 | 43.0 | 48.5 |  [config](https://github.com/tensorflow/models/blob/master/official/vision/beta/configs/experiments/retinanet/coco_spinenet96_tpu.yaml) \| [TB.dev](https://tensorboard.dev/experiment/n2UN83TkTdyKZn3slCWulg/#scalars&_smoothingWeight=0)|
| SpineNet-143 | 1280x1280     |    500    | 524.0 | 67.0 | 50.0 | [config](https://github.com/tensorflow/models/blob/master/official/vision/beta/configs/experiments/retinanet/coco_spinenet143_tpu.yaml) \| [TB.dev](https://tensorboard.dev/experiment/n2UN83TkTdyKZn3slCWulg/#scalars&_smoothingWeight=0)|

#### Mobile-size RetinaNet (Trained from scratch):

| Backbone    | Resolution | Epochs | FLOPs (B) | Params (M) | Box AP | Download |
| ----------- | :--------: | -----: | --------: | ---------: | -----: | --------:|
| MobileNetv2 | 256x256    | 600    | -         | 2.27       | 23.5   | [config](https://github.com/tensorflow/models/blob/master/official/vision/beta/configs/experiments/retinanet/coco_mobilenetv2_tpu.yaml) |
| Mobile SpineNet-49  | 384x384    | 600    | 1.0      | 2.32       | 28.1   | [config](https://github.com/tensorflow/models/blob/master/official/vision/beta/configs/experiments/retinanet/coco_spinenet49_mobile_tpu.yaml) \| [ckpt](https://storage.cloud.google.com/tf_model_garden/vision/retinanet/spinenet49mobile.tar.gz) |

### Instance Segmentation Baselines

#### Mask R-CNN (Trained from scratch)

| Backbone     | Resolution    | Epochs  | FLOPs (B)  | Params (M) | Box AP | Mask AP | Download |
| ------------ |:-------------:| -------:|-----------:|-----------:|-------:|--------:|---------:|
ResNet50-FPN | 640x640    | 350    | 227.7     | 46.3       | 42.3   | 37.6    | [config](https://github.com/tensorflow/models/blob/master/official/vision/beta/configs/experiments/maskrcnn/r50fpn_640_coco_scratch_tpu4x4.yaml) |
| SpineNet-49  | 640x640       |  350    | 215.7      | 40.8       | 42.6   | 37.9    | [config](https://github.com/tensorflow/models/blob/master/official/vision/beta/configs/experiments/maskrcnn/coco_spinenet49_mrcnn_tpu.yaml) |
SpineNet-96  | 1024x1024  | 500    | 315.0     | 55.2       | 48.1   | 42.4    | [config](https://github.com/tensorflow/models/blob/master/official/vision/beta/configs/experiments/maskrcnn/coco_spinenet96_mrcnn_tpu.yaml) |
SpineNet-143 | 1280x1280  | 500    | 498.8     | 79.2       | 49.3   | 43.4    | [config](https://github.com/tensorflow/models/blob/master/official/vision/beta/configs/experiments/maskrcnn/coco_spinenet143_mrcnn_tpu.yaml) |


#### Cascade RCNN-RS (Trained from scratch)

backbone     | resolution | epochs | params (M) | box AP | mask AP | download
------------ | :--------: | -----: | ---------: | -----: | ------: | -------:
SpineNet-49  | 640x640    | 500    | 56.4       | 46.4   | 40.0    | [config](https://github.com/tensorflow/models/blob/master/official/vision/beta/configs/experiments/maskrcnn/coco_spinenet49_cascadercnn_tpu.yaml)|
SpineNet-143 | 1280x1280  | 500    | 94.9       | 51.9   | 45.0    | [config](https://github.com/tensorflow/models/blob/master/official/vision/beta/configs/experiments/maskrcnn/coco_spinenet143_cascadercnn_tpu.yaml)|

## Semantic Segmentation

* We support [DeepLabV3](https://arxiv.org/pdf/1706.05587.pdf) and
  [DeepLabV3+](https://arxiv.org/pdf/1802.02611.pdf) architectures, with
  Dilated ResNet backbones.
* Backbones are pre-trained on ImageNet.

### PASCAL-VOC

| Model      | Backbone           | Resolution | Steps | mIoU | Download |
| ---------- | :----------------: | :--------: | ----: | ---: | --------:|
| DeepLabV3  | Dilated Resnet-101 | 512x512    | 30k   | 78.7 |          |
| DeepLabV3+ | Dilated Resnet-101 | 512x512    | 30k   | 79.2 |          |

### CITYSCAPES

| Model      | Backbone           | Resolution | Steps | mIoU  | Download |
| ---------- | :----------------: | :--------: | ----: | ----: | --------:|
| DeepLabV3+ | Dilated Resnet-101 | 1024x2048  | 90k   | 78.79 |          |

## Video Classification

### Common Settings and Notes

* We provide models for video classification with two backbones:
  [SlowOnly](https://arxiv.org/abs/1812.03982) and 3D-ResNet (R3D) used in
  [Spatiotemporal Contrastive Video Representation Learning](https://arxiv.org/abs/2008.03800).
* Training and evaluation details:
  * All models are trained from scratch with vision modality (RGB) for 200
    epochs.
  * We use batch size of 1024 and cosine learning rate decay with linear warmup
    in first 5 epochs.
  * We follow [SlowFast](https://arxiv.org/abs/1812.03982) to perform 30-view
    evaluation.

### Kinetics-400 Action Recognition Baselines

| Model    | Input (frame x stride) |  Top-1  |  Top-5  | Download |
| -------- |:----------------------:|--------:|--------:|---------:|
| SlowOnly | 8 x 8                  |  74.1   |  91.4   | [config](https://github.com/tensorflow/models/blob/master/official/vision/beta/configs/experiments/video_classification/k400_slowonly8x8_tpu.yaml) |
| SlowOnly | 16 x 4                 |  75.6   |  92.1   | [config](https://github.com/tensorflow/models/blob/master/official/vision/beta/configs/experiments/video_classification/k400_slowonly16x4_tpu.yaml) |
| R3D-50   | 32 x 2                 |  77.0   |  93.0   | [config](https://github.com/tensorflow/models/blob/master/official/vision/beta/configs/experiments/video_classification/k400_3d-resnet50_tpu.yaml) |

### Kinetics-600 Action Recognition Baselines

| Model    | Input (frame x stride) |  Top-1  |  Top-5  | Download |
| -------- |:----------------------:|--------:|--------:|---------:|
| SlowOnly | 8 x 8                  |  77.3   |  93.6   | [config](https://github.com/tensorflow/models/blob/master/official/vision/beta/configs/experiments/video_classification/k600_slowonly8x8_tpu.yaml) |
| R3D-50   | 32 x 2                 |  79.5   |  94.8   | [config](https://github.com/tensorflow/models/blob/master/official/vision/beta/configs/experiments/video_classification/k600_3d-resnet50_tpu.yaml) |

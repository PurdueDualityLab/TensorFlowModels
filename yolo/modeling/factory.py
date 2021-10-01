# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Contains common factory functions yolo neural networks."""
import tensorflow as tf
from typing import Mapping, Union

from absl import logging
from official.vision.beta.modeling.backbones import factory as backbone_factory
from official.vision.beta.modeling.decoders import factory as decoder_factory

from yolo.configs import yolo
from yolo.modeling import yolo_model
from yolo.modeling.heads import yolo_head
from yolo.modeling.layers import detection_generator

def build_yolo_detection_generator(model_config: yolo.Yolo, anchor_boxes):
  model = detection_generator.YoloLayer(
      classes=model_config.num_classes,
      anchors=anchor_boxes,
      iou_thresh=model_config.detection_generator.iou_thresh,
      nms_thresh=model_config.detection_generator.nms_thresh,
      max_boxes=model_config.detection_generator.max_boxes,
      pre_nms_points=model_config.detection_generator.pre_nms_points,
      nms_type=model_config.detection_generator.nms_type,
      box_type=model_config.detection_generator.box_type.get(),
      path_scale=model_config.detection_generator.path_scales.get(),
      scale_xy=model_config.detection_generator.scale_xy.get(),
      label_smoothing=model_config.loss.label_smoothing,
      use_scaled_loss=model_config.loss.use_scaled_loss,
      update_on_repeat=model_config.loss.update_on_repeat,
      truth_thresh=model_config.loss.truth_thresh.get(),
      loss_type=model_config.loss.box_loss_type.get(),
      max_delta=model_config.loss.max_delta.get(),
      iou_normalizer=model_config.loss.iou_normalizer.get(),
      cls_normalizer=model_config.loss.cls_normalizer.get(),
      obj_normalizer=model_config.loss.obj_normalizer.get(),
      ignore_thresh=model_config.loss.ignore_thresh.get(),
      objectness_smooth=model_config.loss.objectness_smooth.get())
  return model


def build_yolo_head(input_specs, model_config: yolo.Yolo, l2_regularization):
  min_level = min(map(int, input_specs.keys()))
  max_level = max(map(int, input_specs.keys()))
  head = yolo_head.YoloHead(
      min_level=min_level,
      max_level=max_level,
      classes=model_config.num_classes,
      boxes_per_level=model_config.anchor_boxes.anchors_per_scale,
      norm_momentum=model_config.norm_activation.norm_momentum,
      norm_epsilon=model_config.norm_activation.norm_epsilon,
      kernel_regularizer=l2_regularization,
      smart_bias=model_config.head.smart_bias)
  return head


def build_yolo(input_specs, model_config, l2_regularization):
  backbone = model_config.backbone.get()
  anchor_dict, anchor_free = model_config.anchor_boxes.get(backbone.min_level,
                                                           backbone.max_level)
  backbone = backbone_factory.build_backbone(input_specs, 
                                    model_config.backbone,
                                    model_config.norm_activation,
                                    l2_regularization)
  decoder = decoder_factory.build_decoder(backbone.output_specs, 
                                          model_config,
                                          l2_regularization)

  head = build_yolo_head(decoder.output_specs, model_config, l2_regularization)
  detection_generator = build_yolo_detection_generator(model_config,anchor_dict)

  model = yolo_model.Yolo(
      backbone=backbone,
      decoder=decoder,
      head=head,
      detection_generator=detection_generator)
  model.build(input_specs.shape)

  model.summary(print_fn=logging.info)
  if anchor_free is not None:
    logging.info(f"Anchor Boxes: None -> Model is operating anchor-free.")
    logging.info(" --> anchors_per_scale set to 1. ")
  else:
    logging.info(f"Anchor Boxes: {anchor_dict}")

  losses = detection_generator.get_losses()
  return model, losses

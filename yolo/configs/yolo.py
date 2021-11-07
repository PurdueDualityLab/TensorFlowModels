# Lint as: python3
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""YOLO configuration definition."""
from typing import List, Optional, Union

from official.core import exp_factory
from official.modeling import hyperparams
from official.modeling.hyperparams import config_definitions as cfg
from official.vision.beta.configs import common
from yolo import optimization

from yolo.configs import backbones
from yolo.configs import decoders
import numpy as np
import dataclasses

MIN_LEVEL = 1
MAX_LEVEL = 7

def _build_dict(min_level, max_level, value):
  vals = {str(key): value for key in range(min_level, max_level + 1)}
  vals["all"] = None
  return lambda: vals

def _build_path_scales(min_level, max_level):
  return lambda: {str(key): 2**key for key in range(min_level, max_level + 1)}

@dataclasses.dataclass
class FPNConfig(hyperparams.Config):
  def get(self):
    values = self.as_dict()
    if "all" in values and values["all"] is not None:
      for key in values:
        if key != 'all':
          values[key] = values["all"]
    return values

# pylint: disable=missing-class-docstring
@dataclasses.dataclass
class TfExampleDecoder(hyperparams.Config):
  regenerate_source_id: bool = False
  coco91_to_80: bool = True 

@dataclasses.dataclass
class TfExampleDecoderLabelMap(hyperparams.Config):
  regenerate_source_id: bool = False
  label_map: str = ''

@dataclasses.dataclass
class DataDecoder(hyperparams.OneOfConfig):
  type: Optional[str] = 'simple_decoder'
  simple_decoder: TfExampleDecoder = TfExampleDecoder()
  label_map_decoder: TfExampleDecoderLabelMap = TfExampleDecoderLabelMap()

@dataclasses.dataclass
class Mosaic(hyperparams.Config):
  mosaic_frequency: float = 0.0
  mixup_frequency: float = 0.0
  mosaic_center: float = 0.2
  mosaic_crop_mode: Optional[str] = None
  aug_scale_min: float = 1.0
  aug_scale_max: float = 1.0
  jitter: float = 0.0
  random_pad: float = False

@dataclasses.dataclass
class Parser(hyperparams.Config):
  max_num_instances: int = 200
  letter_box: Optional[bool] = True
  random_flip: bool = True
  random_pad: float = False
  jitter: float = 0.0
  aug_scale_min: float = 1.0
  aug_scale_max: float = 1.0
  aug_rand_saturation: float = 0.0
  aug_rand_brightness: float = 0.0
  aug_rand_hue: float = 0.0
  aug_rand_angle: float = 0.0
  aug_rand_translate: float = 0.0
  aug_rand_perspective: float = 0.0
  use_tie_breaker: bool = True
  best_match_only: bool = False
  anchor_thresh: float = -0.01
  area_thresh: float = 0.1
  mosaic: Mosaic = Mosaic()

@dataclasses.dataclass
class DataConfig(cfg.DataConfig):
  """Input config for training."""
  global_batch_size: int = 64
  input_path: str = ''
  tfds_name: str = None
  tfds_split: str = None
  global_batch_size: int = 1
  is_training: bool = True
  dtype: str = 'float16'
  decoder: DataDecoder = DataDecoder()
  parser: Parser = Parser()
  shuffle_buffer_size: int = 10000
  tfds_download: bool = True
  cache: bool = False

@dataclasses.dataclass
class YoloHead(hyperparams.Config):
  """Parameterization for the YOLO Head."""
  smart_bias: bool = True

@dataclasses.dataclass
class YoloDetectionGenerator(hyperparams.Config):
  box_type: FPNConfig = dataclasses.field(
      default_factory=_build_dict(MIN_LEVEL, MAX_LEVEL, "original"))
  scale_xy: FPNConfig = dataclasses.field(
      default_factory=_build_dict(MIN_LEVEL, MAX_LEVEL, 1.0))
  path_scales: FPNConfig = dataclasses.field(
      default_factory=_build_path_scales(MIN_LEVEL, MAX_LEVEL))
  nms_type: str = 'greedy'
  iou_thresh: float = 0.001
  nms_thresh: float = 0.6
  max_boxes: int = 200
  pre_nms_points: int = 5000

@dataclasses.dataclass
class YoloLoss(hyperparams.Config):
  ignore_thresh: FPNConfig = dataclasses.field(
      default_factory=_build_dict(MIN_LEVEL, MAX_LEVEL, 0.0))
  truth_thresh: FPNConfig = dataclasses.field(
      default_factory=_build_dict(MIN_LEVEL, MAX_LEVEL, 1.0))
  box_loss_type: FPNConfig = dataclasses.field(
      default_factory=_build_dict(MIN_LEVEL, MAX_LEVEL, 'ciou'))
  iou_normalizer: FPNConfig = dataclasses.field(
      default_factory=_build_dict(MIN_LEVEL, MAX_LEVEL, 1.0))
  cls_normalizer: FPNConfig = dataclasses.field(
      default_factory=_build_dict(MIN_LEVEL, MAX_LEVEL, 1.0))
  object_normalizer: FPNConfig = dataclasses.field(
      default_factory=_build_dict(MIN_LEVEL, MAX_LEVEL, 1.0))
  max_delta: FPNConfig = dataclasses.field(
      default_factory=_build_dict(MIN_LEVEL, MAX_LEVEL, np.inf))
  objectness_smooth: FPNConfig = dataclasses.field(
      default_factory=_build_dict(MIN_LEVEL, MAX_LEVEL, 0.0))
  label_smoothing: float = 0.0
  use_scaled_loss: bool = True
  update_on_repeat: bool = True


class Box(hyperparams.Config):
  box: List[int] = dataclasses.field(default=list)

@dataclasses.dataclass
class AnchorBoxes(hyperparams.Config):
  boxes: List[Box] = None
  level_limits: Optional[List[int]] = None
  anchors_per_scale: int = 3

  def get(self, min_level, max_level):
    if self.level_limits is None:
      boxes = [box.box for box in self.boxes]
    else:
      boxes = [[1.0, 1.0]] * ((max_level - min_level) + 1) * self.anchors_per_scale
      # self.anchors_per_scale = 1
    
    anchors_per_level = dict()
    start = 0
    for i in range(min_level, max_level + 1):
      anchors_per_level[str(i)] = boxes[start:start + self.anchors_per_scale]
      start += self.anchors_per_scale
    return anchors_per_level, self.level_limits

@dataclasses.dataclass
class Yolo(hyperparams.Config):
  input_size: Optional[List[int]] = dataclasses.field(
      default_factory=lambda: [512, 512, 3])
  backbone: backbones.Backbone = backbones.Backbone(
      type='darknet', darknet=backbones.Darknet(model_id='cspdarknet53'))
  # distinct_fpn: Optional[decoders.Decoder] = None
  distinct_fpn: decoders.Decoder = decoders.Decoder(type=None)
  decoder: decoders.Decoder = decoders.Decoder(
      type='yolo_decoder', yolo_decoder=decoders.YoloDecoder(
        version='v4', type='regular'
      )
  )
  head: YoloHead = YoloHead()
  detection_generator: YoloDetectionGenerator = YoloDetectionGenerator()
  loss: YoloLoss = YoloLoss()
  norm_activation: common.NormActivation = common.NormActivation(
      activation='mish',
      use_sync_bn=True,
      norm_momentum=0.99,
      norm_epsilon=0.001)
  num_classes: int = 80
  anchor_boxes: AnchorBoxes = AnchorBoxes()
  darknet_based_model: bool = False

@dataclasses.dataclass
class YoloTask(cfg.TaskConfig):
  per_category_metrics: bool = False
  smart_bias_lr: float = 0.0
  model: Yolo = Yolo()
  train_data: DataConfig = DataConfig(is_training=True)
  validation_data: DataConfig = DataConfig(is_training=False)
  weight_decay: float = 0.0
  annotation_file: Optional[str] = None
  init_checkpoint: Optional[str] = None
  init_checkpoint_modules: Union[
      str, List[str]] = 'all'  # all, backbone, and/or decoder


COCO_INPUT_PATH_BASE = 'coco'
COCO_TRAIN_EXAMPLES = 118287
COCO_VAL_EXAMPLES = 5000
GLOBAL_SEED = 1000

@exp_factory.register_config_factory('yolo')
def yolo() -> cfg.ExperimentConfig:
  """Yolo general config."""
  return cfg.ExperimentConfig(
      task=YoloTask(),
      trainer=cfg.TrainerConfig(
        optimizer_config=optimization.OptimizationConfig()
        ),
      restrictions=[
          'task.train_data.is_training != None',
          'task.validation_data.is_training != None'
      ])

@exp_factory.register_config_factory('yolo_darknet')
def yolo_darknet() -> cfg.ExperimentConfig:
  """COCO object detection with YOLOv3 and v4"""
  train_batch_size = 64
  eval_batch_size = 8
  train_epochs = 300
  steps_per_epoch = COCO_TRAIN_EXAMPLES // train_batch_size
  validation_interval = 5

  max_num_instances = 200
  config = cfg.ExperimentConfig(
      runtime=cfg.RuntimeConfig(mixed_precision_dtype='bfloat16'),
      task=YoloTask(
          smart_bias_lr=0.1,
          init_checkpoint='',
          init_checkpoint_modules='backbone',
          annotation_file=None,
          weight_decay=0.0,
          model=Yolo(
              darknet_based_model = True,
              norm_activation=common.NormActivation(use_sync_bn=True),
              head=YoloHead(smart_bias=True), 
              loss=YoloLoss(use_scaled_loss=False, update_on_repeat=True)),
          train_data=DataConfig(
              is_training=True,
              global_batch_size=train_batch_size,
              seed=GLOBAL_SEED, 
              dtype='float32', 
              parser=Parser(
                letter_box=False,
                aug_rand_saturation= 1.5,
                aug_rand_brightness= 1.5,
                aug_rand_hue= 0.1,
                use_tie_breaker=True, 
                best_match_only=False, 
                anchor_thresh=0.213,
                area_thresh=0.1,
                max_num_instances=max_num_instances, 
                mosaic=Mosaic(
                  mosaic_frequency= 0.75,
                  mixup_frequency= 0.0,
                  mosaic_crop_mode= 'crop',
                  mosaic_center= 0.2
                )
              )),
          validation_data=DataConfig(
              is_training=False,
              global_batch_size=eval_batch_size, 
              drop_remainder=True, 
              dtype='float32', 
              parser=Parser(
                letter_box=False,
                use_tie_breaker=True, 
                best_match_only=False, 
                anchor_thresh=0.213,
                area_thresh=0.1,
                max_num_instances=max_num_instances, 
              ))),
      trainer=cfg.TrainerConfig(
          train_steps=train_epochs * steps_per_epoch,
          validation_steps=COCO_VAL_EXAMPLES // eval_batch_size,
          validation_interval=validation_interval * steps_per_epoch,
          steps_per_loop=steps_per_epoch,
          summary_interval=steps_per_epoch,
          checkpoint_interval=steps_per_epoch,
          optimizer_config=optimization.OptimizationConfig({
              'ema':{
                    'average_decay': 0.9998,
                    'trainable_weights_only': False,
                    'dynamic_decay': True,
              },
              'optimizer': {
                  'type': 'sgd_torch',
                  'sgd_torch': {
                      'momentum': 0.949,
                      'momentum_start': 0.949,
                      'nesterov': True,
                      'warmup_steps': 1000,
                      'weight_decay': 0.0005,
                      'sim_torch': True,
                  }
              },
              'learning_rate': {
                  'type': 'stepwise',
                  'stepwise': {
                      'boundaries': [240 * steps_per_epoch, 270*steps_per_epoch],
                      'values': [
                          0.00131 * train_batch_size / 64.0,
                          0.000131 * train_batch_size / 64.0,
                          0.0000131 * train_batch_size / 64.0
                      ]
                  }
              },
              'warmup': {
                  'type': 'linear',
                  'linear': {
                      'warmup_steps': 1000,
                      'warmup_learning_rate': 0
                  }
              }
          })),
      restrictions=[
          'task.train_data.is_training != None',
          'task.validation_data.is_training != None'
      ])

  return config


@exp_factory.register_config_factory('scaled_yolo')
def scaled_yolo() -> cfg.ExperimentConfig:
  """COCO object detection with YOLOv4-csp and v4"""
  train_batch_size = 64
  eval_batch_size = 8
  train_epochs = 300
  warmup_epochs = 3
  
  validation_interval = 5
  steps_per_epoch = COCO_TRAIN_EXAMPLES // train_batch_size

  max_num_instances = 300

  config = cfg.ExperimentConfig(
      runtime=cfg.RuntimeConfig(mixed_precision_dtype='bfloat16'),
      task=YoloTask(
          smart_bias_lr=0.1,
          init_checkpoint_modules=None,
          annotation_file=None,
          weight_decay=0.0,
          model=Yolo(
            darknet_based_model = False,
            norm_activation=common.NormActivation(
              activation='mish',
              use_sync_bn=True, 
              norm_epsilon=0.0001, 
              norm_momentum=0.97),
            head=YoloHead(smart_bias=True), 
            loss=YoloLoss(use_scaled_loss=True)),
          train_data=DataConfig(
              is_training=True,
              global_batch_size=train_batch_size,
              seed=GLOBAL_SEED, 
              dtype='float32',
              parser=Parser(
                aug_rand_saturation = 0.7, 
                aug_rand_brightness = 0.4,
                aug_rand_hue = 0.015, 
                letter_box=True,
                use_tie_breaker=True, 
                best_match_only=True, 
                anchor_thresh=4.0,
                random_pad=False,
                area_thresh=0.1,
                max_num_instances=max_num_instances,
                mosaic=Mosaic(
                  mosaic_crop_mode='scale',
                  mosaic_frequency=1.0, 
                  mixup_frequency=0.0, 
                ) 
              )),
          validation_data=DataConfig(
              is_training=False,
              global_batch_size=eval_batch_size, 
              drop_remainder=True, 
              dtype='float32', 
              parser=Parser(
                letter_box=True,
                use_tie_breaker=True, 
                best_match_only=True, 
                anchor_thresh=4.0,
                area_thresh=0.1,
                max_num_instances=max_num_instances, 
              ))),
      trainer=cfg.TrainerConfig(
          train_steps=train_epochs * steps_per_epoch,
          validation_steps=COCO_VAL_EXAMPLES // eval_batch_size,
          validation_interval=validation_interval * steps_per_epoch,
          steps_per_loop=steps_per_epoch,
          summary_interval=steps_per_epoch,
          checkpoint_interval=steps_per_epoch,
          optimizer_config=optimization.OptimizationConfig({
              'ema':{
                    'average_decay': 0.9999,
                    'trainable_weights_only': False,
                    'dynamic_decay': True,
              },
              'optimizer': {
                  'type': 'sgd_torch',
                  'sgd_torch': {
                      'momentum': 0.937,
                      'momentum_start': 0.8,
                      'nesterov': True,
                      'warmup_steps': steps_per_epoch * warmup_epochs,
                      'weight_decay': 0.0005 * train_batch_size/64.0,
                      'sim_torch': True,
                  }
              },
              'learning_rate': {
                  'type': 'cosine',
                  'cosine': {
                      'initial_learning_rate': 0.01,
                      'alpha': 0.2, 
                      'decay_steps': train_epochs * steps_per_epoch,
                  }
              },
              'warmup': {
                  'type': 'linear',
                  'linear': {
                      'warmup_steps': steps_per_epoch * warmup_epochs,
                      'warmup_learning_rate': 0
                  }
              }
          })),
      restrictions=[
          'task.train_data.is_training != None',
          'task.validation_data.is_training != None'
      ])

  return config

@exp_factory.register_config_factory('scaled_yolo_3l_coco')
def scaled_yolo_3l_coco() -> cfg.ExperimentConfig:
  """COCO object detection with YOLOv4-csp and v4"""
  train_batch_size = 128
  eval_batch_size = 8
  train_epochs = 300
  warmup_epochs = 3
  
  validation_interval = 3
  steps_per_epoch = COCO_TRAIN_EXAMPLES // train_batch_size

  max_num_instances = 300

  config = cfg.ExperimentConfig(
      runtime=cfg.RuntimeConfig(mixed_precision_dtype='bfloat16'),
      task=YoloTask(
          smart_bias_lr=0.1,
          init_checkpoint_modules=None,
          annotation_file=None,
          weight_decay=0.0,
          model=Yolo(
            darknet_based_model = False,
            norm_activation=common.NormActivation(
              activation='mish',
              use_sync_bn=True, 
              norm_epsilon=0.0001, 
              norm_momentum=0.97),
            head=YoloHead(smart_bias=True), 
            loss=YoloLoss(use_scaled_loss=True), 
            anchor_boxes={
              "anchors_per_scale": 3,
              "boxes": [{"box": [12, 16]},   {"box": [19, 36]},   {"box": [40, 28]}, 
                        {"box": [36, 75]},   {"box": [76, 55]},   {"box": [72, 146]}, 
                        {"box": [142, 110]}, {"box": [192, 243]}, {"box": [459, 401]}]
            }
            ),
          train_data=DataConfig(
              is_training=True,
              global_batch_size=train_batch_size,
              seed=GLOBAL_SEED, 
              dtype='float32',
              parser=Parser(
                aug_rand_saturation = 0.7, 
                aug_rand_brightness = 0.4,
                aug_rand_hue = 0.015, 
                letter_box=True,
                use_tie_breaker=True, 
                best_match_only=True, 
                anchor_thresh=4.0,
                random_pad=False,
                area_thresh=0.1,
                max_num_instances=max_num_instances,
                mosaic=Mosaic(
                  mosaic_crop_mode='scale',
                  mosaic_frequency=1.0, 
                  mixup_frequency=0.0, 
                ) 
              )),
          validation_data=DataConfig(
              is_training=False,
              global_batch_size=eval_batch_size, 
              drop_remainder=True, 
              dtype='float32', 
              parser=Parser(
                letter_box=True,
                use_tie_breaker=True, 
                best_match_only=True, 
                anchor_thresh=4.0,
                area_thresh=0.1,
                max_num_instances=max_num_instances, 
              ))),
      trainer=cfg.TrainerConfig(
          train_steps=train_epochs * steps_per_epoch,
          validation_steps=COCO_VAL_EXAMPLES // eval_batch_size,
          validation_interval=validation_interval * steps_per_epoch,
          steps_per_loop=steps_per_epoch,
          summary_interval=steps_per_epoch,
          checkpoint_interval=steps_per_epoch,
          optimizer_config=optimization.OptimizationConfig({
              'ema':{
                    'average_decay': 0.9999,
                    'trainable_weights_only': False,
                    'dynamic_decay': True,
              },
              'optimizer': {
                  'type': 'sgd_torch',
                  'sgd_torch': {
                      'momentum': 0.937,
                      'momentum_start': 0.8,
                      'nesterov': True,
                      'warmup_steps': steps_per_epoch * warmup_epochs,
                      'weight_decay': 0.0005 * train_batch_size/64.0,
                      'sim_torch': True,
                  }
              },
              'learning_rate': {
                  'type': 'cosine',
                  'cosine': {
                      'initial_learning_rate': 0.01,
                      'alpha': 0.2, 
                      'decay_steps': train_epochs * steps_per_epoch,
                  }
              },
              'warmup': {
                  'type': 'linear',
                  'linear': {
                      'warmup_steps': steps_per_epoch * warmup_epochs,
                      'warmup_learning_rate': 0
                  }
              }
          })),
      restrictions=[
          'task.train_data.is_training != None',
          'task.validation_data.is_training != None'
      ])

  return config


@exp_factory.register_config_factory('large_yolo')
def large_yolo() -> cfg.ExperimentConfig:
  """COCO object detection with YOLOv4-csp and v4."""
  train_batch_size = 64
  eval_batch_size = 8
  train_epochs = 300
  fine_tune_epochs = 450
  warmup_epochs = 3

  validation_interval = 3
  steps_per_epoch = COCO_TRAIN_EXAMPLES // train_batch_size

  max_num_instances = 300

  config = cfg.ExperimentConfig(
      runtime=cfg.RuntimeConfig(mixed_precision_dtype='bfloat16'),
      task=YoloTask(
          smart_bias_lr=0.1,
          init_checkpoint_modules='',
          annotation_file=None,
          weight_decay=0.0,
          model=Yolo(
              darknet_based_model=False,
              norm_activation=common.NormActivation(
                  activation='mish',
                  use_sync_bn=True,
                  norm_epsilon=0.0001,
                  norm_momentum=0.97),
              head=YoloHead(smart_bias=True),
              loss=YoloLoss(use_scaled_loss=True)),
          train_data=DataConfig(
              is_training=True,
              global_batch_size=train_batch_size,
              dtype='float32',
              parser=Parser(
                  aug_rand_saturation=0.7,
                  aug_rand_brightness=0.4,
                  aug_rand_hue=0.015,
                  letter_box=True,
                  use_tie_breaker=True,
                  best_match_only=True,
                  anchor_thresh=4.0,
                  random_pad=False,
                  area_thresh=0.1,
                  max_num_instances=max_num_instances,
                  mosaic=Mosaic(
                      mosaic_crop_mode='scale',
                      mosaic_frequency=1.0,
                      mixup_frequency=0.0,
                  ))),
          validation_data=DataConfig(
              is_training=False,
              global_batch_size=eval_batch_size,
              drop_remainder=True,
              dtype='float32',
              parser=Parser(
                  letter_box=True,
                  use_tie_breaker=True,
                  best_match_only=True,
                  anchor_thresh=4.0,
                  area_thresh=0.1,
                  max_num_instances=max_num_instances,
              ))),
      trainer=cfg.TrainerConfig(
          train_steps=train_epochs * steps_per_epoch,
          validation_steps=COCO_VAL_EXAMPLES // eval_batch_size,
          validation_interval=validation_interval * steps_per_epoch,
          steps_per_loop=steps_per_epoch,
          summary_interval=steps_per_epoch,
          checkpoint_interval=steps_per_epoch,
          optimizer_config=optimization.OptimizationConfig({
              'ema': {
                  'average_decay': 0.9999,
                  'trainable_weights_only': False,
                  'dynamic_decay': True,
              },
              'optimizer': {
                  'type': 'sgd_torch',
                  'sgd_torch': {
                      'momentum': 0.937,
                      'momentum_start': 0.9,
                      'nesterov': True,
                      'warmup_steps': steps_per_epoch * warmup_epochs,
                      'weight_decay': 0.0005 * train_batch_size / 64.0,
                  }
              },
              'learning_rate': {
                  'type': 'cosine',
                  'cosine': {
                      'initial_learning_rate': 0.01,
                      'alpha': 0.2,
                      'decay_steps': fine_tune_epochs * steps_per_epoch,
                  }
              },
              'warmup': {
                  'type': 'linear',
                  'linear': {
                      'warmup_steps': steps_per_epoch * warmup_epochs,
                      'warmup_learning_rate': 0
                  }
              }
          })),
      restrictions=[
          'task.train_data.is_training != None',
          'task.validation_data.is_training != None'
      ])

  return config


@exp_factory.register_config_factory('yolo_tiny')
def yolo_tiny() -> cfg.ExperimentConfig:
  """COCO object detection with YOLOv3 and v4"""
  train_batch_size = 256
  eval_batch_size = 8
  train_epochs = 600
  steps_per_epoch = COCO_TRAIN_EXAMPLES // train_batch_size
  validation_interval = 5

  max_num_instances = 200
  config = cfg.ExperimentConfig(
      runtime=cfg.RuntimeConfig(mixed_precision_dtype='bfloat16'),
      task=YoloTask(
          smart_bias_lr=0.1,
          init_checkpoint='',
          init_checkpoint_modules='backbone',
          annotation_file=None,
          weight_decay=0.0,
          model=Yolo(
              darknet_based_model = True,
              norm_activation=common.NormActivation(use_sync_bn=True),
              head=YoloHead(smart_bias=True), 
              loss=YoloLoss(use_scaled_loss=False, update_on_repeat=True)),
          train_data=DataConfig(
              is_training=True,
              global_batch_size=train_batch_size,
              seed=GLOBAL_SEED, 
              dtype='float32', 
              parser=Parser(
                letter_box=False,
                aug_rand_saturation= 1.5,
                aug_rand_brightness= 1.5,
                aug_rand_hue= 0.1,
                use_tie_breaker=True, 
                best_match_only=False, 
                # anchor_thresh=0.213,
                anchor_thresh=0.4,
                area_thresh=0.1,
                max_num_instances=max_num_instances, 
                mosaic=Mosaic(
                  mosaic_frequency= 0.75,
                  mixup_frequency= 0.0,
                  mosaic_crop_mode= 'crop',
                  mosaic_center= 0.2
                )
              )),
          validation_data=DataConfig(
              is_training=False,
              global_batch_size=eval_batch_size, 
              drop_remainder=True, 
              dtype='float32', 
              parser=Parser(
                letter_box=False,
                use_tie_breaker=True, 
                best_match_only=False, 
                anchor_thresh=0.213,
                area_thresh=0.1,
                max_num_instances=max_num_instances, 
              ))),
      trainer=cfg.TrainerConfig(
          train_steps=train_epochs * steps_per_epoch,
          validation_steps=COCO_VAL_EXAMPLES // eval_batch_size,
          validation_interval=validation_interval * steps_per_epoch,
          steps_per_loop=steps_per_epoch,
          summary_interval=steps_per_epoch,
          checkpoint_interval=steps_per_epoch,
          optimizer_config=optimization.OptimizationConfig({
              'ema':{
                    'average_decay': 0.9998,
                    'trainable_weights_only': False,
                    'dynamic_decay': True,
              },
              'optimizer': {
                  'type': 'sgd_torch',
                  'sgd_torch': {
                      'momentum': 0.9,
                      'momentum_start': 0.9,
                      'nesterov': True,
                      'warmup_steps': 1000,
                      'weight_decay': 0.0005,
                      'sim_torch': True,
                  }
              },
              'learning_rate': {
                  'type': 'stepwise',
                  'stepwise': {
                      'boundaries': [0.8 * train_epochs * steps_per_epoch],
                      'values': [
                          0.00261 * train_batch_size / 64.0,
                          0.000261 * train_batch_size / 64.0,
                      ]
                  }
              },
              'warmup': {
                  'type': 'linear',
                  'linear': {
                      'warmup_steps': 1000,
                      'warmup_learning_rate': 0
                  }
              }
          })),
      restrictions=[
          'task.train_data.is_training != None',
          'task.validation_data.is_training != None'
      ])

  return config
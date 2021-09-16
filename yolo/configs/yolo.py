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
# from yolo.ops.preprocessing_ops import random_pad
from typing import Dict, List, Optional, Union
import dataclasses
from yolo.modeling.layers import detection_generator

from official.core import exp_factory
from official.modeling import hyperparams
from official.modeling.hyperparams import config_definitions as cfg
from official.vision.beta.configs import common

from yolo import optimization

from yolo.configs import backbones
import numpy as np
import regex as re

COCO_INPUT_PATH_BASE = 'coco'
IMAGENET_TRAIN_EXAMPLES = 1281167
IMAGENET_VAL_EXAMPLES = 50000
IMAGENET_INPUT_PATH_BASE = 'imagenet-2012-tfrecord'


# default param classes
@dataclasses.dataclass
class ModelConfig(hyperparams.Config):
  def as_dict(self):
    model_kwargs = super().as_dict()
    if self.boxes is not None:
      model_kwargs.update({'boxes': [str(b) for b in self.boxes]})
    else:
      model_kwargs.update({'boxes': None})
    return model_kwargs

  def get_boxes(self):
    if self.boxes is None:
      return None
    boxes = []
    key = re.compile('([\d\.]+)')
    for box in self.boxes:
      if isinstance(box, list) or isinstance(box, tuple):
        boxes.append(box)
      elif isinstance(box, str):
        boxes.append([float(val) for val in key.findall(box)])
      elif isinstance(box, int):
        raise IOError('unsupported input type, only strings or tuples')
    return boxes

@dataclasses.dataclass
class FPNConfig(hyperparams.Config):
  def get(self):
    values = self.as_dict()
    if "all" in values and values["all"] is not None:
      for key in values:
        if key != 'all':
          values[key] = values["all"]
    print(values)
    return values


# dataset parsers
@dataclasses.dataclass
class Mosaic(hyperparams.Config):
  max_resolution: int = 640
  mosaic_frequency: float = 0.75
  mixup_frequency: float = 0.0
  crop_area: List[int] = dataclasses.field(default_factory=lambda: [0.2, 1.0])
  crop_area_mosaic: List[int] = dataclasses.field(
      default_factory=lambda: [1.0, 1.0])
  aspect_ratio_mode: str = 'crop'
  random_pad: bool = False
  mosaic_crop_mode: Optional[str] = 'crop_scale'
  aug_scale_min: Optional[float] = None
  aug_scale_max: Optional[float] = None
  jitter: Optional[float] = None
  resize: Optional[float] = None
  output_resolution: Optional[List[int]] = None  


@dataclasses.dataclass
class Parser(hyperparams.Config):
  max_num_instances: int = 200
  letter_box: Optional[bool] = True
  random_flip: bool = True
  random_pad: float = False
  jitter: float = 0.0
  resize: float = 1.0
  jitter_mosaic: float = 0.0
  resize_mosaic: float = 1.0
  aug_rand_angle: float = 0.0
  aug_rand_saturation: float = 0.0
  aug_rand_brightness: float = 0.0
  aug_rand_hue: float = 0.0
  aug_scale_min: float = 1.0
  aug_scale_max: float = 1.0
  aug_rand_translate: float = 0.0
  mosaic_scale_min: float = 1.0
  mosaic_scale_max: float = 1.0
  mosaic_translate: float = 0.0
  use_tie_breaker: bool = True
  use_scale_xy: bool = False
  best_match_only: bool = False
  anchor_thresh: float = 0.213
  area_thresh: float = 0.1
  stride: Optional[int] = None
  mosaic: Mosaic = Mosaic()


# pylint: disable=missing-class-docstring
@dataclasses.dataclass
class TfExampleDecoder(hyperparams.Config):
  regenerate_source_id: bool = False


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
class DataConfig(cfg.DataConfig):
  """Input config for training."""
  global_batch_size: int = 64
  input_path: str = ''  #'gs://tensorflow2/coco_records/train/2017*'
  tfds_name: str = None  #'coco'
  tfds_split: str = None  #'train'
  global_batch_size: int = 1
  is_training: bool = True
  dtype: str = 'float16'
  decoder: DataDecoder = DataDecoder()
  parser: Parser = Parser()
  shuffle_buffer_size: int = 10000
  tfds_download: bool = True
  cache: bool = False


@dataclasses.dataclass
class YoloDecoder(hyperparams.Config):
  """if the name is specified, or version is specified we ignore 
  input parameters and use version and name defaults"""
  version: Optional[str] = None
  type: Optional[str] = None
  use_fpn: Optional[bool] = None
  use_spatial_attention: bool = False
  use_separable_conv: bool = False
  csp_stack: Optional[bool] = None
  fpn_depth: Optional[int] = None
  fpn_filter_scale: Optional[int] = None
  path_process_len: Optional[int] = None
  max_level_process_len: Optional[int] = None
  embed_spp: Optional[bool] = None
  activation: Optional[str] = 'same'

@dataclasses.dataclass
class YoloHead(hyperparams.Config):
  """if the name is specified, or version is specified we ignore 
  input parameters and use version and name defaults"""
  smart_bias: bool = True

def _build_dict(min_level, max_level, value):
  vals = {str(key): value for key in range(min_level, max_level + 1)}
  vals["all"] = None
  return lambda: vals

def _build_path_scales(min_level, max_level):
  return lambda: {str(key): 2**key for key in range(min_level, max_level + 1)}

@dataclasses.dataclass
class YoloDetectionGenerator(hyperparams.Config):
  box_type: FPNConfig = dataclasses.field(
      default_factory=_build_dict(3, 7, "original"))
  scale_xy: FPNConfig = dataclasses.field(
      default_factory=_build_dict(3, 7, 1.0))
  path_scales: FPNConfig = dataclasses.field(
      default_factory=_build_path_scales(3, 7))
  nms_type: str = 'greedy'
  iou_thresh: float = 0.001
  nms_thresh: float = 0.6
  max_boxes: int = 200
  pre_nms_points: int = 5000

@dataclasses.dataclass
class YoloLoss(hyperparams.Config):
  ignore_thresh: FPNConfig = dataclasses.field(
      default_factory=_build_dict(3, 7, 0.0))
  truth_thresh: FPNConfig = dataclasses.field(
      default_factory=_build_dict(3, 7, 1.0))
  box_loss_type: FPNConfig = dataclasses.field(
      default_factory=_build_dict(3, 7, 'ciou'))
  iou_normalizer: FPNConfig = dataclasses.field(
      default_factory=_build_dict(3, 7, 1.0))
  cls_normalizer: FPNConfig = dataclasses.field(
      default_factory=_build_dict(3, 7, 1.0))
  obj_normalizer: FPNConfig = dataclasses.field(
      default_factory=_build_dict(3, 7, 1.0))
  max_delta: FPNConfig = dataclasses.field(
      default_factory=_build_dict(3, 7, np.inf))
  objectness_smooth: FPNConfig = dataclasses.field(
      default_factory=_build_dict(3, 7, 0.0))
  label_smoothing: float = 0.0
  use_scaled_loss: bool = True
  update_on_repeat: bool = True

@dataclasses.dataclass
class Yolo(ModelConfig):
  input_size: Optional[List[int]] = dataclasses.field(default_factory=lambda: [512, 512, 3])
  backbone: backbones.Backbone = backbones.Backbone(type='darknet', 
    darknet=backbones.Darknet(model_id='cspdarknet53'))
  decoder: YoloDecoder = YoloDecoder(version='v4', type='regular')
  head: YoloHead = YoloHead()
  detection_generator: YoloDetectionGenerator = YoloDetectionGenerator()
  loss: YoloLoss = YoloLoss()
  norm_activation: common.NormActivation = common.NormActivation(
      activation='leaky',
      use_sync_bn=True,
      norm_momentum=0.99,
      norm_epsilon=0.001)
  num_classes: int = 91
  boxes_per_scale: int = 3
  boxes: Optional[List[str]] = None
  anchor_free_limits: Optional[int] = None

# model task
@dataclasses.dataclass
class YoloTask(cfg.TaskConfig):
  model: Yolo = Yolo()
  train_data: DataConfig = DataConfig(is_training=True)
  validation_data: DataConfig = DataConfig(is_training=False)
  weight_decay: float = 5e-4
  annotation_file: Optional[str] = None
  gradient_clip_norm: float = 0.0
  
  smart_bias_lr: float = 0.0
  coco91to80: bool = False
  reduced_logs: bool = True
  init_checkpoint_modules: str = None
  per_category_metrics: bool = False


@dataclasses.dataclass
class YoloSubDivTask(YoloTask):
  subdivisions: int = 4


COCO_INPUT_PATH_BASE = 'coco'
COCO_TRIAN_EXAMPLES = 118287
COCO_VAL_EXAMPLES = 5000


@exp_factory.register_config_factory('yolo_custom')
def yolo_custom() -> cfg.ExperimentConfig:
  """COCO object detection with YOLO."""
  train_batch_size = 1
  eval_batch_size = 1
  base_default = 1200000
  num_batches = 1200000 * 64 / train_batch_size

  config = cfg.ExperimentConfig(
      runtime=cfg.RuntimeConfig(
          #            mixed_precision_dtype='float16',
          #            loss_scale='dynamic',
          num_gpus=2),
      task=YoloTask(
          model=Yolo(),
          train_data=DataConfig(  # input_path=os.path.join(
              # COCO_INPUT_PATH_BASE, 'train*'),
              is_training=True,
              global_batch_size=train_batch_size,
              parser=Parser(),
              shuffle_buffer_size=2),
          validation_data=DataConfig(
              # input_path=os.path.join(COCO_INPUT_PATH_BASE,
              #                        'val*'),
              is_training=False,
              global_batch_size=eval_batch_size,
              shuffle_buffer_size=2)),
      trainer=cfg.TrainerConfig(
          steps_per_loop=2000,
          summary_interval=8000,
          checkpoint_interval=10000,
          train_steps=num_batches,
          validation_steps=1000,
          validation_interval=10,
          optimizer_config=optimization.OptimizationConfig({
              'optimizer': {
                  'type': 'sgd',
                  'sgd': {
                      'momentum': 0.9
                  }
              },
              'learning_rate': {
                  'type': 'stepwise',
                  'stepwise': {
                      'boundaries': [
                          int(400000 / base_default * num_batches),
                          int(450000 / base_default * num_batches)
                      ],
                      'values': [
                          0.00261 * train_batch_size / 64,
                          0.000261 * train_batch_size / 64,
                          0.0000261 * train_batch_size / 64
                      ]
                  }
              },
              'warmup': {
                  'type': 'linear',
                  'linear': {
                      'warmup_steps': 1000 * 64 // num_batches,
                      'warmup_learning_rate': 0
                  }
              }
          })),
      restrictions=[
          'task.train_data.is_training != None',
          'task.validation_data.is_training != None'
      ])

  return config

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

import os
from typing import Dict, List, Optional
import dataclasses

from official.core import exp_factory
from official.modeling import hyperparams
from official.modeling import optimization
from official.modeling.hyperparams import config_definitions as cfg
from official.vision.beta.configs import backbones
from official.vision.beta.configs import common

import tensorflow as tf

@dataclasses.dataclass
class Parser(hyperparams.Config):
  image_w: int = 416
  image_h: int = 416
  fixed_size: bool = False
  jitter_im: float = 0.1
  jitter_boxes: float = 0.005
  net_down_scale: int = 32
  max_process_size: int = 608
  min_process_size: int = 320
  pct_rand: float = 0.5


@dataclasses.dataclass
class DataConfig(cfg.DataConfig):
  """Input config for training."""
  input_path: str = ''
  global_batch_size: int = 0
  is_training: bool = False
  dtype: str = 'bfloat16'
  decoder: DataDecoder = DataDecoder()
  parser: Parser = Parser()
  shuffle_buffer_size: int = 10000


@dataclasses.dataclass
class YoloLayer(hyperparams.Config):
  thresh: int = 0.45
  class_thresh: int = 0.45


@dataclasses.dataclass
class Yolo(hyperparams.Config):
  input_shape: List[int] = dataclasses.field(default_factory=list)
  version: str = "v3"
  model: str = "regular"
  path_scales: List[int] = None
  x_y_scales: List[int] = None
  use_tie_breaker: bool = False
  clip_grads_norm: Optional[float] = None
  policy: str = "float32"
  weight_decay: float = 5e-4


@dataclasses.dataclass
class YoloTask(cfg.TaskConfig):
  num_classes: int = 80
  masks: Dict[str, List[int]] = None
  anchors: List[List[int]] = None
  max_boxes: int = 200
  scale_boxes: int = 416
  scale_mult: float = 1.0

  model: Yolo = Yolo()
  train_data: DataConfig = DataConfig(is_training=True)
  validation_data: DataConfig = DataConfig(is_training=False)
  filter: YoloLayer = YoloLayer()
  init_checkpoint: Optional[str] = None
  init_checkpoint_modules: str = 'all'  # all or backbone
  annotation_file: Optional[str] = None
  gradient_clip_norm: float = 0.0
  per_category_metrics = False

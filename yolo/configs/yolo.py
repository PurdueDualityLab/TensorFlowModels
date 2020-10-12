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
class GridpointGenerator():
    low_memory = False
    reset = True

@dataclasses.dataclass
class Anchors():
    prediction_size = 416
    boxes = [(12, 16), (19, 36), (40, 28), (36, 75),(76, 55), (72, 146), (142, 110),(192, 243), (459, 401)] 

@dataclasses.dataclass
class YoloLossLayer(hyperparams.Config):
    thresh: int = 0.45
    class_thresh: int = 0.45
    boxes = Anchors()
    masks = {"1024": [6, 7, 8], "512": [3, 4, 5], "256": [0, 1, 2]}
    path_scales = {"1024": 32, "512": 16, "256": 8}
    x_y_scales = {"1024": 1.05, "512": 1.1, "256": 1.2}
    use_tie_breaker: bool = True

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
    box_param: Anchors = Anchors()
    shuffle_buffer_size: int = 10000

@dataclasses.dataclass
class DataConfig(cfg.DataConfig):
    """Input config for training."""
    input_path: str = ''
    global_batch_size: int = 10
    is_training: bool = True
    dtype: str = 'float16'
    decoder: DataDecoder = None
    parser: Parser = Parser()
    shuffle_buffer_size: int = 10000


@dataclasses.dataclass
class Darknet(hyperparams.Config):
    input_shape: List[int] = dataclasses.field(default_factory=list)

@dataclasses.dataclass
class YoloNeck(hyperparams.Config):
    input_shape: List[int] = dataclasses.field(default_factory=list)

@dataclasses.dataclass
class YoloHead(hyperparams.Config):
    input_shape: List[int] = dataclasses.field(default_factory=list)

@dataclasses.dataclass
class Yolov3(hyperparams.Config):
    input_shape: List[int] = dataclasses.field(default_factory=list)
    model: str = "regular"
    backbone = None
    head = None 

@dataclasses.dataclass
class Yolov4(hyperparams.Config):
    input_shape: List[int] = dataclasses.field(default_factory=list)
    backbone = None
    neck = None
    head = None

@dataclasses.dataclass
class YoloTask(cfg.TaskConfig):
    num_classes: int = 80
    input_size: List[int] = dataclasses.field(default_factory=list)
    masks: Dict[str, List[int]] = None
    anchors: List[List[int]] = None
    min_level: int = 3
    max_level: int = 5
    weight_decay: float = 5e-4
    gradient_clip_norm: float = 0.0
    max_boxes: int = 200


    train_data: DataConfig = DataConfig(is_training=True)
    validation_data: DataConfig = DataConfig(is_training=False)
    filter: YoloLayer = YoloLayer()


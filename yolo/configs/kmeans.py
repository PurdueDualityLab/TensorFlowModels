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
  shuffle_buffer_size: int = 10000
  tfds_download: bool = True
  cache: bool = False

@dataclasses.dataclass
class KMeans(hyperparams.Config):
  mosaic_frequency: float = 0.0
  mixup_frequency: float = 0.0
  mosaic_center: float = 0.2
  mosaic_crop_mode: Optional[str] = None
  aug_scale_min: float = 1.0
  aug_scale_max: float = 1.0
  jitter: float = 0.0
  random_pad: float = False
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

"""Backbones configurations."""

import dataclasses

from official.modeling import hyperparams

from official.vision.beta.configs import backbones


@dataclasses.dataclass
class Darknet(hyperparams.Config):
  """Darknet config."""
  model_id: str = 'darknet53'
  width_scale: float = 1.0
  depth_scale: float = 1.0
  dilate: bool = False
  min_level: int = 3
  max_level: int = 5


@dataclasses.dataclass
class Backbone(backbones.Backbone):
  darknet: Darknet = Darknet()

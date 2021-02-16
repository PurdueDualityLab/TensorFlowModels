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
"""Decoder configurations."""
from typing import Dict

# Import libraries
import dataclasses

from official.modeling import hyperparams

# Note these do not subclass from hyperparams.Config, 
# for some reason Dicts are not supported in their
# immutable types

@dataclasses.dataclass
class CenterNet2D():
  """CenterNet for 2D Object Detection Decoder."""
  task_outputs: Dict[str, int] = dataclasses.field(
    default_factory=lambda: {'heatmap': 91, 'local_offset': 2, 'object_size': 2})
  heatmap_bias: float = -2.19

@dataclasses.dataclass
class CenterNet3D():
  """CenterNet for 3D Object Detection Decoder."""
  task_outputs: Dict[str, int] = dataclasses.field(
    default_factory=lambda: {'heatmap': 91,
    'local_offset': 2, 
    'object_size': 3,
    'depth' : 1,
    'orientation': 8})
  heatmap_bias: float = -2.19

@dataclasses.dataclass
class CenterNetPose():
  """CenterNet for Pose Estimation Decoder."""
  task_outputs: Dict[str, int] = dataclasses.field(
    default_factory=lambda: {'heatmap': 17,
    'joint_locs': 17 * 2, 
    'joint_offset': 2})
  heatmap_bias: float = -2.19
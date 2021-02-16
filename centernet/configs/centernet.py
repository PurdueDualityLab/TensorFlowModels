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
from official.modeling.hyperparams import config_definitions as cfg
from official.vision.beta.configs import common
from official.vision.beta.configs import backbones

from centernet.configs import backbones

# Note these do not subclass from hyperparams.Config,
# for some reason Dicts are not supported in their
# immutable types


@dataclasses.dataclass
class CenterNetDecoder(hyperparams.Config):
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


@dataclasses.dataclass
class CenterNetDecoder(hyperparams.Config):
  heatmap_bias: float = -2.19


@dataclasses.dataclass
class CenterNet(hyperparams.Config):
  num_classes: int = 80
  decoder: CenterNetDecoder = CenterNetDecoder()


@dataclasses.dataclass
class CenterNetDetection(cfg.TaskConfig):
  use_centers: bool = True
  use_corners: bool = False
  predict_3d: bool = False


@dataclasses.dataclass
class CenterNetSubTasks(cfg.TaskConfig):
  detection: CenterNetDetection = CenterNetDetection()
  # kp_detection: bool = False
  segmentation: bool = False
  # pose: bool = False
  # reid: bool = False
  # temporal: bool = False


@dataclasses.dataclass
class CenterNetTask(cfg.TaskConfig):
  model: CenterNet = CenterNet()
  subtasks: CenterNetSubTasks = CenterNetSubTasks()
  def _get_output_length_dict(self):
    lengths = {}
    assert self.subtasks.detection is not None or self.subtasks.kp_detection \
        or self.subtasks.segmentation, "You must specify at least one " \
        "subtask to CenterNet"

    if self.subtasks.detection:
      assert self.subtasks.detection.use_centers or \
          self.subtasks.detection.use_corners, "Cannot use CenterNet without " \
          "heatmaps"
      if self.subtasks.detection.use_centers:
        lengths.update({
          'ct_heatmaps': self.model.num_classes, # renamed from heatmap
          'ct_offset': 2, # renamed from local_offset
          'ct_size': 2 # renamed from object_size
        })
      if self.subtasks.detection.use_corners:
        lengths.update({
          'tl_heatmaps': self.model.num_classes,
          'tl_offset': 2,
          'tl_size': 2,
          'br_heatmaps': self.model.num_classes,
          'br_offset': 2,
          'br_size': 2,
        })
      if self.subtasks.detection.predict_3d:
        lengths.update({
          'depth': 1,
          'orientation': 8
        })

    if self.subtasks.segmentation:
      lengths['seg_heatmaps'] = self.model.num_classes

    # if self.subtasks.pose:
    #   lengths.update({
    #     'pose_heatmaps': 17,
    #     'joint_locs': 17 * 2,
    #     'joint_offset': 2
    #   })

    return lengths

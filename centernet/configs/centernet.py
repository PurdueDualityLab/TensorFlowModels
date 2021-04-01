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
"""CenterNet configuration definition."""
from typing import ClassVar, Dict, List, Optional, Tuple, Union

# Import libraries
import dataclasses

from official.modeling import hyperparams
from official.modeling.hyperparams import config_definitions as cfg
from official.vision.beta.configs import common
from centernet.configs import backbones


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

# dataset parsers
@dataclasses.dataclass
class Parser(hyperparams.Config):
  image_w: int = 512
  image_h: int = 512
  fixed_size: bool = True
  max_num_instances: int = 200
  min_process_size: int = 320
  max_process_size: int = 608
  letter_box: bool = True
  random_flip: bool = True
  pct_rand: float = 0.0
  jitter_im: float = 0.0
  jitter_boxes: float = 0.000
  aug_rand_transalate: float = 0.0
  aug_rand_saturation: float = 0.0
  aug_rand_brightness: float = 0.0
  aug_rand_zoom: float = 0.0
  aug_rand_hue: float = 0.0
  keep_thresh: float = 0.0
  mosaic_frequency: float = 1.0
  use_tie_breaker: bool = True

@dataclasses.dataclass
class DataConfig(cfg.DataConfig):
  """Input config for training."""
  input_path: str = 'D:/Datasets/coco/2017/1.1.0/val' #'gs://tensorflow2/coco_records/train/2017*'
  tfds_name: str = None #'coco'
  tfds_split: str = 'validation' #'train'
  global_batch_size: int = 32
  is_training: bool = True
  dtype: str = 'float16'
  decoder: DataDecoder = DataDecoder()
  parser: Parser = Parser()
  shuffle_buffer_size: int = 10000
  tfds_download: bool = True

@dataclasses.dataclass
class Loss(hyperparams.Config):
  pass


@dataclasses.dataclass
class DetectionLoss(Loss):
  detection_weight: float = 1.0
  corner_pull_weight: float = 0.1 # alpha
  corner_push_weight: float = 0.1 # beta
  offset_weight: float = 1 # gamma


@dataclasses.dataclass
class SegmentationLoss(Loss):
  pass


@dataclasses.dataclass
class Losses(hyperparams.Config):
  detection: DetectionLoss = DetectionLoss()
  segmentation: SegmentationLoss = SegmentationLoss()


@dataclasses.dataclass
class CenterNetDecoder(hyperparams.Config):
  heatmap_bias: float = -2.19


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
class CenterNetBase(hyperparams.OneOfConfig):
  backbone: backbones.Backbone = backbones.Backbone(type='hourglass')
  decoder: CenterNetDecoder = CenterNetDecoder()

@dataclasses.dataclass
class CenterNet(hyperparams.Config):
  num_classes: int = 90
  gaussian_iou: float = 0.7
  max_num_instances: int = 200
  input_size: Optional[List[int]] = dataclasses.field(
    default_factory=lambda: [None, None, 3])
  base: Union[str, CenterNetBase] = CenterNetBase()

@dataclasses.dataclass
class CenterNetTask(cfg.TaskConfig):
  model: CenterNet = CenterNet()
  train_data: DataConfig = DataConfig(is_training=True)
  validation_data: DataConfig = DataConfig(is_training=False)
  subtasks: CenterNetSubTasks = CenterNetSubTasks()
  losses: Losses = Losses()

  annotation_file: Optional[str] = None
  per_category_metrics: bool = False

  weight_decay: float = 5e-4

  def _get_output_length_dict(self):
    lengths = {}
    assert self.subtasks.detection is not None or self.subtasks.kp_detection \
        or self.subtasks.segmentation, "You must specify at least one " \
        "subtask to CenterNet"

    if self.subtasks.detection:
      # TODO: locations of the ground truths will also be passed in from the
      # data pipeline which need to be mapped accordingly
      assert self.subtasks.detection.use_centers or \
          self.subtasks.detection.use_corners, "Cannot use CenterNet without " \
          "heatmaps"
      if self.subtasks.detection.use_centers:
        lengths.update({
          'ct_heatmaps': self.model.num_classes,
          'ct_offset': 2,
        })
        if not self.subtasks.detection.use_corners:
          lengths['ct_size'] = 2

      if self.subtasks.detection.use_corners:
        lengths.update({
          'tl_heatmaps': self.model.num_classes,
          'tl_offset': 2,
          'br_heatmaps': self.model.num_classes,
          'br_offset': 2
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
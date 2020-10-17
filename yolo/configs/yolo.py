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
import tensorflow as tf
from typing import Dict, List, Optional, Tuple, Union
import dataclasses
import os

from official.core import exp_factory
from official.modeling import hyperparams, optimization
from official.modeling.hyperparams import config_definitions as cfg
from yolo.configs import cfg_defs as yolo_cfg

COCO_INPUT_PATH_BASE = 'coco'
IMAGENET_TRAIN_EXAMPLES = 1281167
IMAGENET_VAL_EXAMPLES = 50000
IMAGENET_INPUT_PATH_BASE = 'imagenet-2012-tfrecord'


# dataset parsers
@dataclasses.dataclass
class Parser(hyperparams.Config):
    image_w: int = 416
    image_h: int = 416
    fixed_size: bool = False
    jitter_im: float = 0.1
    jitter_boxes: float = 0.005
    net_down_scale: int = 32
    min_process_size: int = 320
    max_process_size: int = 608
    max_num_instances: int = 200
    random_flip: bool = True
    pct_rand: float = 0.5
    seed: int = 10
    shuffle_buffer_size: int = 10000

@dataclasses.dataclass
class DataConfig(cfg.DataConfig):
    """Input config for training."""
    input_path: str = ''
    global_batch_size: int = 10
    is_training: bool = True
    dtype: str = 'float16'
    decoder = None
    parser: Parser = Parser()
    shuffle_buffer_size: int = 10000

# Loss and Filter definitions
@dataclasses.dataclass
class Gridpoints(hyperparams.Config):
    low_memory: bool = False
    reset: bool = True

@dataclasses.dataclass
class YoloBackbone(hyperparams.Config):
    version: str = "CSPDarknet53"
    name: str = "regular"
    cfg: Optional[Dict] = None

@dataclasses.dataclass
class YoloNeck(hyperparams.Config):
    version: str = "v4"
    name: str = "regular"
    cfg: Optional[Dict] = None

@dataclasses.dataclass
class YoloHead(hyperparams.Config):
    version: str = "v4"
    name: str = "regular"
    cfg: Optional[Dict] = None

@dataclasses.dataclass
class YoloLayer(hyperparams.Config):
    generator_params: Gridpoints = Gridpoints()
    iou_thresh: float = 0.45
    class_thresh: float = 0.45
    max_boxes: int = 200
    anchor_generation_scale: int = 416
    use_nms: bool = True

@dataclasses.dataclass
class YoloLoss(hyperparams.Config):
    ignore_thresh: float = 0.7
    truth_thresh: float = 1.0
    use_tie_breaker: bool = None

# model definition
@dataclasses.dataclass
class Yolov4regular(yolo_cfg.YoloCFG):
    model: str = "regular"
    backbone: Optional[Dict] = None #"regular"
    neck: Optional[Dict] = None #"regular"
    head: Optional[Dict] = None #"regular"
    head_filter: YoloLayer = YoloLayer()
    _boxes: List[str] = dataclasses.field(default_factory=lambda:["12, 16", "19, 36", "40, 28", "36, 75", "76, 55", "72, 146", "142, 110" ,"192, 243", "459, 401"])
    masks: Dict = dataclasses.field(default_factory=lambda: {5: [6, 7, 8], 4: [3, 4, 5], 3: [0, 1, 2]})
    path_scales: Dict = dataclasses.field(default_factory=lambda: {5: 32, 4: 16, 3: 8})
    x_y_scales: Dict = dataclasses.field(default_factory=lambda: {5: 1.05, 4: 1.1, 3: 1.2})
    use_tie_breaker: bool = None

@dataclasses.dataclass
class Yolov3regular(yolo_cfg.YoloCFG):
    model: str = "regular"
    backbone: Optional[Dict] = None #"regular"
    head: Optional[Dict] = None #"regular"
    head_filter: YoloLayer = YoloLayer()
    _boxes: List[str] = dataclasses.field(default_factory=lambda:["10, 13", "16, 30", "33, 23", "30, 61", "62, 45", "59, 119", "116, 90" ,"156, 198", "373, 326"])
    masks: Dict = dataclasses.field(default_factory=lambda: {5: [6, 7, 8], 4: [3, 4, 5], 3: [0, 1, 2]})
    path_scales: Dict = dataclasses.field(default_factory=lambda: {5: 32, 4: 16, 3: 8})
    x_y_scales: Dict = dataclasses.field(default_factory=lambda: {5: 1.0, 4: 1.0, 3: 1.0})
    use_tie_breaker: bool = None

@dataclasses.dataclass
class Yolov3spp(yolo_cfg.YoloCFG):
    model: str = "spp"
    backbone: Optional[Dict] = None #"regular"
    head: Optional[Dict] = None #"spp"
    head_filter: YoloLayer = YoloLayer()
    _boxes: List[str] = dataclasses.field(default_factory=lambda:["10, 13", "16, 30", "33, 23", "30, 61", "62, 45", "59, 119", "116, 90" ,"156, 198", "373, 326"])
    masks: Dict = dataclasses.field(default_factory=lambda: {5: [6, 7, 8], 4: [3, 4, 5], 3: [0, 1, 2]})
    path_scales: Dict = dataclasses.field(default_factory=lambda: {5: 32, 4: 16, 3: 8})
    x_y_scales: Dict = dataclasses.field(default_factory=lambda: {5: 1.0, 4: 1.0, 3: 1.0})
    use_tie_breaker: bool = None

@dataclasses.dataclass
class Yolov3tiny(yolo_cfg.YoloCFG):
    model: str = "tiny"
    backbone: Optional[Dict] = None #"tiny"
    head: Optional[Dict] = None #"tiny"
    head_filter: YoloLayer = YoloLayer()
    _boxes: List[str] = dataclasses.field(default_factory=lambda:["10, 14", "23, 27", "37, 58","81, 82", "135, 169", "344, 319"])
    masks: Dict = dataclasses.field(default_factory=lambda: {5: [3, 4, 5], 3: [0, 1, 2]})
    path_scales: Dict = dataclasses.field(default_factory=lambda: {5: 32, 3: 8})
    x_y_scales: Dict = dataclasses.field(default_factory=lambda: {5: 1.0, 3: 1.0})
    use_tie_breaker: bool = None

@dataclasses.dataclass
class Yolo(hyperparams.OneOfConfig):
    type: str = "v4"
    v3: Yolov3regular = Yolov3regular()
    v3_spp: Yolov3spp = Yolov3spp()
    v3_tiny: Yolov3tiny = Yolov3tiny()
    v4: Yolov4regular = Yolov4regular()

# model task
@dataclasses.dataclass
class YoloTask(yolo_cfg.TaskConfig):
    _input_size: Optional[List[int]] = None
    model:Yolo = Yolo()
    loss:YoloLoss = YoloLoss()
    train_data: DataConfig = DataConfig(is_training=True)
    validation_data: DataConfig = DataConfig(is_training=False)
    num_classes: int = 80
    min_level: int = 3
    max_level: int = 5
    weight_decay: float = 5e-4
    gradient_clip_norm: float = 0.0

    init_checkpoint_modules: str = 'all'  # all or backbone
    init_checkpoint: Optional[str] = None
    annotation_file: Optional[str] = None
    per_category_metrics = False

    load_original_weights: bool = True
    backbone_from_darknet: bool = True
    head_from_darknet: bool = False


@exp_factory.register_config_factory('yolo_v4_coco')
def yolo_v4_coco() -> cfg.ExperimentConfig:
  """COCO object detection with YOLO."""
  train_batch_size = 4096
  eval_batch_size = 4096
  steps_per_epoch = IMAGENET_TRAIN_EXAMPLES // train_batch_size

  config = cfg.ExperimentConfig(
      runtime=cfg.RuntimeConfig(mixed_precision_dtype='float32'),
      task=YoloTask(
          model=Yolo(
              type='v4',
              v4=Yolov4regular(
                  backbone=YoloBackbone(),
                  neck=YoloNeck(),
                  head=YoloHead())),
          train_data=DataConfig(
              input_path=os.path.join(COCO_INPUT_PATH_BASE, 'train*'),
              is_training=True,
              global_batch_size=train_batch_size,
              parser=Parser()),
          validation_data=DataConfig(
              input_path=os.path.join(COCO_INPUT_PATH_BASE, 'val*'),
              is_training=False,
              global_batch_size=eval_batch_size)),
      trainer=cfg.TrainerConfig(
          steps_per_loop=steps_per_epoch,
          summary_interval=steps_per_epoch,
          checkpoint_interval=steps_per_epoch,
          train_steps=90 * steps_per_epoch,
          validation_steps=IMAGENET_VAL_EXAMPLES // eval_batch_size,
          validation_interval=steps_per_epoch,
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
                          30 * steps_per_epoch, 60 * steps_per_epoch,
                          80 * steps_per_epoch
                      ],
                      'values': [0.8, 0.08, 0.008, 0.0008]
                  }
              },
              'warmup': {
                  'type': 'linear',
                  'linear': {
                      'warmup_steps': 5 * steps_per_epoch,
                      'warmup_learning_rate': 0
                  }
              }
          })),
      restrictions=[
          'task.train_data.is_training != None',
          'task.validation_data.is_training != None'
      ])

  return config

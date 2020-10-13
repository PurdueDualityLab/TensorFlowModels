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
from official.modeling import hyperparams
from official.modeling import optimization
from official.modeling.hyperparams import config_definitions as cfg
from official.vision.beta.configs import backbones
from official.vision.beta.configs import common

# model definitions
@dataclasses.dataclass
class Yolov4regular(hyperparams.Config):
    model: str = "regular"
    backbone: Optional[Dict] = None #"regular"
    neck: Optional[Dict] = None #"regular"
    head: Optional[Dict] = None #"regular"
    use_tie_breaker: bool = True

@dataclasses.dataclass
class Yolov3regular(hyperparams.Config):
    model: str = "regular"
    backbone: Optional[Dict] = None #"regular"
    head: Optional[Dict] = None #"regular"
    use_tie_breaker: bool = False

@dataclasses.dataclass
class Yolov3spp(hyperparams.Config):
    model: str = "spp"
    backbone: Optional[Dict] = None #"regular"
    head: Optional[Dict] = None #"spp"
    use_tie_breaker: bool = False

@dataclasses.dataclass
class Yolov3tiny(hyperparams.Config):
    model: str = "tiny"
    backbone: Optional[Dict] = None #"tiny"
    head: Optional[Dict] = None #"tiny"
    use_tie_breaker: bool = False

# loss function parameters
@dataclasses.dataclass
class Gridpoints(hyperparams.Config):
    low_memory: bool = False
    reset: bool = True

@dataclasses.dataclass
class Anchors(hyperparams.Config):
    prediction_scale: int = 416
    _boxes: List[str] = dataclasses.field(default_factory=lambda:["12, 16", "19, 36", "40, 28", "36, 75", "76, 55", "72, 146", "142, 110" ,"192, 243", "459, 401"])

    @property
    def boxes(self):
        boxes = []
        for box in self._boxes:
            f = []
            for b in box.split(","):
                f.append(int(b.strip()))
            boxes.append(f)
        return boxes
    
    @boxes.setter 
    def input_size(self, box_string):
        self._boxes = box_string

@dataclasses.dataclass
class YoloLossLayer(hyperparams.Config):
    thresh: int = 0.45
    class_thresh: int = 0.45
    anchors: Anchors = Anchors()
    masks: Dict = dataclasses.field(default_factory=lambda: {"1024": [6, 7, 8], "512": [3, 4, 5], "256": [0, 1, 2]})
    path_scales: Dict = dataclasses.field(default_factory=lambda: {"1024": 32, "512": 16, "256": 8})
    x_y_scales: Dict = dataclasses.field(default_factory=lambda: {"1024": 1.05, "512": 1.1, "256": 1.2})
    use_tie_breaker: bool = True
    generator_params: Gridpoints = Gridpoints()

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
    anchors: Anchors = Anchors()

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


# model selction
@dataclasses.dataclass
class Yolo(hyperparams.OneOfConfig):
    type: str = "v4"
    v3: Yolov3regular = Yolov3regular()
    v3_spp: Yolov3spp = Yolov3spp()
    v3_tiny: Yolov3tiny = Yolov3tiny()
    v4: Yolov4regular = Yolov4regular()

# model task
@dataclasses.dataclass
class YoloTask(cfg.TaskConfig):
    num_classes: int = 80
    _input_size: Optional[List[int]] = None
    model:Yolo = Yolo()
    lossfilter:YoloLossLayer = YoloLossLayer()
    min_level: int = 3
    max_level: int = 5
    weight_decay: float = 5e-4
    gradient_clip_norm: float = 0.0
    max_boxes: int = 200
    train_data: DataConfig = DataConfig(is_training=True)
    validation_data: DataConfig = DataConfig(is_training=False)
    init_checkpoint: Optional[str] = None
    init_checkpoint_modules: str = 'all'  # all or backbone
    annotation_file: Optional[str] = None
    gradient_clip_norm: float = 0.0
    per_category_metrics = False
    load_original_weights: bool = True
    backbone_from_darknet: bool = True
    head_from_darknet: bool = False
    weights_file: Optional[str] = None
    use_nms: bool = True


    @property
    def input_size(self):
        if self._input_size == None:
            return [None, None, 3]
        else:
            return self._input_size
    
    @input_size.setter 
    def input_size(self, input_size):
        self._input_size = input_size

    def get_build_model_dict(self):
        task_dict = {
            "input_shape":[None] + self.input_size, 
            "classes":self.num_classes, 
            "weight_decay": self.weight_decay,
            "max_boxes": self.max_boxes,
            "model": self.model.type
        }

        model_dict = self.model.as_dict()
        task_dict.update(model_dict[self.model.type])
        return task_dict



    


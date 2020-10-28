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
from official.vision.beta.configs import common

from yolo.configs import backbones

COCO_INPUT_PATH_BASE = 'coco'
IMAGENET_TRAIN_EXAMPLES = 1281167
IMAGENET_VAL_EXAMPLES = 50000
IMAGENET_INPUT_PATH_BASE = 'imagenet-2012-tfrecord'


# default param classes
@dataclasses.dataclass
class AnchorCFG(hyperparams.Config):
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
    def boxes(self, box_list):
        setter = []
        for value in box_list:
            value = str(list(value))
            setter.append(value[1:-1])
        self._boxes = setter


@dataclasses.dataclass
class ModelConfig(hyperparams.Config):
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
        model_cfg = getattr(self.model, self.model.type)
        model_kwargs = model_cfg.as_dict()

        # TODO: Better method
        for k in ('_boxes', 'backbone', 'head', 'neck', 'head_filter'):
            model_kwargs.pop(k)
        return model_kwargs


@dataclasses.dataclass
class Yolov3Anchors(AnchorCFG):
    """RevNet config."""
    # Specifies the depth of RevNet.
    _boxes: List[str] = dataclasses.field(default_factory=lambda: [
        "10, 13", "16, 30", "33, 23", "30, 61", "62, 45", "59, 119", "116, 90",
        "156, 198", "373, 326"
    ])
    masks: Dict = dataclasses.field(default_factory=lambda: {
        5: [6, 7, 8],
        4: [3, 4, 5],
        3: [0, 1, 2]
    })
    path_scales: Dict = dataclasses.field(default_factory=lambda: {
        5: 32,
        4: 16,
        3: 8
    })
    x_y_scales: Dict = dataclasses.field(default_factory=lambda: {
        5: 1.0,
        4: 1.0,
        3: 1.0
    })


@dataclasses.dataclass
class Yolov4Anchors(AnchorCFG):
    """RevNet config."""
    # Specifies the depth of RevNet.
    _boxes: List[str] = dataclasses.field(default_factory=lambda: [
        "12, 16", "19, 36", "40, 28", "36, 75", "76, 55", "72, 146",
        "142, 110", "192, 243", "459, 401"
    ])
    masks: Dict = dataclasses.field(default_factory=lambda: {
        5: [6, 7, 8],
        4: [3, 4, 5],
        3: [0, 1, 2]
    })
    path_scales: Dict = dataclasses.field(default_factory=lambda: {
        5: 32,
        4: 16,
        3: 8
    })
    x_y_scales: Dict = dataclasses.field(default_factory=lambda: {
        5: 1.05,
        4: 1.1,
        3: 1.2
    })


@dataclasses.dataclass
class YoloTinyv4Anchors(AnchorCFG):
    """RevNet config."""
    # Specifies the depth of RevNet.
    _boxes: List[str] = dataclasses.field(
        default_factory=lambda:
        ["10, 14", "23, 27", "37, 58", "81, 82", "135, 169", "344, 319"])
    masks: Dict = dataclasses.field(default_factory=lambda: {
        5: [3, 4, 5],
        4: [0, 1, 2]
    })
    path_scales: Dict = dataclasses.field(default_factory=lambda: {
        5: 32,
        4: 8
    })
    x_y_scales: Dict = dataclasses.field(default_factory=lambda: {
        5: 1.0,
        4: 1.0
    })


@dataclasses.dataclass
class YoloTinyv3Anchors(AnchorCFG):
    """RevNet config."""
    # Specifies the depth of RevNet.
    _boxes: List[str] = dataclasses.field(
        default_factory=lambda:
        ["10, 14", "23, 27", "37, 58", "81, 82", "135, 169", "344, 319"])
    masks: Dict = dataclasses.field(default_factory=lambda: {
        5: [3, 4, 5],
        3: [0, 1, 2]
    })
    path_scales: Dict = dataclasses.field(default_factory=lambda: {
        5: 32,
        3: 8
    })
    x_y_scales: Dict = dataclasses.field(default_factory=lambda: {
        5: 1.0,
        3: 1.0
    })


@dataclasses.dataclass
class Anchors(hyperparams.OneOfConfig):
    """Configuration for backbones.
    Attributes:
        type: 'str', type of backbone be used, one the of fields below.
        resnet: resnet backbone config.
        revnet: revnet backbone config.
        efficientnet: efficientnet backbone config.
        spinenet: spinenet backbone config.
        mobilenet: mobilenet backbone config.
    """
    type: Optional[str] = None
    v3: Yolov3Anchors = Yolov3Anchors()
    v4: Yolov4Anchors = Yolov4Anchors()
    tiny: YoloTinyv3Anchors = YoloTinyv3Anchors()
    tinyv4: YoloTinyv4Anchors = YoloTinyv4Anchors()


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
    tfds_name: str = 'coco'
    tfds_split: str = 'train'
    global_batch_size: int = 10
    is_training: bool = True
    dtype: str = 'float16'
    decoder = None
    parser: Parser = Parser()
    shuffle_buffer_size: int = 10000


@dataclasses.dataclass
class YoloHead(hyperparams.Config):
    version: str = "v4"
    name: str = "regular"


@dataclasses.dataclass
class YoloLossLayer(hyperparams.Config):
    iou_thresh: float = 0.45
    class_thresh: float = 0.45
    ignore_thresh: float = 0.7
    loss_type: str = "ciou"
    max_boxes: int = 200
    anchor_generation_scale: int = 416
    use_tie_breaker: bool = True
    use_nms: bool = True


@dataclasses.dataclass
class Yolov3(hyperparams.Config):
    backbone: backbones.Backbone = backbones.Backbone(
        type="darknet", darknet=backbones.DarkNet(model_id="darknet53"))
    head: YoloHead = YoloHead(version="v3", name="regular")


@dataclasses.dataclass
class Yolov3spp(hyperparams.Config):
    backbone: backbones.Backbone = backbones.Backbone(
        type="darknet", darknet=backbones.DarkNet(model_id="darknet53"))
    head: YoloHead = YoloHead(version="v3", name="spp")


@dataclasses.dataclass
class Yolov3tiny(hyperparams.Config):
    backbone: backbones.Backbone = backbones.Backbone(
        type="darknet", darknet=backbones.DarkNet(model_id="darknettiny"))
    head: YoloHead = YoloHead(version="v3", name="tiny")


@dataclasses.dataclass
class Yolov4(hyperparams.Config):
    backbone: backbones.Backbone = backbones.Backbone(
        type="darknet", darknet=backbones.DarkNet(model_id="cspdarknet53"))
    head: YoloHead = YoloHead(version="v4", name="regular")


@dataclasses.dataclass
class Yolov4tiny(hyperparams.Config):
    backbone: backbones.Backbone = backbones.Backbone(
        type="darknet", darknet=backbones.DarkNet(model_id="cspdarknettiny"))
    head: YoloHead = YoloHead(version="v4", name="tinyv4")



@dataclasses.dataclass
class YoloFamilySelector(hyperparams.OneOfConfig):
    type: Optional[str] = None
    v3: Yolov3 = Yolov3()
    v3_spp: Yolov3spp = Yolov3spp()
    v3_tiny: Yolov3tiny = Yolov3tiny()
    v4: Yolov4 = Yolov4()
    v4_tiny: Yolov4tiny = Yolov4tiny()


@dataclasses.dataclass
class Yolo(ModelConfig):
    num_classes: int = 80
    _input_size: Optional[List[int]] = None
    min_level: Optional[int] = None
    max_level: int = 5
    anchors: Anchors = Anchors(type="tiny")
    base: YoloFamilySelector = YoloFamilySelector(type="v3_tiny")
    decoder: YoloLossLayer = YoloLossLayer()
    norm_activation_backbone: common.NormActivation = common.NormActivation(
        activation="mish",
        use_sync_bn=False,
        norm_momentum=0.99,
        norm_epsilon=0.001)
    norm_activation_head: common.NormActivation = common.NormActivation(
        activation="leaky",
        use_sync_bn=False,
        norm_momentum=0.99,
        norm_epsilon=0.001)


# model task
@dataclasses.dataclass
class YoloTask(cfg.TaskConfig):
    model: Yolo = Yolo()
    train_data: DataConfig = DataConfig(is_training=True)
    validation_data: DataConfig = DataConfig(is_training=False)
    weight_decay: float = 5e-4
    annotation_file: Optional[str] = None
    gradient_clip_norm: float = 0.0
    per_category_metrics: bool = False


@exp_factory.register_config_factory('yolo_v4_coco')
def yolo_v4_coco() -> cfg.ExperimentConfig:
    """COCO object detection with YOLO."""
    train_batch_size = 4096
    eval_batch_size = 4096
    steps_per_epoch = IMAGENET_TRAIN_EXAMPLES // train_batch_size

    config = cfg.ExperimentConfig(
        runtime=cfg.RuntimeConfig(mixed_precision_dtype='float32'),
        task=YoloTask(model=Yolo(type='v4'),
                      train_data=DataConfig(input_path=os.path.join(
                          COCO_INPUT_PATH_BASE, 'train*'),
                                            is_training=True,
                                            global_batch_size=train_batch_size,
                                            parser=Parser()),
                      validation_data=DataConfig(
                          input_path=os.path.join(COCO_INPUT_PATH_BASE,
                                                  'val*'),
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


if __name__ == "__main__":
    config = YoloTask()
    print(config)

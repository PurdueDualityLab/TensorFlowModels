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
"""All necessary imports for registration."""

# pylint: disable=unused-import
from official.common import registry_imports

import yolo
from yolo.modeling.backbones import darknet
from yolo.modeling.backbones import swin
from yolo.modeling.decoders import yolo_decoder
# from yolo.modeling import factory

from yolo.configs import darknet_classification
from yolo.configs import yolo

# from yolo.configs.yolo import yolo_custom
from yolo.configs.darknet_classification import darknet_classification
from yolo.configs.darknet_classification import ImageClassificationTask
from yolo.configs.yolo import YoloTask

from yolo.tasks.image_classification import ImageClassificationTask
from yolo.tasks.yolo import YoloTask

from yolo.optimization import optimizer_factory
from yolo.optimization.configs import learning_rate_config, optimization_config, optimizer_config
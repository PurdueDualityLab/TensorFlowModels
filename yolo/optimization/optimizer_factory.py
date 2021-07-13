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

"""Optimizer factory class."""
from typing import Callable, Union


import gin
import tensorflow as tf
import tensorflow_addons.optimizers as tfa_optimizers
from official.modeling.optimization import configs

from yolo.optimization import SGDAccumulated, SGD
from official.modeling.optimization import optimizer_factory
from official.modeling.optimization import lr_schedule
from official.modeling.optimization.configs import optimization_config as opt_cfg


optimizer_factory.OPTIMIZERS_CLS.update({
    # 'sgd': tf.keras.optimizers.SGD,
    'sgd_dymo': SGD.SGD,
    'sgd_accum': SGDAccumulated.SGDAccumulated
})

OPTIMIZERS_CLS = optimizer_factory.OPTIMIZERS_CLS
LR_CLS = optimizer_factory.LR_CLS
WARMUP_CLS = optimizer_factory.WARMUP_CLS


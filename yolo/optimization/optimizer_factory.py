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

from yolo.optimization import (SGDAccumulated, SGDMomentumWarmup,
                               SGDMomentumWarmupW, ScaledYoloSGD)
from yolo.optimization import ema_optimizer
from official.modeling.optimization import optimizer_factory  #, ema_optimizer
from official.modeling.optimization import lr_schedule
from yolo.optimization.configs import optimization_config as opt_cfg

optimizer_factory.OPTIMIZERS_CLS.update({
    'sgd_dymo': SGDMomentumWarmup.SGDMomentumWarmup,
    'sgd_dymow': SGDMomentumWarmupW.SGDMomentumWarmupW,
    'sgd_accum': SGDAccumulated.SGDAccumulated,
    'scaled_sgd': ScaledYoloSGD.ScaledYoloSGD
})

OPTIMIZERS_CLS = optimizer_factory.OPTIMIZERS_CLS
LR_CLS = optimizer_factory.LR_CLS
WARMUP_CLS = optimizer_factory.WARMUP_CLS


class OptimizerFactory(optimizer_factory.OptimizerFactory):
  """Optimizer factory class.

  This class builds learning rate and optimizer based on an optimization config.
  To use this class, you need to do the following:
  (1) Define optimization config, this includes optimizer, and learning rate
      schedule.
  (2) Initialize the class using the optimization config.
  (3) Build learning rate.
  (4) Build optimizer.

  This is a typical example for using this class:
  params = {
        'optimizer': {
            'type': 'sgd',
            'sgd': {'momentum': 0.9}
        },
        'learning_rate': {
            'type': 'stepwise',
            'stepwise': {'boundaries': [10000, 20000],
                         'values': [0.1, 0.01, 0.001]}
        },
        'warmup': {
            'type': 'linear',
            'linear': {'warmup_steps': 500, 'warmup_learning_rate': 0.01}
        }
    }
  opt_config = OptimizationConfig(params)
  opt_factory = OptimizerFactory(opt_config)
  lr = opt_factory.build_learning_rate()
  optimizer = opt_factory.build_optimizer(lr)
  """

  def get_bias_lr_schedule(self, bias_lr):
    print(self._warmup_config)
    temp = self._warmup_config.warmup_learning_rate
    self._warmup_config.warmup_learning_rate = bias_lr
    lr = self.build_learning_rate()
    self._warmup_config.warmup_learning_rate = temp
    return lr

  @gin.configurable
  def build_optimizer(
      self,
      lr: Union[tf.keras.optimizers.schedules.LearningRateSchedule, float],
      postprocessor: Callable[[tf.keras.optimizers.Optimizer],
                              tf.keras.optimizers.Optimizer] = None):
    """Build optimizer.

    Builds optimizer from config. It takes learning rate as input, and builds
    the optimizer according to the optimizer config. Typically, the learning
    rate built using self.build_lr() is passed as an argument to this method.

    Args:
      lr: A floating point value, or a
        tf.keras.optimizers.schedules.LearningRateSchedule instance.
      postprocessor: An optional function for postprocessing the optimizer. It
        takes an optimizer and returns an optimizer.

    Returns:
      tf.keras.optimizers.Optimizer instance.
    """

    optimizer_dict = self._optimizer_config.as_dict()
    ## Delete clipnorm and clipvalue if None
    if optimizer_dict['clipnorm'] is None:
      del optimizer_dict['clipnorm']
    if optimizer_dict['clipvalue'] is None:
      del optimizer_dict['clipvalue']

    optimizer_dict['learning_rate'] = lr

    optimizer = OPTIMIZERS_CLS[self._optimizer_type](**optimizer_dict)

    if self._use_ema:
      optimizer = ema_optimizer.ExponentialMovingAverage(
          optimizer, **self._ema_config.as_dict())
    if postprocessor:
      optimizer = postprocessor(optimizer)
    assert isinstance(optimizer, tf.keras.optimizers.Optimizer), (
        'OptimizerFactory.build_optimizer returning a non-optimizer object: '
        '{}'.format(optimizer))

    return optimizer

  @gin.configurable
  def add_ema(self, optimizer):
    """Build optimizer.

    Builds optimizer from config. It takes learning rate as input, and builds
    the optimizer according to the optimizer config. Typically, the learning
    rate built using self.build_lr() is passed as an argument to this method.

    Args:
      lr: A floating point value, or a
        tf.keras.optimizers.schedules.LearningRateSchedule instance.
      postprocessor: An optional function for postprocessing the optimizer. It
        takes an optimizer and returns an optimizer.

    Returns:
      tf.keras.optimizers.Optimizer instance.
    """

    if self._use_ema:
      optimizer = ema_optimizer.ExponentialMovingAverage(
          optimizer, **self._ema_config.as_dict())
    return optimizer

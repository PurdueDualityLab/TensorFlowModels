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
"""Exponential moving average optimizer."""

from typing import List, Optional, Text

import tensorflow as tf
from tensorflow.python.keras import optimizer_v2
from official.modeling.optimization import ema_optimizer
import logging
# pylint: disable=protected-access


class ExponentialMovingAverage(ema_optimizer.ExponentialMovingAverage):
  """Optimizer that computes an exponential moving average of the variables.

  Empirically it has been found that using the moving average of the trained
  parameters of a deep network is better than using its trained parameters
  directly. This optimizer allows you to compute this moving average and swap
  the variables at save time so that any code outside of the training loop
  will use by default the average values instead of the original ones.

  Example of usage for training:
  ```python
  opt = tf.keras.optimizers.SGD(learning_rate)
  opt = ExponentialMovingAverage(opt)

  opt.shadow_copy(model)
  ```

  At test time, swap the shadow variables to evaluate on the averaged weights:
  ```python
  opt.swap_weights()
  # Test eval the model here
  opt.swap_weights()
  ```
  """

  def __init__(self,
               optimizer: tf.keras.optimizers.Optimizer,
               trainable_weights_only: bool = True,
               average_decay: float = 0.99,
               start_step: int = 0,
               dynamic_decay: bool = True,
               name: Text = 'ExponentialMovingAverage',
               **kwargs):
    """Construct a new ExponentialMovingAverage optimizer.

    Args:
      optimizer: `tf.keras.optimizers.Optimizer` that will be
        used to compute and apply gradients.
      trainable_weights_only: 'bool', if True, only model trainable weights will
        be updated. Otherwise, all model weights will be updated. This mainly
        affects batch normalization parameters.
      average_decay: float. Decay to use to maintain the moving averages
        of trained variables.
      start_step: int. What step to start the moving average.
      dynamic_decay: bool. Whether to change the decay based on the number
        of optimizer updates. Decay will start at 0.1 and gradually increase
        up to `average_decay` after each optimizer update. This behavior is
        similar to `tf.train.ExponentialMovingAverage` in TF 1.x.
      name: Optional name for the operations created when applying
        gradients. Defaults to "moving_average".
      **kwargs: keyword arguments. Allowed to be {`clipnorm`,
        `clipvalue`, `lr`, `decay`}.
    """
    super().__init__(
        optimizer=optimizer,
        trainable_weights_only=trainable_weights_only,
        average_decay=average_decay,
        start_step=start_step,
        dynamic_decay=dynamic_decay,
        name=name,
        **kwargs)
    logging.info("EMA is enabled.")
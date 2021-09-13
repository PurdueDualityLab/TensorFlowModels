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
import logging
from tensorflow_addons.optimizers import MovingAverage

# pylint: disable=protected-access


class ExponentialMovingAverage(MovingAverage):
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

  def __init__(
        self,
        optimizer,
        trainable_weights_only: bool = True,
        average_decay = 0.99,
        start_step: int = 0,
        dynamic_decay: bool = True,
        name: str = "MovingAverage",
        **kwargs):

    super().__init__(
      optimizer = optimizer, 
      average_decay = average_decay, 
      num_updates = None, 
      start_step = start_step, 
      dynamic_decay = dynamic_decay, 
      name = name
    ) 
    self._trainable_weights_only = trainable_weights_only

  def shadow_copy(self, model):
    """Creates shadow variables for the given model weights."""
    if self._trainable_weights_only:
      model_weights = model.trainable_variables
    else:
      model_weights = model.variables
    super().shadow_copy(model_weights)

    
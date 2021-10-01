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

"""Yolo models."""

import tensorflow as tf
from yolo.modeling.layers import nn_blocks

class Yolo(tf.keras.Model):
  """The YOLO model class."""

  def __init__(self,
               backbone=None,
               decoder=None,
               head=None,
               detection_generator=None,
               **kwargs):
    """Detection initialization function.
    Args:
      backbone: `tf.keras.Model` a backbone network.
      decoder: `tf.keras.Model` a decoder network.
      head: `RetinaNetHead`, the RetinaNet head.
      filter: the detection generator.
      **kwargs: keyword arguments to be passed.
    """
    super(Yolo, self).__init__(**kwargs)

    self._config_dict = {
        'backbone': backbone,
        'decoder': decoder,
        'head': head,
        'detection_generator': detection_generator
    }

    # model components
    self._backbone = backbone
    self._decoder = decoder
    self._head = head
    self._detection_generator = detection_generator
    self._fused = False
    return

  def call(self, inputs, training=False):
    maps = self._backbone(inputs)
    decoded_maps = self._decoder(maps)
    raw_predictions = self._head(decoded_maps)
    if training:
      return {"raw_output": raw_predictions}
    else:
      # Post-processing.
      predictions = self._detection_generator(raw_predictions)
      predictions.update({"raw_output": raw_predictions})
      return predictions

  @property
  def backbone(self):
    return self._backbone

  @property
  def decoder(self):
    return self._decoder

  @property
  def head(self):
    return self._head

  @property
  def detection_generator(self):
    return self._detection_generator

  def get_config(self):
    return self._config_dict

  @classmethod
  def from_config(cls, config):
    return cls(**config)

  def get_weight_groups(self, train_vars):
    """Sort the list of trainable variables into groups for optimization. 

    Args:
      train_vars: a list of tf.Variables that need to get sorted into their 
        respective groups.

    Returns:
      weights: a list of tf.Variables for the weights.
      bias: a list of tf.Variables for the bias.
      other: a list of tf.Variables for the other operations.
    """
    bias = []
    weights = []
    other = []
    for var in train_vars:
      if "bias" in var.name:
        bias.append(var)
      elif "beta" in var.name:
        bias.append(var)
      elif "kernel" in var.name or "weight" in var.name:
        weights.append(var)
      else:
        other.append(var)
    return weights, bias, other
  
  def fuse(self):
    print("Fusing Conv Batch Norm Layers.")
    if not self._fused:
      self._fused = True
      for layer in self.submodules:
        if isinstance(layer, nn_blocks.ConvBN):
          layer.fuse()
      self.summary()
    return 

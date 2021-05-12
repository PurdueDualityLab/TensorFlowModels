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
"""Tests for resnet."""

import numpy as np
import tensorflow as tf
# Import libraries
from absl.testing import parameterized
from tensorflow.python.distribute import combinations, strategy_combinations

# from yolo.modeling.backbones import darknet
from yolo.modeling.heads import yolo_head as heads


class YoloDecoderTest(parameterized.TestCase, tf.test.TestCase):

  def test_network_creation(self):
    """Test creation of ResNet family models."""
    tf.keras.backend.set_image_data_format('channels_last')
    input_shape = {
        '3': [1, 52, 52, 256],
        '4': [1, 26, 26, 512],
        '5': [1, 13, 13, 1024]
    }
    classes = 100
    bps = 3
    head = heads.YoloHead(classes=classes, boxes_per_level=bps)

    inputs = {}
    for key in input_shape.keys():
      inputs[key] = tf.ones(input_shape[key], dtype=tf.float32)

    endpoints = head(inputs)
    # print(endpoints)

    for key in endpoints.keys():
      ishape = input_shape[key]
      ishape[-1] = (classes + 5) * bps
      self.assertAllEqual(endpoints[key].shape.as_list(), ishape)

  def test_serialize_deserialize(self):
    # Create a network object that sets all of its config options.
    tf.keras.backend.set_image_data_format('channels_last')
    input_shape = {
        '3': [1, 52, 52, 256],
        '4': [1, 26, 26, 512],
        '5': [1, 13, 13, 1024]
    }
    classes = 100
    bps = 3
    head = heads.YoloHead(classes=classes, boxes_per_level=bps)

    inputs = {}
    for key in input_shape.keys():
      inputs[key] = tf.ones(input_shape[key], dtype=tf.float32)

    _ = head(inputs)

    a = head.get_config()

    b = heads.YoloHead.from_config(a)

    print(a)
    self.assertAllEqual(head.get_config(), b.get_config())


if __name__ == '__main__':
  from yolo.utils.run_utils import prep_gpu
  prep_gpu()
  tf.test.main()

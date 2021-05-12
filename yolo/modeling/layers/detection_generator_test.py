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
from yolo.modeling.layers import detection_generator as dg


class YoloDecoderTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(
      (True),
      (False),
  )
  def test_network_creation(self, nms):
    """Test creation of ResNet family models."""
    tf.keras.backend.set_image_data_format('channels_last')
    input_shape = {
        '3': [1, 52, 52, 255],
        '4': [1, 26, 26, 255],
        '5': [1, 13, 13, 255]
    }
    classes = 80
    masks = {'3': [0, 1, 2], '4': [3, 4, 5], '5': [6, 7, 8]}
    anchors = [[12.0, 19.0], [31.0, 46.0], [96.0, 54.0], [46.0, 114.0],
               [133.0, 127.0], [79.0, 225.0], [301.0, 150.0], [172.0, 286.0],
               [348.0, 340.0]]
    layer = dg.YoloLayer(masks, anchors, classes, max_boxes=10)

    inputs = {}
    for key in input_shape.keys():
      inputs[key] = tf.ones(input_shape[key], dtype=tf.float32)

    endpoints = layer(inputs)

    boxes = endpoints['bbox']
    classes = endpoints['classes']

    self.assertAllEqual(boxes.shape.as_list(), [1, 10, 4])
    self.assertAllEqual(classes.shape.as_list(), [1, 10])


if __name__ == '__main__':
  from yolo.utils.run_utils import prep_gpu
  prep_gpu()
  tf.test.main()

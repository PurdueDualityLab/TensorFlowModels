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

# Lint as: python3
"""Tests for yolo heads."""

# Import libraries
from absl.testing import parameterized
import tensorflow as tf

from official.vision.beta.projects.yolo.losses import yolo_loss


class YoloDecoderTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(
      (True),
      (False),
  )
  def test_loss_init(self, scaled):
    """Test creation of YOLO family models."""

    def inpdict(input_shape, dtype = tf.float32):
      inputs = {}
      for key in input_shape:
        inputs[key] = tf.ones(input_shape[key], dtype=dtype)
      return inputs

    tf.keras.backend.set_image_data_format('channels_last')
    input_shape = {
        '3': [1, 52, 52, 255],
        '4': [1, 26, 26, 255],
        '5': [1, 13, 13, 255]
    }
    classes = 80
    bps = 3
    masks = {'3': [0, 1, 2], '4': [3, 4, 5], '5': [6, 7, 8]}
    anchors = [[12.0, 19.0], [31.0, 46.0], [96.0, 54.0], [46.0, 114.0],
               [133.0, 127.0], [79.0, 225.0], [301.0, 150.0], [172.0, 286.0],
               [348.0, 340.0]]
    box_type = {key:"scaled" for key in masks.keys()}
    keys = ['3', '4', '5']
    path_strides = {key: 2**int(key) for key in keys}

    loss = yolo_loss.YoloLoss(
      keys, 
      classes, 
      anchors, 
      masks = masks, 
      path_strides=path_strides, 
      truth_thresholds={key: 1.0 for key in keys}, 
      ignore_thresholds={key: 0.7 for key in keys}, 
      loss_types = {key: "ciou" for key in keys},
      iou_normalizers = {key: 0.05 for key in keys},
      cls_normalizers = {key: 0.5 for key in keys},
      obj_normalizers = {key: 1.0 for key in keys},
      objectness_smooths = {key: 1.0 for key in keys},
      box_types = {key: "scaled" for key in keys},
      scale_xys = {key: 2.0 for key in keys},
      max_deltas = {key: 30.0 for key in keys},
      label_smoothing=0.0,
      use_scaled_loss=scaled,
      update_on_repeat=True
    )

    count = inpdict({
        '3': [1, 52, 52, 3, 1],
        '4': [1, 26, 26, 3, 1],
        '5': [1, 13, 13, 3, 1]
    })
    ind = inpdict({
        '3': [1, 300, 3],
        '4': [1, 300, 3],
        '5': [1, 300, 3]
    }, tf.int32)
    truths = inpdict({
        '3': [1, 300, 8],
        '4': [1, 300, 8],
        '5': [1, 300, 8]
    })
    boxes = tf.ones([1, 300, 4], dtype = tf.float32)
    classes = tf.ones([1, 300], dtype = tf.float32)

    gt = {
      "true_conf": count, 
      "inds": ind, 
      "upds":truths, 
      "bbox":boxes, 
      "classes":classes
    }

    loss_val, metric_loss, metric_dict = loss(gt, inpdict(input_shape))


if __name__ == '__main__':
  tf.test.main()

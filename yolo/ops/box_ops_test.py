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

"""Tests for box_ops.py."""

import tensorflow as tf
import numpy as np
from absl.testing import parameterized

from yolo.ops import box_ops

class BoxOpsTest(parameterized.TestCase, tf.test.TestCase):
    @parameterized.parameters(
        ([[0., 0., 1., 1.]], [[0., 0., 1., 1.]], [1.]),
        ([[0., 0., 0.5, 0.5]], [[0.5, 0.5, 1., 1.]], [0.]),
        ([[0., 0., 1, .75]], [[0, 0.25, 1., 1.]], [0.5])
    )
    def testcompute_iou(self,
                        box1,
                        box2,
                        target_val):
        #test IOU
        val = box_ops.compute_iou(tf.convert_to_tensor(box1),
            tf.convert_to_tensor(box2), yxyx = True)
        self.assertAllEqual(val, target_val)

    @parameterized.parameters(
        ([[0., 0., 1., 1.]], [[0., 0., 1., 1.]], [1.]),
        ([[0., 0., 0.5, 0.5]], [[0.5, 0.5, 1., 1.]], [-0.5]),
        ([[0., 0., 1, .75]], [[0., 0.25, 1., 1.]], [0.5])
    )
    def testcompute_giou(self,
                         box1,
                         box2,
                         target_val):
        #test GIOU
        _, val = box_ops.compute_giou(tf.convert_to_tensor(box1),
            tf.convert_to_tensor(box2), yxyx = True)
        self.assertAllEqual(val, target_val)

    @parameterized.parameters(
        ([[0., 0., 1., 1.]], [[0., 0., 1., 1.]], [1.]),
        ([[0., 0., 0.5, 0.5]], [[0.5, 0.5, 1., 1.]], [-0.25]),
        ([[0., 0., 1, .75]], [[0., 0.25, 1., 1.]], [0.46875])
    )
    def testcompute_diou(self, box1, box2, target_val):
        #test DIOU
        _, val = box_ops.compute_diou(tf.convert_to_tensor(box1),
            tf.convert_to_tensor(box2), yxyx = True)
        self.assertAllEqual(val, target_val)
    
    @parameterized.parameters(
        ([[0., 0., 1., 1.]], [[0., 0., 1., 1.]], [1.]),
        ([[0., 0., 0.5, 0.5]], [[0.5, 0.5, 1., 1.]], [-0.25]),
        ([[0., 0., 1, .75]], [[0., 0.25, 1., 1.]], [0.46875])
    )
    def testcompute_ciou(self,
                         box1,
                         box2,
                         target_val):
        #test CIOU
        _, val = box_ops.compute_ciou(tf.convert_to_tensor(box1),
            tf.convert_to_tensor(box2), yxyx = True)
        self.assertAllEqual(val, target_val)

    @parameterized.parameters(
        (4, 200),
        (16, 200),
    )
    def testyxyx_to_xcycwh(self,
                           batch_size,
                           num_instances):
        tensor = tf.convert_to_tensor(
            np.random.rand(batch_size, num_instances, 4))
        tensor = box_ops.yxyx_to_xcycwh(tensor)
        self.assertAllEqual(tf.shape(tensor).numpy(),
                            [batch_size, num_instances, 4])
    
    @parameterized.parameters(
        (4, 200),
        (16, 200),
    )
    def testxcycwh_to_yxyx(self,
                           batch_size,
                           num_instances):
        tensor = tf.convert_to_tensor(
            np.random.rand(batch_size, num_instances, 4))
        tensor = box_ops.xcycwh_to_yxyx(tensor)
        self.assertAllEqual(tf.shape(tensor).numpy(),
                            [batch_size, num_instances, 4])


if __name__ == '__main__':
  tf.test.main()

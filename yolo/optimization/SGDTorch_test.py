# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Test for SGDTorch."""

from yolo.optimization import SGDTorch

import tensorflow as tf


class SGDTorchTest(tf.test.TestCase):
    
    def testTorch(self):
        opt = SGDTorch.SGDTorch(momentum=0.5,
                                warmup_steps=1,
                                sim_torch=True)
        var0 = tf.Variable([1.0, 2.0], dtype = tf.float32)
        var1 = tf.Variable([3.0, 4.0], dtype = tf.float32)
        grad0 = tf.constant([0.1,0.1], dtype = tf.float32)
        grad1 = tf.constant([0.01,0.01], dtype = tf.float32)
        sgd_op = opt.apply_gradients(zip([grad0, grad1], [var0, var1]))
        self.assertAllClose([0.999, 1.999], self.evaluate(var0))
        self.assertAllClose([2.9999, 3.9999], self.evaluate(var1))
        sgd_op = opt.apply_gradients(zip([grad0, grad1], [var0, var1]))
        self.assertAllClose([0.9975, 1.9975], self.evaluate(var0))
        self.assertAllClose([2.99975, 3.99975], self.evaluate(var1))

    def testTF(self):
        opt = SGDTorch.SGDTorch(momentum=0.5,
                                warmup_steps=1,
                                sim_torch=False)
        var0 = tf.Variable([1.0, 2.0], dtype = tf.float32)
        var1 = tf.Variable([3.0, 4.0], dtype = tf.float32)
        grad0 = tf.constant([0.1,0.1], dtype = tf.float32)
        grad1 = tf.constant([0.01,0.01], dtype = tf.float32)
        sgd_op = opt.apply_gradients(zip([grad0, grad1], [var0, var1]))
        self.assertAllClose([0.999, 1.999], self.evaluate(var0))
        self.assertAllClose([2.9999, 3.9999], self.evaluate(var1))
        sgd_op = opt.apply_gradients(zip([grad0, grad1], [var0, var1]))
        self.assertAllClose([0.9975, 1.9975], self.evaluate(var0))
        self.assertAllClose([2.99975, 3.99975], self.evaluate(var1))


if __name__ == '__main__':
    tf.test.main()
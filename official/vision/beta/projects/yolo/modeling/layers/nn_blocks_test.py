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
from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from official.vision.beta.projects.yolo.modeling.layers import nn_blocks


class CSPConnectTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(('same', 224, 224, 64, 1),
                                  ('downsample', 224, 224, 64, 2))
  def test_pass_through(self, width, height, filters, mod):
    x = tf.keras.Input(shape=(width, height, filters))
    test_layer = nn_blocks.CSPRoute(filters=filters, filter_scale=mod)
    test_layer2 = nn_blocks.CSPConnect(filters=filters, filter_scale=mod)
    outx, px = test_layer(x)
    outx = test_layer2([outx, px])
    print(outx)
    print(outx.shape.as_list())
    self.assertAllEqual(
        outx.shape.as_list(),
        [None, np.ceil(width // 2),
         np.ceil(height // 2), (filters)])

  @parameterized.named_parameters(('same', 224, 224, 64, 1),
                                  ('downsample', 224, 224, 128, 2))
  def test_gradient_pass_though(self, filters, width, height, mod):
    loss = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.SGD()
    test_layer = nn_blocks.CSPRoute(filters, filter_scale=mod)
    path_layer = nn_blocks.CSPConnect(filters, filter_scale=mod)

    init = tf.random_normal_initializer()
    x = tf.Variable(
        initial_value=init(shape=(1, width, height, filters), dtype=tf.float32))
    y = tf.Variable(
        initial_value=init(
            shape=(1, int(np.ceil(width // 2)), int(np.ceil(height // 2)),
                   filters),
            dtype=tf.float32))

    with tf.GradientTape() as tape:
      x_hat, x_prev = test_layer(x)
      x_hat = path_layer([x_hat, x_prev])
      grad_loss = loss(x_hat, y)
    grad = tape.gradient(grad_loss, test_layer.trainable_variables)
    optimizer.apply_gradients(zip(grad, test_layer.trainable_variables))

    self.assertNotIn(None, grad)


class CSPRouteTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(('same', 224, 224, 64, 1),
                                  ('downsample', 224, 224, 64, 2))
  def test_pass_through(self, width, height, filters, mod):
    x = tf.keras.Input(shape=(width, height, filters))
    test_layer = nn_blocks.CSPRoute(filters=filters, filter_scale=mod)
    outx, _ = test_layer(x)
    print(outx)
    print(outx.shape.as_list())
    self.assertAllEqual(
        outx.shape.as_list(),
        [None, np.ceil(width // 2),
         np.ceil(height // 2), (filters / mod)])

  @parameterized.named_parameters(('same', 224, 224, 64, 1),
                                  ('downsample', 224, 224, 128, 2))
  def test_gradient_pass_though(self, filters, width, height, mod):
    loss = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.SGD()
    test_layer = nn_blocks.CSPRoute(filters, filter_scale=mod)
    path_layer = nn_blocks.CSPConnect(filters, filter_scale=mod)

    init = tf.random_normal_initializer()
    x = tf.Variable(
        initial_value=init(shape=(1, width, height, filters), dtype=tf.float32))
    y = tf.Variable(
        initial_value=init(
            shape=(1, int(np.ceil(width // 2)), int(np.ceil(height // 2)),
                   filters),
            dtype=tf.float32))

    with tf.GradientTape() as tape:
      x_hat, x_prev = test_layer(x)
      x_hat = path_layer([x_hat, x_prev])
      grad_loss = loss(x_hat, y)
    grad = tape.gradient(grad_loss, test_layer.trainable_variables)
    optimizer.apply_gradients(zip(grad, test_layer.trainable_variables))

    self.assertNotIn(None, grad)


class CSPStackTest(tf.test.TestCase, parameterized.TestCase):

  def build_layer(self, layer_type, filters, filter_scale, count, stack_type,
                  downsample):
    if stack_type is not None:
      layers = []
      if layer_type == 'residual':
        for _ in range(count):
          layers.append(
              nn_blocks.DarkResidual(
                  filters=filters // filter_scale, filter_scale=filter_scale))
      else:
        for _ in range(count):
          layers.append(nn_blocks.ConvBN(filters=filters))

      if stack_type == 'model':
        layers = tf.keras.Sequential(layers=layers)
    else:
      layers = None

    stack = nn_blocks.CSPStack(
        filters=filters,
        filter_scale=filter_scale,
        downsample=downsample,
        model_to_wrap=layers)
    return stack

  @parameterized.named_parameters(
      ('no_stack', 224, 224, 64, 2, 'residual', None, 0, True),
      ('residual_stack', 224, 224, 64, 2, 'residual', 'list', 2, True),
      ('conv_stack', 224, 224, 64, 2, 'conv', 'list', 3, False),
      ('callable_no_scale', 224, 224, 64, 1, 'residual', 'model', 5, False))
  def test_pass_through(self, width, height, filters, mod, layer_type,
                        stack_type, count, downsample):
    x = tf.keras.Input(shape=(width, height, filters))
    test_layer = self.build_layer(layer_type, filters, mod, count, stack_type,
                                  downsample)
    outx = test_layer(x)
    print(outx)
    print(outx.shape.as_list())
    if downsample:
      self.assertAllEqual(outx.shape.as_list(),
                          [None, width // 2, height // 2, filters])
    else:
      self.assertAllEqual(outx.shape.as_list(), [None, width, height, filters])

  @parameterized.named_parameters(
      ('no_stack', 224, 224, 64, 2, 'residual', None, 0, True),
      ('residual_stack', 224, 224, 64, 2, 'residual', 'list', 2, True),
      ('conv_stack', 224, 224, 64, 2, 'conv', 'list', 3, False),
      ('callable_no_scale', 224, 224, 64, 1, 'residual', 'model', 5, False))
  def test_gradient_pass_though(self, width, height, filters, mod, layer_type,
                                stack_type, count, downsample):
    loss = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.SGD()

    init = tf.random_normal_initializer()
    x = tf.Variable(
        initial_value=init(shape=(1, width, height, filters), dtype=tf.float32))

    if not downsample:
      y = tf.Variable(
          initial_value=init(
              shape=(1, width, height, filters), dtype=tf.float32))
    else:
      y = tf.Variable(
          initial_value=init(
              shape=(1, width // 2, height // 2, filters), dtype=tf.float32))
    test_layer = self.build_layer(layer_type, filters, mod, count, stack_type,
                                  downsample)

    with tf.GradientTape() as tape:
      x_hat = test_layer(x)
      grad_loss = loss(x_hat, y)
    grad = tape.gradient(grad_loss, test_layer.trainable_variables)
    optimizer.apply_gradients(zip(grad, test_layer.trainable_variables))

    self.assertNotIn(None, grad)


class ConvBNTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('valid', (3, 3), 'valid', (1, 1)), ('same', (3, 3), 'same', (1, 1)),
      ('downsample', (3, 3), 'same', (2, 2)), ('test', (1, 1), 'valid', (1, 1)))
  def test_pass_through(self, kernel_size, padding, strides):
    if padding == 'same':
      pad_const = 1
    else:
      pad_const = 0
    x = tf.keras.Input(shape=(224, 224, 3))
    test_layer = nn_blocks.ConvBN(
        filters=64,
        kernel_size=kernel_size,
        padding=padding,
        strides=strides,
        trainable=False)
    outx = test_layer(x)
    print(outx.shape.as_list())
    test = [
        None,
        int((224 - kernel_size[0] + (2 * pad_const)) / strides[0] + 1),
        int((224 - kernel_size[1] + (2 * pad_const)) / strides[1] + 1), 64
    ]
    print(test)
    self.assertAllEqual(outx.shape.as_list(), test)

  @parameterized.named_parameters(('filters', 3))
  def test_gradient_pass_though(self, filters):
    loss = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.SGD()
    with tf.device('/CPU:0'):
      test_layer = nn_blocks.ConvBN(filters, kernel_size=(3, 3), padding='same')

    init = tf.random_normal_initializer()
    x = tf.Variable(
        initial_value=init(shape=(1, 224, 224, 3), dtype=tf.float32))
    y = tf.Variable(
        initial_value=init(shape=(1, 224, 224, filters), dtype=tf.float32))

    with tf.GradientTape() as tape:
      x_hat = test_layer(x)
      grad_loss = loss(x_hat, y)
    grad = tape.gradient(grad_loss, test_layer.trainable_variables)
    optimizer.apply_gradients(zip(grad, test_layer.trainable_variables))
    self.assertNotIn(None, grad)


class DarkResidualTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(('same', 224, 224, 64, False),
                                  ('downsample', 223, 223, 32, True),
                                  ('oddball', 223, 223, 32, False))
  def test_pass_through(self, width, height, filters, downsample):
    mod = 1
    if downsample:
      mod = 2
    x = tf.keras.Input(shape=(width, height, filters))
    test_layer = nn_blocks.DarkResidual(filters=filters, downsample=downsample)
    outx = test_layer(x)
    print(outx)
    print(outx.shape.as_list())
    self.assertAllEqual(
        outx.shape.as_list(),
        [None, np.ceil(width / mod),
         np.ceil(height / mod), filters])

  @parameterized.named_parameters(('same', 64, 224, 224, False),
                                  ('downsample', 32, 223, 223, True),
                                  ('oddball', 32, 223, 223, False))
  def test_gradient_pass_though(self, filters, width, height, downsample):
    loss = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.SGD()
    test_layer = nn_blocks.DarkResidual(filters, downsample=downsample)

    if downsample:
      mod = 2
    else:
      mod = 1

    init = tf.random_normal_initializer()
    x = tf.Variable(
        initial_value=init(shape=(1, width, height, filters), dtype=tf.float32))
    y = tf.Variable(
        initial_value=init(
            shape=(1, int(np.ceil(width / mod)), int(np.ceil(height / mod)),
                   filters),
            dtype=tf.float32))

    with tf.GradientTape() as tape:
      x_hat = test_layer(x)
      grad_loss = loss(x_hat, y)
    grad = tape.gradient(grad_loss, test_layer.trainable_variables)
    optimizer.apply_gradients(zip(grad, test_layer.trainable_variables))

    self.assertNotIn(None, grad)


class DarkSppTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(('RouteProcessSpp', 224, 224, 3, [5, 9, 13]),
                                  ('test1', 300, 300, 10, [2, 3, 4, 5]),
                                  ('test2', 256, 256, 5, [10]))
  def test_pass_through(self, width, height, channels, sizes):
    x = tf.keras.Input(shape=(width, height, channels))
    test_layer = nn_blocks.SPP(sizes=sizes)
    outx = test_layer(x)
    self.assertAllEqual(outx.shape.as_list(),
                        [None, width, height, channels * (len(sizes) + 1)])
    return

  @parameterized.named_parameters(('RouteProcessSpp', 224, 224, 3, [5, 9, 13]),
                                  ('test1', 300, 300, 10, [2, 3, 4, 5]),
                                  ('test2', 256, 256, 5, [10]))
  def test_gradient_pass_though(self, width, height, channels, sizes):
    loss = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.SGD()
    test_layer = nn_blocks.SPP(sizes=sizes)

    init = tf.random_normal_initializer()
    x = tf.Variable(
        initial_value=init(
            shape=(1, width, height, channels), dtype=tf.float32))
    y = tf.Variable(
        initial_value=init(
            shape=(1, width, height, channels * (len(sizes) + 1)),
            dtype=tf.float32))

    with tf.GradientTape() as tape:
      x_hat = test_layer(x)
      grad_loss = loss(x_hat, y)
    grad = tape.gradient(grad_loss, test_layer.trainable_variables)
    optimizer.apply_gradients(zip(grad, test_layer.trainable_variables))

    self.assertNotIn(None, grad)
    return


class DarkRouteProcessTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('test1', 224, 224, 64, 7, False), ('test2', 223, 223, 32, 3, False),
      ('tiny', 223, 223, 16, 1, False), ('spp', 224, 224, 64, 7, False))
  def test_pass_through(self, width, height, filters, repetitions, spp):
    x = tf.keras.Input(shape=(width, height, filters))
    test_layer = nn_blocks.DarkRouteProcess(
        filters=filters, repetitions=repetitions, insert_spp=spp)
    outx = test_layer(x)
    self.assertLen(outx, 2, msg='len(outx) != 2')
    if repetitions == 1:
      filter_y1 = filters
    else:
      filter_y1 = filters // 2
    self.assertAllEqual(
        outx[1].shape.as_list(), [None, width, height, filter_y1])
    self.assertAllEqual(
        filters % 2,
        0,
        msg='Output of a DarkRouteProcess layer has an odd number of filters')
    self.assertAllEqual(outx[0].shape.as_list(), [None, width, height, filters])

  @parameterized.named_parameters(
      ('test1', 224, 224, 64, 7, False), ('test2', 223, 223, 32, 3, False),
      ('tiny', 223, 223, 16, 1, False), ('spp', 224, 224, 64, 7, False))
  def test_gradient_pass_though(self, width, height, filters, repetitions, spp):
    loss = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.SGD()
    test_layer = nn_blocks.DarkRouteProcess(
        filters=filters, repetitions=repetitions, insert_spp=spp)

    if repetitions == 1:
      filter_y1 = filters
    else:
      filter_y1 = filters // 2

    init = tf.random_normal_initializer()
    x = tf.Variable(
        initial_value=init(shape=(1, width, height, filters), dtype=tf.float32))
    y_0 = tf.Variable(
        initial_value=init(shape=(1, width, height, filters), dtype=tf.float32))
    y_1 = tf.Variable(
        initial_value=init(
            shape=(1, width, height, filter_y1), dtype=tf.float32))

    with tf.GradientTape() as tape:
      x_hat_0, x_hat_1 = test_layer(x)
      grad_loss_0 = loss(x_hat_0, y_0)
      grad_loss_1 = loss(x_hat_1, y_1)
    grad = tape.gradient([grad_loss_0, grad_loss_1],
                         test_layer.trainable_variables)
    optimizer.apply_gradients(zip(grad, test_layer.trainable_variables))

    self.assertNotIn(None, grad)
    return


if __name__ == '__main__':
  tf.test.main()

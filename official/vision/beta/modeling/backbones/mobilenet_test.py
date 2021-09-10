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
"""Tests for MobileNet."""

import itertools
# Import libraries

from absl.testing import parameterized
import tensorflow as tf

from official.vision.beta.modeling.backbones import mobilenet


class MobileNetTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(
      'MobileNetV1',
      'MobileNetV2',
      'MobileNetV3Large',
      'MobileNetV3Small',
      'MobileNetV3EdgeTPU',
      'MobileNetMultiAVG',
      'MobileNetMultiMAX',
  )
  def test_serialize_deserialize(self, model_id):
    # Create a network object that sets all of its config options.
    kwargs = dict(
        model_id=model_id,
        filter_size_scale=1.0,
        stochastic_depth_drop_rate=None,
        use_sync_bn=False,
        kernel_initializer='VarianceScaling',
        kernel_regularizer=None,
        bias_regularizer=None,
        norm_momentum=0.99,
        norm_epsilon=0.001,
        output_stride=None,
        min_depth=8,
        divisible_by=8,
        regularize_depthwise=False,
        finegrain_classification_mode=True
    )
    network = mobilenet.MobileNet(**kwargs)

    expected_config = dict(kwargs)
    self.assertEqual(network.get_config(), expected_config)

    # Create another network object from the first object's config.
    new_network = mobilenet.MobileNet.from_config(network.get_config())

    # Validate that the config can be forced to JSON.
    _ = new_network.to_json()

    # If the serialization was successful, the new config should match the old.
    self.assertAllEqual(network.get_config(), new_network.get_config())

  @parameterized.parameters(
      itertools.product(
          [1, 3],
          [
              'MobileNetV1',
              'MobileNetV2',
              'MobileNetV3Large',
              'MobileNetV3Small',
              'MobileNetV3EdgeTPU',
              'MobileNetMultiAVG',
              'MobileNetMultiMAX',
          ],
      ))
  def test_input_specs(self, input_dim, model_id):
    """Test different input feature dimensions."""
    tf.keras.backend.set_image_data_format('channels_last')

    input_specs = tf.keras.layers.InputSpec(shape=[None, None, None, input_dim])
    network = mobilenet.MobileNet(model_id=model_id, input_specs=input_specs)

    inputs = tf.keras.Input(shape=(128, 128, input_dim), batch_size=1)
    _ = network(inputs)

  @parameterized.parameters(
      itertools.product(
          [
              'MobileNetV1',
              'MobileNetV2',
              'MobileNetV3Large',
              'MobileNetV3Small',
              'MobileNetV3EdgeTPU',
              'MobileNetMultiAVG',
              'MobileNetMultiMAX',
          ],
          [32, 224],
      ))
  def test_mobilenet_creation(self, model_id,
                              input_size):
    """Test creation of MobileNet family models."""
    tf.keras.backend.set_image_data_format('channels_last')

    mobilenet_layers = {
        # The number of filters of layers having outputs been collected
        # for filter_size_scale = 1.0
        'MobileNetV1': [128, 256, 512, 1024],
        'MobileNetV2': [24, 32, 96, 320],
        'MobileNetV3Small': [16, 24, 48, 96],
        'MobileNetV3Large': [24, 40, 112, 160],
        'MobileNetV3EdgeTPU': [32, 48, 96, 192],
        'MobileNetMultiMAX': [32, 64, 128, 160],
        'MobileNetMultiAVG': [32, 64, 160, 192],
    }

    network = mobilenet.MobileNet(model_id=model_id,
                                  filter_size_scale=1.0)

    inputs = tf.keras.Input(shape=(input_size, input_size, 3), batch_size=1)
    endpoints = network(inputs)

    for idx, num_filter in enumerate(mobilenet_layers[model_id]):
      self.assertAllEqual(
          [1, input_size / 2 ** (idx+2), input_size / 2 ** (idx+2), num_filter],
          endpoints[str(idx+2)].shape.as_list())

  @parameterized.parameters(
      itertools.product(
          [
              'MobileNetV1',
              'MobileNetV2',
              'MobileNetV3Large',
              'MobileNetV3Small',
              'MobileNetV3EdgeTPU',
              'MobileNetMultiAVG',
              'MobileNetMultiMAX',
          ],
          [1.0, 0.75],
      ))
  def test_mobilenet_scaling(self, model_id,
                             filter_size_scale):
    """Test for creation of a MobileNet classifier."""
    mobilenet_params = {
        ('MobileNetV1', 1.0): 3228864,
        ('MobileNetV1', 0.75): 1832976,
        ('MobileNetV2', 1.0): 2257984,
        ('MobileNetV2', 0.75): 1382064,
        ('MobileNetV3Large', 1.0): 4226432,
        ('MobileNetV3Large', 0.75): 2731616,
        ('MobileNetV3Small', 1.0): 1529968,
        ('MobileNetV3Small', 0.75): 1026552,
        ('MobileNetV3EdgeTPU', 1.0): 2849312,
        ('MobileNetV3EdgeTPU', 0.75): 1737288,
        ('MobileNetMultiAVG', 1.0): 3704416,
        ('MobileNetMultiAVG', 0.75): 2349704,
        ('MobileNetMultiMAX', 1.0): 3174560,
        ('MobileNetMultiMAX', 0.75): 2045816,
    }

    input_size = 224
    network = mobilenet.MobileNet(model_id=model_id,
                                  filter_size_scale=filter_size_scale)
    self.assertEqual(network.count_params(),
                     mobilenet_params[(model_id, filter_size_scale)])

    inputs = tf.keras.Input(shape=(input_size, input_size, 3), batch_size=1)
    _ = network(inputs)

if __name__ == '__main__':
  tf.test.main()

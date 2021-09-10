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
"""Tests for SpineNet."""
# Import libraries
from absl.testing import parameterized
import tensorflow as tf

from official.vision.beta.modeling.backbones import spinenet_mobile


class SpineNetMobileTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(
      (128, 0.6, 1, 0.0, 24),
      (128, 0.65, 1, 0.2, 40),
      (256, 1.0, 1, 0.2, 48),
  )
  def test_network_creation(self, input_size, filter_size_scale, block_repeats,
                            se_ratio, endpoints_num_filters):
    """Test creation of SpineNet models."""
    min_level = 3
    max_level = 7

    tf.keras.backend.set_image_data_format('channels_last')

    input_specs = tf.keras.layers.InputSpec(
        shape=[None, input_size, input_size, 3])
    model = spinenet_mobile.SpineNetMobile(
        input_specs=input_specs,
        min_level=min_level,
        max_level=max_level,
        endpoints_num_filters=endpoints_num_filters,
        resample_alpha=se_ratio,
        block_repeats=block_repeats,
        filter_size_scale=filter_size_scale,
        init_stochastic_depth_rate=0.2,
    )

    inputs = tf.keras.Input(shape=(input_size, input_size, 3), batch_size=1)
    endpoints = model(inputs)

    for l in range(min_level, max_level + 1):
      self.assertIn(str(l), endpoints.keys())
      self.assertAllEqual(
          [1, input_size / 2**l, input_size / 2**l, endpoints_num_filters],
          endpoints[str(l)].shape.as_list())

  def test_serialize_deserialize(self):
    # Create a network object that sets all of its config options.
    kwargs = dict(
        min_level=3,
        max_level=7,
        endpoints_num_filters=256,
        se_ratio=0.2,
        expand_ratio=6,
        block_repeats=1,
        filter_size_scale=1.0,
        init_stochastic_depth_rate=0.2,
        use_sync_bn=False,
        activation='relu',
        norm_momentum=0.99,
        norm_epsilon=0.001,
        kernel_initializer='VarianceScaling',
        kernel_regularizer=None,
        bias_regularizer=None,
        use_keras_upsampling_2d=False,
    )
    network = spinenet_mobile.SpineNetMobile(**kwargs)

    expected_config = dict(kwargs)
    self.assertEqual(network.get_config(), expected_config)

    # Create another network object from the first object's config.
    new_network = spinenet_mobile.SpineNetMobile.from_config(
        network.get_config())

    # Validate that the config can be forced to JSON.
    _ = new_network.to_json()

    # If the serialization was successful, the new config should match the old.
    self.assertAllEqual(network.get_config(), new_network.get_config())


if __name__ == '__main__':
  tf.test.main()

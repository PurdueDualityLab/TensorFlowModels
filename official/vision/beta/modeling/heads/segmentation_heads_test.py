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
"""Tests for segmentation_heads.py."""

# Import libraries
from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from official.vision.beta.modeling.heads import segmentation_heads


class SegmentationHeadTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(
      (2, 'pyramid_fusion'),
      (3, 'pyramid_fusion'),
  )
  def test_forward(self, level, feature_fusion):
    head = segmentation_heads.SegmentationHead(
        num_classes=10, level=level, feature_fusion=feature_fusion)
    backbone_features = {
        '3': np.random.rand(2, 128, 128, 16),
        '4': np.random.rand(2, 64, 64, 16),
    }
    decoder_features = {
        '3': np.random.rand(2, 128, 128, 16),
        '4': np.random.rand(2, 64, 64, 16),
    }
    logits = head(backbone_features, decoder_features)

    if level in decoder_features:
      self.assertAllEqual(logits.numpy().shape, [
          2, decoder_features[str(level)].shape[1],
          decoder_features[str(level)].shape[2], 10
      ])

  def test_serialize_deserialize(self):
    head = segmentation_heads.SegmentationHead(num_classes=10, level=3)
    config = head.get_config()
    new_head = segmentation_heads.SegmentationHead.from_config(config)
    self.assertAllEqual(head.get_config(), new_head.get_config())

if __name__ == '__main__':
  tf.test.main()

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
"""Tests for anchor.py."""

# Import libraries
from absl.testing import parameterized
import numpy as np
import tensorflow as tf
from official.vision.beta.ops import anchor


class AnchorTest(parameterized.TestCase, tf.test.TestCase):

  # The set of parameters are tailored for the MLPerf configuration, where
  # the number of anchors is 495132, rpn_batch_size_per_im=256, and
  # rpn_fg_fraction=0.5.
  @parameterized.parameters(
      (512, 25, 25, 25, 25, (512, 512)),
      (512, 25, 25, 25, 25, (512, 640)),
      (512, 25, 25, 25, 25, (640, 512)),
      (495132, 100, 100, 100, 100, (512, 512)),
      (495132, 200, 100, 128, 100, (512, 512)),
      (495132, 100, 120, 100, 120, (512, 512)),
      (495132, 100, 200, 100, 156, (512, 512)),
      (495132, 200, 200, 128, 128, (512, 512)),
  )
  def testAnchorRpnSample(self, num_anchors, num_positives,
                          num_negatives, expected_positives,
                          expected_negatives, image_size):
    match_results_np = np.empty([num_anchors])
    match_results_np.fill(-2)
    match_results_np[:num_positives] = 0
    match_results_np[num_positives:num_positives + num_negatives] = -1
    match_results = tf.convert_to_tensor(value=match_results_np, dtype=tf.int32)
    anchor_labeler = anchor.RpnAnchorLabeler(
        match_threshold=0.7,
        unmatched_threshold=0.3,
        rpn_batch_size_per_im=256,
        rpn_fg_fraction=0.5)
    rpn_sample_op = anchor_labeler._get_rpn_samples(match_results)
    labels = [v.numpy() for v in rpn_sample_op]
    self.assertLen(labels[0], num_anchors)
    positives = np.sum(np.array(labels[0]) == 1)
    negatives = np.sum(np.array(labels[0]) == 0)
    self.assertEqual(positives, expected_positives)
    self.assertEqual(negatives, expected_negatives)

  @parameterized.parameters(
      # Single scale anchor.
      (5, 5, 1, [1.0], 2.0,
       [[-16, -16, 48, 48], [-16, 16, 48, 80],
        [16, -16, 80, 48], [16, 16, 80, 80]]),
      # Multi scale anchor.
      (5, 6, 1, [1.0], 2.0,
       [[-16, -16, 48, 48], [-16, 16, 48, 80],
        [16, -16, 80, 48], [16, 16, 80, 80], [-32, -32, 96, 96]]),
      # # Multi aspect ratio anchor.
      (6, 6, 1, [1.0, 4.0, 0.25], 2.0,
       [[-32, -32, 96, 96], [-0, -96, 64, 160], [-96, -0, 160, 64]]),

  )
  def testAnchorGeneration(self, min_level, max_level, num_scales,
                           aspect_ratios, anchor_size, expected_boxes):
    image_size = [64, 64]
    anchors = anchor.Anchor(min_level, max_level, num_scales, aspect_ratios,
                            anchor_size, image_size)
    boxes = anchors.boxes.numpy()
    self.assertEqual(expected_boxes, boxes.tolist())

  @parameterized.parameters(
      # Single scale anchor.
      (5, 5, 1, [1.0], 2.0,
       [[-16, -16, 48, 48], [-16, 16, 48, 80],
        [16, -16, 80, 48], [16, 16, 80, 80]]),
      # Multi scale anchor.
      (5, 6, 1, [1.0], 2.0,
       [[-16, -16, 48, 48], [-16, 16, 48, 80],
        [16, -16, 80, 48], [16, 16, 80, 80], [-32, -32, 96, 96]]),
      # # Multi aspect ratio anchor.
      (6, 6, 1, [1.0, 4.0, 0.25], 2.0,
       [[-32, -32, 96, 96], [-0, -96, 64, 160], [-96, -0, 160, 64]]),

  )
  def testAnchorGenerationWithImageSizeAsTensor(self,
                                                min_level,
                                                max_level,
                                                num_scales,
                                                aspect_ratios,
                                                anchor_size,
                                                expected_boxes):
    image_size = tf.constant([64, 64], tf.int32)
    anchors = anchor.Anchor(min_level, max_level, num_scales, aspect_ratios,
                            anchor_size, image_size)
    boxes = anchors.boxes.numpy()
    self.assertEqual(expected_boxes, boxes.tolist())

  @parameterized.parameters(
      (3, 6, 2, [1.0], 2.0),
  )
  def testLabelAnchors(self, min_level, max_level, num_scales,
                       aspect_ratios, anchor_size):
    input_size = [512, 512]
    ground_truth_class_id = 2

    # The matched anchors are the anchors used as ground truth and the anchors
    # at the next octave scale on the same location.
    expected_anchor_locations = [[0, 0, 0], [0, 0, 1]]
    anchor_gen = anchor.build_anchor_generator(min_level, max_level, num_scales,
                                               aspect_ratios, anchor_size)
    anchor_boxes = anchor_gen(input_size)
    anchor_labeler = anchor.AnchorLabeler()

    # Uses the first anchors as ground truth. The ground truth should map to
    # two anchors with two intermediate scales at the same location.
    gt_boxes = anchor_boxes['3'][0:1, 0, 0:4]
    gt_classes = tf.constant([[ground_truth_class_id]], dtype=tf.float32)
    (cls_targets, box_targets, _,
     box_weights) = anchor_labeler.label_anchors(
         anchor_boxes, gt_boxes, gt_classes)

    for k, v in cls_targets.items():
      cls_targets[k] = v.numpy()
    for k, v in box_targets.items():
      box_targets[k] = v.numpy()
    box_weights = box_weights.numpy()

    anchor_locations = np.vstack(
        np.where(cls_targets[str(min_level)] > -1)).transpose()
    self.assertAllClose(expected_anchor_locations, anchor_locations)
    # Two anchor boxes on min_level got matched to the gt_boxes.
    self.assertAllClose(tf.reduce_sum(box_weights), 2)

  @parameterized.parameters(
      (3, 7, [.5, 1., 2.], 2, 8, (256, 256)),
      (3, 8, [1.], 3, 32, (512, 512)),
      (3, 3, [1.], 2, 4, (32, 32)),
  )
  def testEquivalentResult(self, min_level, max_level, aspect_ratios,
                           num_scales, anchor_size, image_size):
    anchor_gen = anchor.build_anchor_generator(
        min_level=min_level,
        max_level=max_level,
        num_scales=num_scales,
        aspect_ratios=aspect_ratios,
        anchor_size=anchor_size)
    anchors = anchor_gen(image_size)
    expected_anchor_gen = anchor.Anchor(min_level, max_level, num_scales,
                                        aspect_ratios, anchor_size, image_size)

    expected_anchors = expected_anchor_gen.multilevel_boxes
    for k in expected_anchors.keys():
      self.assertAllClose(expected_anchors[k], anchors[k])


if __name__ == '__main__':
  tf.test.main()

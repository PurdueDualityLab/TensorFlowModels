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

"""Data parser and processing for Panoptic Mask R-CNN."""

import tensorflow as tf

from official.vision.beta.dataloaders import maskrcnn_input
from official.vision.beta.dataloaders import tf_example_decoder
from official.vision.beta.ops import preprocess_ops


class TfExampleDecoder(tf_example_decoder.TfExampleDecoder):
  """Tensorflow Example proto decoder."""

  def __init__(self, regenerate_source_id, mask_binarize_threshold):
    super(TfExampleDecoder, self).__init__(
        include_mask=True,
        regenerate_source_id=regenerate_source_id,
        mask_binarize_threshold=None)
    self._segmentation_keys_to_features = {
        'image/segmentation/class/encoded':
            tf.io.FixedLenFeature((), tf.string, default_value='')
    }

  def decode(self, serialized_example):
    decoded_tensors = super(TfExampleDecoder, self).decode(serialized_example)
    segmentation_parsed_tensors = tf.io.parse_single_example(
        serialized_example, self._segmentation_keys_to_features)
    segmentation_mask = tf.io.decode_image(
        segmentation_parsed_tensors['image/segmentation/class/encoded'],
        channels=1)
    segmentation_mask.set_shape([None, None, 1])
    decoded_tensors.update({'groundtruth_segmentation_mask': segmentation_mask})
    return decoded_tensors


class Parser(maskrcnn_input.Parser):
  """Parser to parse an image and its annotations into a dictionary of tensors."""

  def __init__(self,
               output_size,
               min_level,
               max_level,
               num_scales,
               aspect_ratios,
               anchor_size,
               rpn_match_threshold=0.7,
               rpn_unmatched_threshold=0.3,
               rpn_batch_size_per_im=256,
               rpn_fg_fraction=0.5,
               aug_rand_hflip=False,
               aug_scale_min=1.0,
               aug_scale_max=1.0,
               skip_crowd_during_training=True,
               max_num_instances=100,
               mask_crop_size=112,
               segmentation_resize_eval_groundtruth=True,
               segmentation_groundtruth_padded_size=None,
               segmentation_ignore_label=255,
               dtype='float32'):
    """Initializes parameters for parsing annotations in the dataset.

    Args:
      output_size: `Tensor` or `list` for [height, width] of output image. The
        output_size should be divided by the largest feature stride 2^max_level.
      min_level: `int` number of minimum level of the output feature pyramid.
      max_level: `int` number of maximum level of the output feature pyramid.
      num_scales: `int` number representing intermediate scales added
        on each level. For instance, num_scales=2 adds one additional
        intermediate anchor scales [2^0, 2^0.5] on each level.
      aspect_ratios: `list` of float numbers representing the aspect raito
        anchors added on each level. The number indicates the ratio of width to
        height. For instance, aspect_ratios=[1.0, 2.0, 0.5] adds three anchors
        on each scale level.
      anchor_size: `float` number representing the scale of size of the base
        anchor to the feature stride 2^level.
      rpn_match_threshold: `float`, match threshold for anchors in RPN.
      rpn_unmatched_threshold: `float`, unmatched threshold for anchors in RPN.
      rpn_batch_size_per_im: `int` for batch size per image in RPN.
      rpn_fg_fraction: `float` for forground fraction per batch in RPN.
      aug_rand_hflip: `bool`, if True, augment training with random
        horizontal flip.
      aug_scale_min: `float`, the minimum scale applied to `output_size` for
        data augmentation during training.
      aug_scale_max: `float`, the maximum scale applied to `output_size` for
        data augmentation during training.
      skip_crowd_during_training: `bool`, if True, skip annotations labeled with
        `is_crowd` equals to 1.
      max_num_instances: `int` number of maximum number of instances in an
        image. The groundtruth data will be padded to `max_num_instances`.
      mask_crop_size: the size which groundtruth mask is cropped to.
      segmentation_resize_eval_groundtruth: `bool`, if True, eval groundtruth
        masks are resized to output_size.
      segmentation_groundtruth_padded_size: `Tensor` or `list` for [height,
        width]. When resize_eval_groundtruth is set to False, the groundtruth
        masks are padded to this size.
      segmentation_ignore_label: `int` the pixel with ignore label will not used
        for training and evaluation.
      dtype: `str`, data type. One of {`bfloat16`, `float32`, `float16`}.
    """
    super(Parser, self).__init__(
        output_size=output_size,
        min_level=min_level,
        max_level=max_level,
        num_scales=num_scales,
        aspect_ratios=aspect_ratios,
        anchor_size=anchor_size,
        rpn_match_threshold=rpn_match_threshold,
        rpn_unmatched_threshold=rpn_unmatched_threshold,
        rpn_batch_size_per_im=rpn_batch_size_per_im,
        rpn_fg_fraction=rpn_fg_fraction,
        aug_rand_hflip=False,
        aug_scale_min=aug_scale_min,
        aug_scale_max=aug_scale_max,
        skip_crowd_during_training=skip_crowd_during_training,
        max_num_instances=max_num_instances,
        include_mask=True,
        mask_crop_size=mask_crop_size,
        dtype=dtype)

    self.aug_rand_hflip = aug_rand_hflip
    self._segmentation_resize_eval_groundtruth = segmentation_resize_eval_groundtruth
    if (not segmentation_resize_eval_groundtruth) and (
        segmentation_groundtruth_padded_size is None):
      raise ValueError(
          'segmentation_groundtruth_padded_size ([height, width]) needs to be'
          'specified when segmentation_resize_eval_groundtruth is False.')
    self._segmentation_groundtruth_padded_size = segmentation_groundtruth_padded_size
    self._segmentation_ignore_label = segmentation_ignore_label

  def _parse_train_data(self, data):
    """Parses data for training.

    Args:
      data: the decoded tensor dictionary from TfExampleDecoder.

    Returns:
      image: image tensor that is preproessed to have normalized value and
        dimension [output_size[0], output_size[1], 3]
      labels: a dictionary of tensors used for training. The following describes
        {key: value} pairs in the dictionary.
        image_info: a 2D `Tensor` that encodes the information of the image and
          the applied preprocessing. It is in the format of
          [[original_height, original_width], [scaled_height, scaled_width]],
        anchor_boxes: ordered dictionary with keys
          [min_level, min_level+1, ..., max_level]. The values are tensor with
          shape [height_l, width_l, 4] representing anchor boxes at each level.
        rpn_score_targets: ordered dictionary with keys
          [min_level, min_level+1, ..., max_level]. The values are tensor with
          shape [height_l, width_l, anchors_per_location]. The height_l and
          width_l represent the dimension of class logits at l-th level.
        rpn_box_targets: ordered dictionary with keys
          [min_level, min_level+1, ..., max_level]. The values are tensor with
          shape [height_l, width_l, anchors_per_location * 4]. The height_l and
          width_l represent the dimension of bounding box regression output at
          l-th level.
        gt_boxes: Groundtruth bounding box annotations. The box is represented
           in [y1, x1, y2, x2] format. The coordinates are w.r.t the scaled
           image that is fed to the network. The tennsor is padded with -1 to
           the fixed dimension [self._max_num_instances, 4].
        gt_classes: Groundtruth classes annotations. The tennsor is padded
          with -1 to the fixed dimension [self._max_num_instances].
        gt_masks: Groundtruth masks cropped by the bounding box and
          resized to a fixed size determined by mask_crop_size.
        gt_segmentation_mask: Groundtruth mask for segmentation head, this is
          resized to a fixed size determined by output_size.
        gt_segmentation_valid_mask: Binary mask that marks the pixels that
          are supposed to be used in computing the segmentation loss while
          training.
    """
    segmentation_mask = data['groundtruth_segmentation_mask']

    # Flips image randomly during training.
    if self.aug_rand_hflip:
      masks = data['groundtruth_instance_masks']
      image_mask = tf.concat([data['image'], segmentation_mask], axis=2)

      image_mask, boxes, masks = preprocess_ops.random_horizontal_flip(
          image_mask, data['groundtruth_boxes'], masks)

      segmentation_mask = image_mask[:, :, -1:]
      image = image_mask[:, :, :-1]

      data['image'] = image
      data['boxes'] = boxes
      data['masks'] = masks

    image, labels = super(Parser, self)._parse_train_data(data)

    image_info = labels['image_info']
    image_scale = image_info[2, :]
    offset = image_info[3, :]

    segmentation_mask = tf.reshape(
        segmentation_mask, shape=[1, data['height'], data['width']])
    segmentation_mask = tf.cast(segmentation_mask, tf.float32)

    # Pad label and make sure the padded region assigned to the ignore label.
    # The label is first offset by +1 and then padded with 0.
    segmentation_mask += 1
    segmentation_mask = tf.expand_dims(segmentation_mask, axis=3)
    segmentation_mask = preprocess_ops.resize_and_crop_masks(
        segmentation_mask, image_scale, self._output_size, offset)
    segmentation_mask -= 1
    segmentation_mask = tf.where(
        tf.equal(segmentation_mask, -1),
        self._segmentation_ignore_label * tf.ones_like(segmentation_mask),
        segmentation_mask)
    segmentation_mask = tf.squeeze(segmentation_mask, axis=0)
    segmentation_valid_mask = tf.not_equal(
        segmentation_mask, self._segmentation_ignore_label)

    labels.update({
        'gt_segmentation_mask': segmentation_mask,
        'gt_segmentation_valid_mask': segmentation_valid_mask})

    return image, labels

  def _parse_eval_data(self, data):
    """Parses data for evaluation.

    Args:
      data: the decoded tensor dictionary from TfExampleDecoder.

    Returns:
      A dictionary of {'images': image, 'labels': labels} where
        image: image tensor that is preproessed to have normalized value and
          dimension [output_size[0], output_size[1], 3]
        labels: a dictionary of tensors used for training. The following
          describes {key: value} pairs in the dictionary.
          source_ids: Source image id. Default value -1 if the source id is
            empty in the groundtruth annotation.
          image_info: a 2D `Tensor` that encodes the information of the image
            and the applied preprocessing. It is in the format of
            [[original_height, original_width], [scaled_height, scaled_width]],
          anchor_boxes: ordered dictionary with keys
            [min_level, min_level+1, ..., max_level]. The values are tensor with
            shape [height_l, width_l, 4] representing anchor boxes at each
            level.
    """
    segmentation_mask = tf.cast(
        data['groundtruth_segmentation_mask'], tf.float32)
    segmentation_mask = tf.reshape(
        segmentation_mask, shape=[1, data['height'], data['width'], 1])
    segmentation_mask += 1

    image, labels = super(Parser, self)._parse_eval_data(data)

    if self._segmentation_resize_eval_groundtruth:
      # Resizes eval masks to match input image sizes. In that case, mean IoU
      # is computed on output_size not the original size of the images.
      image_info = labels['image_info']
      image_scale = image_info[2, :]
      offset = image_info[3, :]
      segmentation_mask = preprocess_ops.resize_and_crop_masks(
          segmentation_mask, image_scale, self._output_size, offset)
    else:
      segmentation_mask = tf.image.pad_to_bounding_box(
          segmentation_mask, 0, 0,
          self._segmentation_groundtruth_padded_size[0],
          self._segmentation_groundtruth_padded_size[1])

    segmentation_mask -= 1
    # Assign ignore label to the padded region.
    segmentation_mask = tf.where(
        tf.equal(segmentation_mask, -1),
        self._segmentation_ignore_label * tf.ones_like(segmentation_mask),
        segmentation_mask)
    segmentation_mask = tf.squeeze(segmentation_mask, axis=0)
    segmentation_valid_mask = tf.not_equal(
        segmentation_mask, self._segmentation_ignore_label)

    labels['groundtruths'].update({
        'gt_segmentation_mask': segmentation_mask,
        'gt_segmentation_valid_mask': segmentation_valid_mask})
    return image, labels

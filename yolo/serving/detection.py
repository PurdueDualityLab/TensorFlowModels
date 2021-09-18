
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
"""Detection input and model functions for serving/inference."""

import tensorflow as tf

from official.vision.beta import configs
from yolo.modeling import factory
from yolo.ops import preprocessing_ops
from official.vision.beta.ops import box_ops
from official.vision.beta.ops import preprocess_ops
from official.vision.beta.serving import export_base

def letterbox(image, desired_size, letter_box = True):
  """Letter box an image for image serving."""

  with tf.name_scope('letter_box'):
    image_size = tf.cast(tf.shape(image)[0:2], tf.float32)

    scaled_size = desired_size
    if letterbox:
      scale = tf.minimum(
          scaled_size[0] / image_size[0], scaled_size[1] / image_size[1])
    else:
      scale = 1.0
    scaled_size = tf.round(image_size * scale)

    # Computes 2D image_scale.
    image_scale = scaled_size / image_size
    image_offset = tf.cast((desired_size - scaled_size) * 0.5, tf.int32)
    offset = (scaled_size - desired_size) * 0.5
    scaled_image = tf.image.resize(image, 
                        tf.cast(scaled_size, tf.int32), 
                        method = 'nearest')

    output_image = tf.image.pad_to_bounding_box(
        scaled_image, image_offset[0], image_offset[1], 
                      desired_size[0], desired_size[1])

    
    image_info = tf.stack([
        image_size,
        tf.cast(desired_size, dtype=tf.float32),
        image_scale,
        tf.cast(offset, tf.float32)])
    return output_image, image_info

class DetectionModule(export_base.ExportModule):
  """Detection Module."""

  def _build_model(self):
    input_shape = [self._batch_size] + self._input_image_size + [3]
    input_specs = tf.keras.layers.InputSpec(shape=input_shape)
    model = factory.build_yolo(input_specs=input_specs, 
                               model_config=self.params.task.model)
    model.build(input_shape)
    return model

  def _build_inputs(self, image):
    """Builds detection model inputs for serving."""
    data_params = self.params.task.validation_data.parser
    # Normalizes image with mean and std pixel values.
    image = image / 255

    (image, image_info) = letterbox(
        image, self._input_image_size, letter_box = data_params.letter_box)
    return image, image_info

  def serve(self, images: tf.Tensor):
    """Cast image to float and run inference.

    Args:
      images: uint8 Tensor of shape [batch_size, None, None, 3]
    Returns:
      Tensor holding detection output logits.
    """
    model_params = self.params.task.model
    with tf.device('cpu:0'):
      images = tf.cast(images, dtype=tf.float32)

      # Tensor Specs for map_fn outputs (images, anchor_boxes, and image_info).
      images_spec = tf.TensorSpec(shape=self._input_image_size + [3],
                                  dtype=tf.float32)

      image_info_spec = tf.TensorSpec(shape=[4, 2], dtype=tf.float32)

      images, image_info = tf.nest.map_structure(
          tf.identity,
          tf.map_fn(
              self._build_inputs,
              elems=images,
              fn_output_signature=(images_spec, image_info_spec),
              parallel_iterations=32))

    input_image_shape = image_info[:, 1, :]

    # To overcome keras.Model extra limitation to save a model with layers that
    # have multiple inputs, we use `model.call` here to trigger the forward
    # path. Note that, this disables some keras magics happens in `__call__`.
    detections = self.model(images, training=False)

    final_outputs = {
      'detection_boxes': detections['bbox'],
      'detection_scores': detections['classes'],
      'detection_classes': detections['confidence'],
      'num_detections': detections['num_detections']
    }

    final_outputs.update({'image_info': image_info})
    return final_outputs

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
"""Detection input and model functions for serving/inference."""

import tensorflow as tf

from official.vision.beta.modeling import factory
from official.vision.beta.ops import preprocess_ops
from official.vision.beta.serving import export_base


MEAN_RGB = (0.485 * 255, 0.456 * 255, 0.406 * 255)
STDDEV_RGB = (0.229 * 255, 0.224 * 255, 0.225 * 255)


class ClassificationModule(export_base.ExportModule):
  """classification Module."""

  def build_model(self, skip_logits_layer=False):
    input_specs = tf.keras.layers.InputSpec(
        shape=[self._batch_size] + self._input_image_size + [3])

    self._model = factory.build_classification_model(
        input_specs=input_specs,
        model_config=self._params.task.model,
        l2_regularizer=None,
        skip_logits_layer=skip_logits_layer)

    return self._model

  def _build_inputs(self, image):
    """Builds classification model inputs for serving."""
    # Center crops and resizes image.
    image = preprocess_ops.center_crop_image(image)

    image = tf.image.resize(
        image, self._input_image_size, method=tf.image.ResizeMethod.BILINEAR)

    image = tf.reshape(
        image, [self._input_image_size[0], self._input_image_size[1], 3])

    # Normalizes image with mean and std pixel values.
    image = preprocess_ops.normalize_image(image,
                                           offset=MEAN_RGB,
                                           scale=STDDEV_RGB)
    return image

  def _run_inference_on_image_tensors(self, images):
    """Cast image to float and run inference.

    Args:
      images: uint8 Tensor of shape [batch_size, None, None, 3]
    Returns:
      Tensor holding classification output logits.
    """
    with tf.device('cpu:0'):
      images = tf.cast(images, dtype=tf.float32)

      images = tf.nest.map_structure(
          tf.identity,
          tf.map_fn(
              self._build_inputs, elems=images,
              fn_output_signature=tf.TensorSpec(
                  shape=self._input_image_size + [3], dtype=tf.float32),
              parallel_iterations=32
              )
          )

    logits = self._model(images, training=False)

    return dict(outputs=logits)

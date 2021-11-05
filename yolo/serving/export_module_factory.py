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

"""Factory for vision export modules."""

from typing import List, Optional

import tensorflow as tf

from official.core import config_definitions as cfg
from official.vision.beta import configs
from official.vision.beta.dataloaders import classification_input
from official.vision.beta.modeling import factory
from official.vision.beta.serving import export_base_v2 as export_base
from official.vision.beta.serving import export_utils
from yolo.configs.yolo import YoloTask, yolo
from yolo.modeling import factory as yolo_factory

def create_yolo_export_module(params, input_type, batch_size, input_image_size, num_channels):
    input_signature = export_utils.get_image_input_signatures(
      input_type, batch_size, input_image_size, num_channels)
    input_specs = tf.keras.layers.InputSpec(
        shape=[batch_size] + input_image_size + [num_channels])
    l2_weight_decay = params.task.weight_decay / 2.0
    l2_regularizer = (
        tf.keras.regularizers.l2(l2_weight_decay) if l2_weight_decay else None)
    model,_ = yolo_factory.build_yolo(input_specs, params.task.model, l2_regularizer)
    

    def preprocess_fn(inputs):
      image_tensor = export_utils.parse_image(inputs, input_type,
                                              input_image_size, num_channels)
      # If input_type is `tflite`, do not apply image preprocessing.
      if input_type == 'tflite':
        return image_tensor
      if len(tf.shape(image_tensor)) == 3:
        image_tensor = tf.expand_dims(image_tensor, axis=0)
      def preprocess_image_fn(inputs):
          image = tf.cast(inputs, dtype=tf.float32)
          image = tf.image.resize(
        image, input_image_size, method=tf.image.ResizeMethod.BILINEAR)
          image = image / 255.
          image.set_shape(input_image_size + [num_channels])
          return image

      images = tf.map_fn(
          preprocess_image_fn, elems=image_tensor,
          fn_output_signature=tf.TensorSpec(
              shape=input_image_size + [num_channels],
              dtype=tf.float32))

      return images
    def postprocess_fn(predictions):
      del predictions['raw_output']
      return predictions

    export_module = export_base.ExportModule(params,
                                           model=model,
                                           input_signature=input_signature,
                                           preprocessor=preprocess_fn,
                                           postprocessor=postprocess_fn)
    return export_module


def create_classification_export_module(params: cfg.ExperimentConfig,
                                        input_type: str,
                                        batch_size: int,
                                        input_image_size: List[int],
                                        num_channels: int = 3):
  """Creats classification export module."""
  input_signature = export_utils.get_image_input_signatures(
      input_type, batch_size, input_image_size, num_channels)
  input_specs = tf.keras.layers.InputSpec(
      shape=[batch_size] + input_image_size + [num_channels])
  model = factory.build_classification_model(
      input_specs=input_specs,
      model_config=params.task.model,
      l2_regularizer=None)

  def preprocess_fn(inputs):
    image_tensor = export_utils.parse_image(inputs, input_type,
                                            input_image_size, num_channels)
    # If input_type is `tflite`, do not apply image preprocessing.
    if input_type == 'tflite':
      return image_tensor

    def preprocess_image_fn(inputs):
      return classification_input.Parser.inference_fn(
          inputs, input_image_size, num_channels)

    images = tf.map_fn(
        preprocess_image_fn, elems=image_tensor,
        fn_output_signature=tf.TensorSpec(
            shape=input_image_size + [num_channels],
            dtype=tf.float32))

    return images

  def postprocess_fn(logits):
    probs = tf.nn.softmax(logits)
    return {'logits': logits, 'probs': probs}

  export_module = export_base.ExportModule(params,
                                           model=model,
                                           input_signature=input_signature,
                                           preprocessor=preprocess_fn,
                                           postprocessor=postprocess_fn)
  return export_module


def get_export_module(params: cfg.ExperimentConfig,
                      input_type: str,
                      batch_size: Optional[int],
                      input_image_size: List[int],
                      num_channels: int = 3) -> export_base.ExportModule:
  """Factory for export modules."""
  print(type(params.task))
  if isinstance(params.task,
                configs.image_classification.ImageClassificationTask):
    export_module = create_classification_export_module(
        params, input_type, batch_size, input_image_size, num_channels)
  elif isinstance(params.task, YoloTask):
    export_module = create_yolo_export_module(params, input_type, batch_size, input_image_size,num_channels)
  else:
    raise ValueError('Export module not implemented for {} task.'.format(
        type(params.task)))
  return export_module

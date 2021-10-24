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
"""Feature Pyramid Network and Path Aggregation variants used in YOLO."""
from typing import Mapping, Optional, Union

from numpy.core.numeric import outer
from official.modeling import hyperparams

import tensorflow as tf
from yolo.modeling.layers import nn_blocks, attention_fpn_blocks
from official.vision.beta.modeling.decoders import factory


@factory.register_decoder_builder('tbifpn_decoder')
def build_tbifpn_decoder(
    input_specs: Mapping[str, tf.TensorShape],
    model_config: hyperparams.Config,
    l2_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
    **kwargs) -> Union[None, tf.keras.Model, tf.keras.layers.Layer]:
  """Builds Yolo FPN/PAN decoder from a config.

  Args:
    input_specs: A `dict` of input specifications. A dictionary consists of
      {level: TensorShape} from a backbone.
    model_config: A OneOfConfig. Model config.
    l2_regularizer: A `tf.keras.regularizers.Regularizer` instance. Default to
      None.
    **kwargs: Additional kwargs arguments.

  Returns:
    A `tf.keras.Model` instance of the Yolo FPN/PAN decoder.
  """
  decoder_cfg = model_config.decoder.get()
  norm_activation_config = model_config.norm_activation

  activation = (
      decoder_cfg.activation if decoder_cfg.activation != 'same' else
      norm_activation_config.activation)

  model = attention_fpn_blocks.TBiFPN(
    input_specs, 
    use_patch_expansion = decoder_cfg.use_patch_expansion, 
    fpn_only=decoder_cfg.fpn_only, 
    repititions=decoder_cfg.repititions, 
    include_detokenization=decoder_cfg.include_detokenization, 
    use_separable_conv=decoder_cfg.use_separable_conv, 
    window_size=decoder_cfg.window_size, 
    token_size=decoder_cfg.token_size, 
    mlp_ratio=decoder_cfg.mlp_ratio, 
    kernel_size=decoder_cfg.kernel_size, 
    use_sync_bn=norm_activation_config.use_sync_bn,
    norm_momentum=norm_activation_config.norm_momentum,
    norm_epsilon=norm_activation_config.norm_epsilon,
    conv_activation=activation,
    kernel_regularizer=l2_regularizer, 
    shift = decoder_cfg.shift,
    expansion_kernel_size = decoder_cfg.expansion_kernel_size, 
  )
  return model


if __name__ == "__main__":
  inputs = {"3": [2, 80, 80, 256], "4": [2, 40, 40, 512], "5": [2, 20, 20, 1024]}

  built = {}
  for k, s in inputs.items():
    built[k] = tf.ones(inputs[k])

  m = TBiFPN(inputs, repititions=2)
  m.build(inputs)

  m(built)
  m.summary()

      



# @factory.register_decoder_builder('yolo_decoder')
# def build_yolo_decoder(input_specs: Mapping[str, tf.TensorShape], 
#                        model_config: hyperparams.Config, 
#                        l2_regularizer: tf.keras.regularizers.Regularizer = None, 
#                        **kwargs) -> Union[None, tf.keras.Model, tf.keras.layers.Layer]:
#   """Builds Yolo FPN/PAN decoder from a config.

#   Args:
#     input_specs: A `dict` of input specifications. A dictionary consists of
#       {level: TensorShape} from a backbone.
#     model_config: A OneOfConfig. Model config.
#     l2_regularizer: A `tf.keras.regularizers.Regularizer` instance. Default to
#       None.

#   Returns:
#     A `tf.keras.Model` instance of the Yolo FPN/PAN decoder.
#   """
#   decoder_cfg = model_config.decoder.get()
#   norm_activation_config = model_config.norm_activation

#   activation = (
#       decoder_cfg.activation
#       if decoder_cfg.activation != "same" else
#       norm_activation_config.activation)

#   if decoder_cfg.version is None:  # custom yolo
#     raise Exception("decoder version cannot be None, specify v3 or v4")

#   if decoder_cfg.version not in YOLO_MODELS:
#     raise Exception(
#         "unsupported model version please select from {v3, v4}, \n\n \
#         or specify a custom decoder config using YoloDecoder in you yaml")

#   if decoder_cfg.type == None:
#     decoder_cfg.type = "regular"

#   if decoder_cfg.type not in YOLO_MODELS[decoder_cfg.version]:
#     raise Exception("unsupported model type please select from \
#         {yolo_model.YOLO_MODELS[decoder_cfg.version].keys()},\
#         \n\n or specify a custom decoder config using YoloDecoder in you yaml")

#   base_model = YOLO_MODELS[decoder_cfg.version][decoder_cfg.type]

#   cfg_dict = decoder_cfg.as_dict()
#   for key in base_model:
#     if cfg_dict[key] is not None:
#       base_model[key] = cfg_dict[key]

#   base_dict = dict(
#       activation=activation,
#       use_spatial_attention=decoder_cfg.use_spatial_attention,
#       use_separable_conv=decoder_cfg.use_separable_conv,
#       use_sync_bn=norm_activation_config.use_sync_bn,
#       norm_momentum=norm_activation_config.norm_momentum,
#       norm_epsilon=norm_activation_config.norm_epsilon,
#       kernel_regularizer=l2_regularizer)

#   base_model.update(base_dict)
#   model = YoloDecoder(input_specs, **base_model)
#   return model
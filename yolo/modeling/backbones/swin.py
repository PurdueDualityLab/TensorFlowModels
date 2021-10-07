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
"""Contains definitions of Swin Backbone Networks."""
from typing import List
import tensorflow as tf
import math
import numpy as np

from official.modeling import hyperparams
from official.vision.beta.modeling.backbones import factory
from yolo.modeling.layers import attention_blocks

@tf.keras.utils.register_keras_serializable(package='yolo')
class SwinTransformer(tf.keras.Model):

  def __init__(self, 
               input_specs=tf.keras.layers.InputSpec(shape=[None, 448, 448, 3]),
               min_level = None, 
               max_level = None, 
               embed_dims = 96, 
               depths = [2, 2, 6, 2], 
               num_heads = [3, 6, 12, 24], 
               window_size = 7, 
               patch_size = 4, 
               mlp_ratio = 4, 
               ignore_shifts = False, 
               qkv_bias = True, 
               qk_scale = None, 
               dropout = 0.0, 
               attention_dropout = 0.0, 
               drop_path = 0.1, 
               absolute_positional_embed = False, 
               norm_layer = 'layer_norm',
               patch_norm = True, 
               activation = 'gelu',
               normalize_endpoints = True,
               kernel_initializer='TruncatedNormal',
               kernel_regularizer=None,
               bias_initializer='zeros',
               bias_regularizer=None, 
               **kwargs):

    if kernel_initializer == 'TruncatedNormal':
      kernel_initializer = tf.keras.initializers.TruncatedNormal(
                                      mean=0.0, stddev=0.02, seed=None)

    self._input_shape = input_specs
    self._min_level = min_level
    self._max_level = max_level

    self._embed_dims = embed_dims
    self._depths = depths
    self._num_layers = len(depths)
    self._num_heads = num_heads

    if not isinstance(window_size, list):
      self._window_size = [window_size] * self._num_layers
    else:
      self._window_size = window_size
    self._patch_size = patch_size
    self._patch_norm = patch_norm
    self._mlp_ratio = mlp_ratio
    self._qkv_bias = qkv_bias
    self._qk_scale = qk_scale
    self._ignore_shifts = ignore_shifts

    self._dropout = dropout
    self._attention_dropout = attention_dropout
    stochastic_drops = np.linspace(0, drop_path, num = sum(self._depths))
    self._drop_path = stochastic_drops.tolist()

    if attention_blocks.USE_SYNC_BN:
      self._norm_layer_key = "sync_batch_norm"
    else:
      self._norm_layer_key = "batch_norm"

    self._absolute_positional_embed=absolute_positional_embed
    self._activation = activation
    self._normalize_endpoints = normalize_endpoints

    # init and regularizer
    self._kernel_initializer = kernel_initializer
    self._bias_initializer = bias_initializer
    self._kernel_regularizer = kernel_regularizer
    self._bias_regularizer = bias_regularizer
    self._init_args = dict(
      kernel_initializer = self._kernel_initializer,
      bias_initializer = self._bias_initializer,
      kernel_regularizer = self._kernel_regularizer ,
      bias_regularizer = self._bias_regularizer,
    )

    inputs = tf.keras.layers.Input(shape=self._input_shape.shape[1:])
    output = self._build_struct(inputs)
    super().__init__(inputs=inputs, outputs=output, name="swin_transformer")

  def _build_struct(self, inputs):
    endpoints = dict() 
    outputs = dict()
    output_specs = dict()

    level = 0 

    embeddings = attention_blocks.PatchEmbed(
                        patch_size=self._patch_size, 
                        embed_dimentions=self._embed_dims, 
                        norm_layer=self._norm_layer_key if self._patch_norm else None, 
                        absolute_positional_embed=self._absolute_positional_embed,
                        activation=None, 
                        kernel_initializer = 'VarianceScaling',
                        bias_initializer = self._bias_initializer,
                        kernel_regularizer = self._kernel_regularizer ,
                        bias_regularizer = self._bias_regularizer)(inputs)
    level += int(math.log2(self._patch_size))
    
    x = tf.keras.layers.Dropout(self._dropout)(embeddings)
    for i in range(self._num_layers):
      dpr = self._drop_path[sum(self._depths[:i]):sum(self._depths[:i + 1])]
      x_output, x = attention_blocks.SwinTransformerBlock(
          ignore_shifts=self._ignore_shifts, 
          depth=self._depths[i], 
          num_heads=self._num_heads[i], 
          window_size=self._window_size[i], 
          mlp_ratio=self._mlp_ratio, 
          qkv_bias=self._qkv_bias, 
          qk_scale=self._qk_scale, 
          dropout=self._dropout, 
          attention_dropout=self._attention_dropout, 
          drop_path=dpr, 
          norm_layer=self._norm_layer_key, 
          downsample='patch_and_merge' if (i < self._num_layers - 1) else None, 
          activation=self._activation, 
          **self._init_args)(x)
      outputs[str(level)] = x_output
      level += 1

    min_key = int(min(outputs.keys()))
    max_key = int(max(outputs.keys()))

    if self._min_level is None:
      self._min_level = max_key
    if self._max_level is None:
      self._max_level = max_key
    
    self._min_level = max(min_key, int(self._min_level))
    for i in range(self._min_level, self._max_level + 1):
      x = outputs[str(i)]
      if self._normalize_endpoints:
        # endpoints[str(i)] = attention_blocks._get_norm_fn(self._norm_layer)()(x)
        endpoints[str(i)] = attention_blocks._get_norm_fn(self._norm_layer_key)()(x)
      else:
        endpoints[str(i)] = x
      output_specs[str(i)] = endpoints[str(i)].get_shape()

    self._output_specs = output_specs
    return endpoints

  @property
  def input_specs(self):
    return self._input_shape

  @property
  def output_specs(self):
    return self._output_specs

  def get_config(self):
    layer_config = dict(
      input_specs=self._input_shape,
      min_level=self._min_level,
      max_level=self._max_level,
      embed_dims=self._embed_dims,
      depths=self._depths,
      num_heads=self._num_heads, 
      window_size=self._window_size,
      patch_size=self._patch_size,
      mlp_ratio=self._mlp_ratio,
      qkv_bias=self._qkv_bias,
      qk_scale=self._qk_scale,
      dropout=self._dropout,
      attention_dropout=self._attention_dropout,
      drop_path=self._drop_path,
      normalize_endpoints=self._normalize_endpoints,
      absolute_positional_embed=self._absolute_positional_embed,
      activation=self._activation,
      kernel_regularizer=self._kernel_regularizer
    )
    return layer_config

@factory.register_backbone_builder('swin')
def build_swin(
    input_specs: tf.keras.layers.InputSpec,
    backbone_config: hyperparams.Config,
    norm_activation_config: hyperparams.Config,
    l2_regularizer: tf.keras.regularizers.Regularizer = None) -> tf.keras.Model:
  """Builds swin."""

  backbone_config = backbone_config.get()
  model = SwinTransformer(
      input_specs=input_specs,
      min_level=backbone_config.min_level,
      max_level=backbone_config.max_level,
      embed_dims=backbone_config.embed_dims,
      depths=backbone_config.depths,
      num_heads=backbone_config.num_heads, 
      window_size=backbone_config.window_size,
      patch_size=backbone_config.patch_size,
      mlp_ratio=backbone_config.mlp_ratio,
      qkv_bias=backbone_config.qkv_bias,
      qk_scale=backbone_config.qk_scale,
      dropout=backbone_config.dropout,
      attention_dropout=backbone_config.attention_dropout,
      drop_path=backbone_config.drop_path,
      normalize_endpoints=backbone_config.normalize_endpoints,
      absolute_positional_embed=backbone_config.absolute_positional_embed,
      activation=norm_activation_config.activation,
      kernel_regularizer=l2_regularizer, 
      ignore_shifts = backbone_config.ignore_shifts)
  model.summary()
  return model

if __name__ == "__main__":
  # # yolo p5 
  # shape = [896, 896, 3]
  # input_specs = tf.keras.layers.InputSpec(shape=[None]+shape)
  # model = SwinTransformer(
  #   input_specs = input_specs, 
  #   min_level = 3, 
  #   max_level = None, 
  #   patch_size = 4, 
  #   embed_dims = 96, 
  #   window_size = [7, 7, 7, 7], 
  #   depths = [2, 2, 6, 2], 
  #   num_heads = [3, 6, 12, 24], 
  #   down_sample_all=False
  # )

  # model.summary()

  # # yolo csp
  # shape = [640, 640, 3]
  # input_specs = tf.keras.layers.InputSpec(shape=[None]+shape)
  # model = SwinTransformer(
  #   input_specs = input_specs, 
  #   min_level = 3, 
  #   max_level = None, 
  #   patch_size = 2, 
  #   embed_dims = 96, 
  #   window_size = [8, 8, 8, 8], 
  #   depths = [2, 2, 6, 2], 
  #   num_heads = [3, 6, 12, 24], 
  #   down_sample_all=True
  # )

  # yolo csp - small
  # shape = [640, 640, 3]
  # input_specs = tf.keras.layers.InputSpec(shape=[None]+shape)
  # model = SwinTransformer(
  #   input_specs = input_specs, 
  #   min_level = 3, 
  #   max_level = None, 
  #   patch_size = 4, 
  #   embed_dims = 96, 
  #   window_size = [8, 8, 8], #, 4], 
  #   depths = [2, 2, 6], #, 2], 
  #   num_heads = [3, 6, 12], #, 24], 
  #   down_sample_all=True
  # )


  # # yolo csp - smaller
  # shape = [448, 448, 3]
  # input_specs = tf.keras.layers.InputSpec(shape=[None]+shape)
  # model = SwinTransformer(
  #   input_specs = input_specs, 
  #   min_level = 3, 
  #   max_level = None, 
  #   patch_size = 4, 
  #   embed_dims = 96, 
  #   window_size = [7, 7, 7, 7],  
  #   depths = [2, 2, 6, 2], 
  #   num_heads = [3, 6, 12, 24], 
  #   down_sample_all=False, 
  #   dense_embeddings=False, 
  # )

  # model.summary()

  # yolo csp - smaller
  # shape = [512, 512, 3]
  # input_specs = tf.keras.layers.InputSpec(shape=[None]+shape)
  # model = SwinTransformer(
  #   input_specs = input_specs, 
  #   min_level = 3, 
  #   max_level = None, 
  #   patch_size = 4, 
  #   embed_dims = 96, 
  #   window_size = [8, 8, 8, 8],  
  #   depths = [2, 2, 6, 2], 
  #   num_heads = [3, 6, 12, 24], 
  # )

  # model.summary()
  shape = [640, 640, 3]
  input_specs = tf.keras.layers.InputSpec(shape=[None]+shape)
  model = SwinTransformer(
    input_specs = input_specs, 
    min_level = 3, 
    max_level = None, 
    patch_size = 4, 
    embed_dims = 96, 
    window_size = [7, 7, 7, 7],  
    depths = [2, 2, 6, 2], 
    num_heads = [3, 6, 12, 24], 
    down_sample_all=False, 
    dense_embeddings=False, 
  )
  model.summary()

  input1 = tf.ones([1] + shape)

  output = model(input1)
  print({k:v.get_shape() for k,v in output.items()})
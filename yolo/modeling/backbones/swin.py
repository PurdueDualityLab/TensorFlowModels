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
               min_level = 3, 
               max_level = None, 
               patch_size = 4, 
               window_size = 7, 
               embed_dims = 96, 
               patch_norm = True, 
               depths = [2, 2, 6, 2], 
               num_heads = [3, 6, 12, 24], 
               mlp_ratio = 4, 
               use_bias_qkv = True, 
               qk_scale = None, 

               dense_embeddings = False, 
               absolute_positional_embed = False, 
               down_sample_all = False, 
               normalize_endpoints = True, 

               drop = 0.0, 
               attention_drop = 0.0, 
               drop_path = 0.1, 
               
               kernel_initializer='VarianceScaling',
               kernel_regularizer=None,
               bias_initializer='zeros',
               bias_regularizer=None, 
               activation = 'gelu',
               **kwargs):

    if kernel_initializer == 'TruncatedNormal':
      kernel_initializer = tf.keras.initializers.TruncatedNormal(
                                      mean=0.0, stddev=0.02, seed=None)

    self._input_shape = input_specs
    self._min_level = min_level
    self._max_level = max_level

    self._patch_size = patch_size
    self._patch_norm = patch_norm
    self._window_size = window_size
    self._embed_dims = embed_dims

    self._depths = depths
    self._num_heads = num_heads

    self._mlp_ratio = mlp_ratio
    self._use_bias_qkv = use_bias_qkv
    self._qk_scale = qk_scale
    
    self._drop = drop 
    self._attention_drop = attention_drop
    self._drop_path = drop_path
    self._normalize_endpoints = normalize_endpoints
    self._norm_fn = tf.keras.layers.LayerNormalization

    # init and regularizer
    self._kernel_initializer = kernel_initializer
    self._bias_initializer = bias_initializer
    self._kernel_regularizer = kernel_regularizer
    self._bias_regularizer = bias_regularizer

    self._activation = activation
    self._dense_embeddings = dense_embeddings
    self._absolute_positional_embed = absolute_positional_embed

    self._down_sample_all = down_sample_all

    inputs = tf.keras.layers.Input(shape=self._input_shape.shape[1:])
    output = self._build_struct(inputs)
    super().__init__(inputs=inputs, outputs=output, name="swin_transformer")

  def _build_struct(self, inputs):
    outputs = dict()

    if self._dense_embeddings:
      x = attention_blocks.PatchExtractEmbed(
              patch_size=self._patch_size, 
              embed_dimentions=self._embed_dims, 
              kernel_initializer = self._kernel_initializer,
              kernel_regularizer = self._kernel_regularizer, 
              bias_initializer = self._bias_initializer, 
              bias_regularizer = self._bias_regularizer, 
              drop=self._drop, 
              activation = None)(inputs)
    else:
      x = attention_blocks.PatchEmbed(
              patch_size=self._patch_size, 
              embed_dimentions=self._embed_dims, 
              use_layer_norm=self._patch_norm, 
              kernel_initializer = self._kernel_initializer,
              kernel_regularizer = self._kernel_regularizer, 
              bias_initializer = self._bias_initializer, 
              bias_regularizer = self._bias_regularizer, 
              drop=self._drop, 
              absolute_positional_embed=self._absolute_positional_embed)(inputs)

    base = key = int(math.log2(self._patch_size))
    outputs[str(key)] = x 
    
    stochastic_drops = np.linspace(0, self._drop_path, num = sum(self._depths))
    stochastic_drops = stochastic_drops.tolist()

    num_layers = len(self._depths)
    for i in range(num_layers):
      depth = self._depths[i]
      num_heads = self._num_heads[i]
      if isinstance(self._window_size, List):
        window_size = self._window_size[i]
      else:
        window_size = self._window_size

      if self._down_sample_all:
        downsample = True
      else: 
        downsample = i < num_layers - 1

      layer = attention_blocks.SwinTransformerBlock(
          depth=depth, 
          num_heads=num_heads, 
          window_size=window_size, 
          mlp_ratio=self._mlp_ratio, 
          use_bias_qkv=self._use_bias_qkv, 
          qk_scale=self._qk_scale, 
          activation = self._activation,
          drop=self._drop, 
          attention_drop=self._attention_drop, 
          drop_path=stochastic_drops[sum(self._depths[:i]):sum(self._depths[:i+1])], 
          downsample_type= 'patch_and_merge' if downsample else None, 
          kernel_initializer = self._kernel_initializer,
          kernel_regularizer = self._kernel_regularizer, 
          bias_initializer = self._bias_initializer, 
          bias_regularizer = self._bias_regularizer)

      if downsample:
        key += 1
      
      x = layer(x)
      outputs[str(key)] = x
    
    endpoints = dict() 
    output_spec = dict()

    if self._max_level is None:
      self._max_level = key
    if self._min_level is None:
      self._min_level = key
    for i in range(self._min_level, self._max_level + 1):
      x = outputs[str(i)]
      if self._normalize_endpoints:
        endpoints[str(i)] = self._norm_fn()(x)
      else:
        endpoints[str(i)] = x
      output_spec[str(i)] = endpoints[str(i)].get_shape()
    
    self._output_specs = output_spec
    return endpoints

  @property
  def input_specs(self):
    return self._input_shape

  @property
  def output_specs(self):
    return self._output_specs

  def get_config(self):
    layer_config = {
        'min_level': self._min_size,
        'max_level': self._max_size,
        'kernel_initializer': self._kernel_initializer,
        'kernel_regularizer': self._kernel_regularizer,
        'bias_regularizer': self._bias_regularizer,
        'norm_momentum': self._norm_momentum,
        'norm_epsilon': self._norm_epislon,
        'use_layer_norm': self._use_layer_norm,
        'use_bn': self._use_bn,
        'use_sync_bn': self._use_sync_bn,
        'activation': self._activation,
    }
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
      patch_size=backbone_config.patch_size,
      embed_dims=backbone_config.embed_dims,
      window_size=backbone_config.window_size,
      depths=backbone_config.depths,
      num_heads=backbone_config.num_heads, 
      activation=norm_activation_config.activation,
      kernel_regularizer=l2_regularizer,
      drop = backbone_config.drop, 
      attention_drop = backbone_config.attention_drop, 
      drop_path = backbone_config.drop_path, 
      down_sample_all = backbone_config.downsample_all)
  model.summary()
  return model

if __name__ == "__main__":
  shape = [640, 640, 3]
  input_specs = tf.keras.layers.InputSpec(shape=[None]+shape)
  model = SwinTransformer(
    input_specs = input_specs, 
    min_level = 3, 
    max_level = None, 
    patch_size = 4, 
    embed_dims = 96, 
    window_size = [8, 8, 8, 4], 
    depths = [2, 2, 6, 2], 
    num_heads = [3, 6, 12, 24], 
    
  )

  model.summary()
  input1 = tf.ones([1] + shape)

  output = model(input1)
  print({k:v.get_shape() for k,v in output.items()})
  # output = layer(input1)
  # print(output.shape)
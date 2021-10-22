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
"""Contains common building blocks for yolo neural networks."""
import logging
from official.vision.beta.ops import spatial_transform_ops
from typing import List, Tuple
from yolo.modeling.layers import nn_blocks
import tensorflow as tf
import numpy as np
from official.modeling import tf_utils
from official.vision.beta.modeling.layers import nn_layers
from functools import partial

USE_SYNC_BN = True
SHIFT = True

def _get_activation_fn(activation, leaky_alpha = 0.1):
  if activation == 'leaky':
    return tf.keras.layers.LeakyReLU(alpha=leaky_alpha)
  elif activation == 'mish':
    return lambda x: x * tf.math.tanh(tf.math.softplus(x))
  return tf_utils.get_activation(activation)

def _get_norm_fn(norm_layer, 
                 axis = -1, 
                 momentum = 0.97, 
                 epsilon = 0.0001, 
                 moving_mean_initializer='zeros', 
                 moving_variance_initializer='ones'):
  kwargs = dict(
    axis = axis,
    momentum = momentum, 
    epsilon = epsilon, 
    moving_mean_initializer = moving_mean_initializer, 
    moving_variance_initializer = moving_variance_initializer
  )
  if norm_layer is None:
    return None
  elif norm_layer == 'layer_norm' or norm_layer == 'bn_ln_norm':
    fn = tf.keras.layers.LayerNormalization
    return partial(fn, axis = axis, epsilon = epsilon)
  elif norm_layer == 'sync_batch_norm':
    fn = tf.keras.layers.experimental.SyncBatchNormalization
    return partial(fn, **kwargs)
  fn = tf.keras.layers.BatchNormalization
  return partial(fn, **kwargs)

def _get_initializer(initializer):
  if initializer == 'TruncatedNormal':
    initializer = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.02)
  return initializer

def window_partition(x, window_size):
  """
  Args:
    x: (B, H, W, C)
    window_size (int): window size

  Returns:
    windows: (num_windows*B, window_size, window_size, C)
  """
  _, H, W, C = x.shape
  if isinstance(window_size, int):
    window_size = (window_size, window_size)
  x = tf.reshape(x, [-1, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C])
  x = tf.transpose(x, perm=(0, 1, 3, 2, 4, 5))
  x = tf.reshape(x, [-1, window_size[0], window_size[1], C])
  return x

def window_reverse(x, window_size, H, W):
  """
  Args:
    windows: (num_windows*B, window_size, window_size, C)
    window_size (int): Window size
    H (int): Height of image
    W (int): Width of image

  Returns:
    x: (B, H, W, C)
  """
  _, _, _, C = x.shape
  if isinstance(window_size, int):
    window_size = (window_size, window_size)
  x = tf.reshape(x, [-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C])
  x = tf.transpose(x, perm=(0, 1, 3, 2, 4, 5))
  x = tf.reshape(x, [-1, H, W, C])
  return x

class Identity(tf.keras.layers.Layer):

  def call(self, inputs):
    return inputs

class SpatialAttention(tf.keras.layers.Layer):

  """Windowd multi head attention with a location based relative bias that is learned. 
  
  Functionally equivalent to multi head self attention but adds a gather for the 
  positional bias. 
  """

  def __init__(self, 
               num_heads, 
               qk_scale = None, 
               attention_dropout = 0.0, 
               attention_activation = 'softmax', # typically jsut soft max, more for future developments of something better 
               relative_bias_regularizer=None, 
               relative_bias_initializer='TruncatedNormal', 
               **kwargs):
    super().__init__(**kwargs)

    # arration params
    self._num_heads = num_heads
    self._qk_scale = qk_scale 

    # dropout
    self._attention_dropout = attention_dropout
    self._attention_activation = attention_activation

    # init and regularizer
    self._relative_bias_initializer = _get_initializer(relative_bias_initializer)
    self._relative_bias_regularizer = relative_bias_regularizer
    return
  
  def build(self, input_shape):

    q_shape = input_shape[0]
    kv_shape = input_shape[1]

    self.q_dim = q_shape[-1]
    head_dims = self.q_dim//self._num_heads
    self.scale = self._qk_scale or head_dims ** -0.5 # x/sqrt(num_heads) = x/sqrt(d_k)

    _, qWh, qWw, C = q_shape
    qN = qWh * qWw 
    _, kWh, kWw, C = kv_shape
    kN = kWh * kWw 
    self._window_size_q = [qWh, qWw] 
    self._window_size_k = [kWh, kWw] 

    # # biases to apply to each "position" learned 
    num_elements = (qWh + kWh - 1) * (qWw + kWw - 1)
    self._realtive_positional_bias_table = self.add_weight(
      name = '{}_realtive_positional_bias_table'.format(self.name), 
      shape = [num_elements, self._num_heads], 
      initializer = self._relative_bias_initializer, 
      regularizer = self._relative_bias_regularizer, 
      trainable = True)

    # get the postions to associate the above bais table with    
    coords_h = np.arange(qWh)
    coords_w = np.arange(qWw)
    coords = np.stack(np.meshgrid(coords_h, coords_w, indexing='ij')) # 2, qWh, qWw
    coords_flatten_q = coords.reshape(2, -1) # 2, qWh*qWw

    coords_h = np.arange(kWh)
    coords_w = np.arange(kWw)
    coords = np.stack(np.meshgrid(coords_h, coords_w, indexing='ij')) # 2, kWh, kWw
    coords_flatten_k = coords.reshape(2, -1) # 2, kWh*kWw

    relative_coords = coords_flatten_q[:, :, None] - coords_flatten_k[:, None, :] # 2, qWh*qWw, kWh*kWw
    relative_coords = relative_coords.transpose([1, 2, 0]) # qWh*qWw, kWh*kWw, 2
    relative_coords[:, :, 0] += kWh - 1 # shift to start from 0
    relative_coords[:, :, 1] += kWw - 1
    relative_coords[:, :, 0] *= (qWw + kWw - 1)
    relative_positional_indexes = relative_coords.sum(-1) # qWh*qWw, kWh*kWw
    self._relative_positional_indexes = tf.Variable(
      initial_value=tf.convert_to_tensor(relative_positional_indexes), 
      trainable=False, 
      name='{}_realtive_indexes'.format(self.name))

    self.attn_drop = tf.keras.layers.Dropout(self._attention_dropout)
    self.act = _get_activation_fn(self._attention_activation) # softmax
    return 
  
  def get_indexed_bias(self):
    # compute the relative poisiton bias
    num_elems_q = self._window_size_q[0] * self._window_size_q[1]
    num_elems_k = self._window_size_k[0] * self._window_size_k[1]
    indexes = tf.reshape(self._relative_positional_indexes, [-1])
    relative_position_bias = tf.gather(self._realtive_positional_bias_table, indexes)
    relative_position_bias = tf.reshape(relative_position_bias, [num_elems_q, num_elems_k, -1]) # Wh*Ww,Wh*Ww,nH
    relative_position_bias = tf.transpose(relative_position_bias, perm=(2, 0, 1)) # nH, Wh*Ww, Wh*Ww
    return tf.expand_dims(relative_position_bias, axis = 0)

  def reshape_projection(self, x, N, C):
    x = tf.reshape(x, [-1, N, self._num_heads, C // self._num_heads])
    x = tf.transpose(x, perm=(0, 2, 1, 3))
    return x

  def call(self, qkv, mask = None, training = None):
    q, k, v = qkv 

    _, qWh, qWw, C = q.shape
    qN = qWh * qWw 
    _, kWh, kWw, C = k.shape
    kN = kWh * kWw 

    q = self.reshape_projection(q, qN, C)
    k = self.reshape_projection(k, kN, C)
    v = self.reshape_projection(v, kN, C)

    # compute the matrix mul attention
    q = q * self.scale
    attn = tf.matmul(q, k, transpose_b = True)

    # compute the relative poisiton bias
    relative_position_bias = self.get_indexed_bias()
    attn = attn + relative_position_bias

    if mask is not None:
      num_windows = mask.shape[0]
      mask = tf.cast(tf.expand_dims(tf.expand_dims(mask, axis = 1), axis = 0), attn.dtype)
      attn = tf.reshape(attn, [-1, num_windows, self._num_heads, qN, kN]) + mask
      attn = tf.reshape(attn, [-1, self._num_heads, qN, kN])

    attn = self.act(attn)
    if training:
      attn = self.attn_drop(attn)

    x = tf.einsum("bhij,bhjk->bihk", attn, v)
    x = tf.reshape(x, [-1, qWh, qWw, C])
    return x, attn

class WindowedAttention(tf.keras.layers.Layer):
  def __init__(self, 
               num_heads, 
               kernel_size = 1,
               post_kernel_size = None,
               groups = 1, 
               strides = 1, 
               use_separable_conv = True, 
               dilation_rate = 1, 
               qkv_bias = True, 
               qk_scale = None, 
               attention_dropout = 0.0, 
               attention_activation = 'softmax', # typically jsut soft max, more for future developments of something better 
               relative_bias_initializer='TruncatedNormal', 
               kernel_initializer='TruncatedNormal',
               kernel_regularizer=None,
               bias_initializer='zeros',
               bias_regularizer=None, 
               project_attention = True, 
               projection_dropout = 0.0, 
               **kwargs):
    super().__init__(**kwargs)
    self._pre_kernel_size = kernel_size
    self._post_kernel_size = post_kernel_size or kernel_size
    self._groups = groups
    self._strides = strides
    self._use_separable_conv = use_separable_conv
    self._dilation_rate = dilation_rate

    self._num_heads = num_heads
    self._qkv_bias = qkv_bias
    self._qk_scale = qk_scale 

    self._project_attention = project_attention

    # dropout
    self._attention_dropout = attention_dropout
    self._projection_dropout = projection_dropout

    # activation 
    self._attention_activation = attention_activation

    # init and regularizer
    self._relative_bias_initializer = _get_initializer(relative_bias_initializer)
    self._relative_bias_regularizer = bias_regularizer
    self._init_args = dict(
      kernel_initializer = _get_initializer(kernel_initializer),
      bias_initializer = bias_initializer,
      kernel_regularizer = kernel_regularizer,
      bias_regularizer = bias_regularizer,
    )
    return 

  def build(self, input_shape):
    self.dims = input_shape[-1]

    self.attention = SpatialAttention(
      self._num_heads, 
      qk_scale=self._qk_scale, 
      attention_dropout=self._attention_dropout, 
      attention_activation=self._attention_activation,
      relative_bias_regularizer=self._relative_bias_regularizer,
      relative_bias_initializer=self._relative_bias_initializer)

    self.q = nn_blocks.ConvBN(
      filters = self.dims, 
      groups = self._groups, 
      kernel_size = self._pre_kernel_size, 
      strides = self._strides, 
      padding = "same", 
      use_bias = self._qkv_bias,
      use_bn = False, 
      activation = None, 
      use_separable_conv = self._use_separable_conv, 
      dilation_rate = self._dilation_rate, 
      **self._init_args
    )

    self.kv = nn_blocks.ConvBN(
      filters = self.dims * 2, 
      groups = self._groups, 
      kernel_size = self._pre_kernel_size, 
      strides = self._strides, 
      padding = "same", 
      use_bias = self._qkv_bias,
      use_bn = False, 
      activation = None, 
      use_separable_conv = self._use_separable_conv, 
      dilation_rate = self._dilation_rate, 
      **self._init_args
    )

    if self._project_attention:
      self.projection = nn_blocks.ConvBN(
        filters = self.dims, 
        groups = self._groups, 
        kernel_size = self._post_kernel_size, 
        strides = self._strides, 
        padding = "same", 
        use_bias = self._qkv_bias,
        use_bn = False, 
        activation = None, 
        use_separable_conv = self._use_separable_conv, 
        dilation_rate = self._dilation_rate, 
        **self._init_args
      )
    
  def call(self, query, source, mask = None, training = None):
    querys = self.q(query)
    keys_and_values = self.kv(source)

    _, H, W, C = keys_and_values.shape
    keys_and_values = tf.reshape(keys_and_values, [-1, H, W, 2, C//2])
    keys = keys_and_values[..., 0, :]
    values = keys_and_values[..., 1, :]

    x, attn = self.attention([querys, keys, values], mask = mask, training = training)

    if self._project_attention:
      x = self.projection(x)
    return x, attn


if __name__ == "__main__":
  inputs = {"3": [2, 7, 7, 256], "4": [2, 14, 14, 256], "5": [2, 28, 28, 256]}

  built = {}
  for k, s in inputs.items():
    built[k] = tf.ones(inputs[k])

  layer = WindowedAttention(16, kernel_size=(3, 3))
  output, _ = layer(built["5"], built["4"])
  print(output.shape)
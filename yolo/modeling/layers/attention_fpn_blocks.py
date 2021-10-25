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
from yolo.modeling.layers.attention_utils import *
import tensorflow as tf
import numpy as np
from official.modeling import tf_utils
from official.vision.beta.modeling.layers import nn_layers
from functools import partial

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
               use_bn=False,
               use_sync_bn=False,
               norm_momentum=0.99,
               norm_epsilon=0.001,
               dilation_rate = 1, 
               qkv_bias = True, 
               qk_scale = None, 
               attention_dropout = 0.0, 
               attention_activation = 'softmax', # typically jsut soft max, more for future developments of something better 
               project_attention = True, 
               projection_dropout = 0.0, 
               projection_expansion = 1.0,
               projection_use_bias = False, 
               projection_activation = None, 
               relative_bias_initializer='TruncatedNormal', 
               kernel_initializer='TruncatedNormal',
               kernel_regularizer=None,
               bias_initializer='zeros',
               bias_regularizer=None, 
               **kwargs):
    super().__init__(**kwargs)
    self._pre_kernel_size = kernel_size
    self._post_kernel_size = post_kernel_size or kernel_size
    self._groups = groups
    self._strides = strides
    self._use_separable_conv = use_separable_conv
    self._dilation_rate = dilation_rate

    self._use_bn = use_bn
    self._use_sync_bn = use_sync_bn
    self._norm_momentum = norm_momentum
    self._norm_epsilon = norm_epsilon

    self._num_heads = num_heads
    self._qkv_bias = qkv_bias
    self._qk_scale = qk_scale 

    self._project_attention = project_attention
    self._projection_dropout = projection_dropout
    self._projection_expansion = projection_expansion
    self._projection_use_bias = projection_use_bias
    self._projection_activation = projection_activation

    # dropout
    self._attention_dropout = attention_dropout
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
        filters = int(self.dims * self._projection_expansion),
        groups = self._groups, 
        kernel_size = self._post_kernel_size, 
        strides = self._strides, 
        padding = "same", 
        use_bias = self._projection_use_bias,
        use_bn = self._use_bn,
        use_sync_bn = self._use_sync_bn,
        norm_momentum = self._norm_momentum,
        norm_epsilon = self._norm_epsilon,
        activation = self._projection_activation, 
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

class ShiftedWindowAttention(tf.keras.layers.Layer):

  def __init__(self, 
               query_window_size, 
               source_window_size,
               num_heads, 
               shift = False, 
               kernel_size = 1,
               post_kernel_size = None,
               groups = 1, 
               strides = 1, 
               use_separable_conv = True, 
               use_bn=True,
               use_sync_bn=False,
               norm_momentum=0.99,
               norm_epsilon=0.001,
               dilation_rate = 1, 
               qkv_bias = True, 
               qk_scale = None, 
               drop_path = 0.0,
               attention_dropout = 0.0, 
               attention_activation = 'softmax', # typically jsut soft max, more for future developments of something better 
               project_attention = True, 
               projection_dropout = 0.0, 
               projection_expansion = 1.0,
               projection_use_bias = False, 
               projection_activation = None, 
               relative_bias_initializer='TruncatedNormal', 
               kernel_initializer='TruncatedNormal',
               kernel_regularizer=None,
               bias_initializer='zeros',
               bias_regularizer=None, 
               **kwargs):

    super().__init__(**kwargs)
    self._kernel_size = kernel_size
    self._post_kernel_size = post_kernel_size or kernel_size
    self._groups = groups
    self._strides = strides
    self._use_separable_conv = use_separable_conv
    self._dilation_rate = dilation_rate

    if isinstance(query_window_size, int):
      query_window_size = (query_window_size, query_window_size)
    self._query_window_size = query_window_size

    if isinstance(source_window_size, int):
      source_window_size = (source_window_size, source_window_size)
    self._source_window_size = source_window_size

    self._query_shift_size = (query_window_size[0]//2, query_window_size[1]//2)
    self._source_shift_size = (source_window_size[0]//2, source_window_size[1]//2)

    self._num_heads = num_heads
    self._qkv_bias = qkv_bias
    self._qk_scale = qk_scale 
    self._shift = shift

    self._use_bn = use_bn
    self._use_sync_bn = use_sync_bn
    self._norm_momentum = norm_momentum
    self._norm_epsilon = norm_epsilon

    self._project_attention = project_attention
    self._projection_dropout = projection_dropout
    self._projection_expansion = projection_expansion
    self._projection_use_bias = projection_use_bias
    self._projection_activation = projection_activation

    # dropout
    self._attention_activation = attention_activation
    self._attention_dropout = attention_dropout
    self._drop_path = drop_path

    # init and regularizer
    self._init_args = dict(
      kernel_initializer = _get_initializer(kernel_initializer),
      bias_initializer = bias_initializer,
      kernel_regularizer = kernel_regularizer,
      bias_regularizer = bias_regularizer,
      relative_bias_initializer = _get_initializer(relative_bias_initializer)
    )
    return 


  def build(self, input_shape):
    self.attention = WindowedAttention(
      self._num_heads,
      kernel_size=self._kernel_size,
      post_kernel_size=self._post_kernel_size,
      groups=self._groups, 
      strides=self._strides, 
      use_separable_conv=self._use_separable_conv, 
      dilation_rate=self._dilation_rate, 
      qkv_bias=self._qkv_bias, 
      qk_scale=self._qk_scale, 
      use_bn = self._use_bn,
      use_sync_bn = self._use_sync_bn,
      norm_momentum = self._norm_momentum,
      norm_epsilon = self._norm_epsilon,
      attention_dropout=self._attention_dropout, 
      attention_activation=self._attention_activation, 
      project_attention = self._project_attention,
      projection_dropout = self._projection_dropout,
      projection_expansion = self._projection_expansion,
      projection_use_bias = self._projection_use_bias,
      projection_activation = self._projection_activation,
      **self._init_args 
    )

    if self._drop_path > 0.0:
      self.drop_path = nn_layers.StochasticDepth(self._drop_path)
    else:
      self.drop_path = Identity()
    return

  def call(self, query, source, mask = None, training = None):

    (q, q_windows, q_shifts, 
    q_split_size, qH, qW, 
    qC, qHp, qWp) = pad_and_shift_input(query, 
                                        self._query_window_size, 
                                        self._query_shift_size, 
                                        shift = self._shift)

    (s, s_windows, s_shifts, 
    s_split_size, sH, sW, 
    sC, sHp, sWp) = pad_and_shift_input(source, 
                                        self._source_window_size, 
                                        self._source_shift_size, 
                                        shift = self._shift)


    attn_windows, attn = self.attention(q_windows, 
                                        s_windows, 
                                        mask = mask, 
                                        training = training) # output is in the queries frame of ref

    x_output = upad_and_unshift(attn_windows, q_split_size, qHp,qWp, 
        self._query_window_size, q_shifts, qH, qW)

    return x_output

class ShiftedWindowSelfAttention(tf.keras.layers.Layer):

  def __init__(self, 
               window_size, 
               num_heads, 
               shift = False, 
               kernel_size = 1,
               post_kernel_size = None,
               groups = 1, 
               strides = 1, 
               use_separable_conv = True, 
               use_bn=True,
               use_sync_bn=False,
               norm_momentum=0.99,
               norm_epsilon=0.001,
               dilation_rate = 1, 
               qkv_bias = True, 
               qk_scale = None, 
               drop_path = 0.0,
               attention_dropout = 0.0, 
               attention_activation = 'softmax', # typically jsut soft max, more for future developments of something better 
               project_attention = True, 
               projection_dropout = 0.0, 
               projection_expansion = 1.0,
               projection_use_bias = False, 
               projection_activation = None, 
               relative_bias_initializer='TruncatedNormal', 
               kernel_initializer='TruncatedNormal',
               kernel_regularizer=None,
               bias_initializer='zeros',
               bias_regularizer=None, 
               **kwargs):

    super().__init__(**kwargs)
    self._kernel_size = kernel_size
    self._post_kernel_size = post_kernel_size or kernel_size
    self._groups = groups
    self._strides = strides
    self._use_separable_conv = use_separable_conv
    self._dilation_rate = dilation_rate

    if isinstance(window_size, int):
      window_size = (window_size, window_size)
    self._window_size = window_size
    self._shift_size = (window_size[0]//2, window_size[1]//2)

    self._num_heads = num_heads
    self._qkv_bias = qkv_bias
    self._qk_scale = qk_scale 
    self._shift = shift

    self._use_bn = use_bn
    self._use_sync_bn = use_sync_bn
    self._norm_momentum = norm_momentum
    self._norm_epsilon = norm_epsilon

    self._project_attention = project_attention
    self._projection_dropout = projection_dropout
    self._projection_expansion = projection_expansion
    self._projection_use_bias = projection_use_bias
    self._projection_activation = projection_activation

    # dropout
    self._attention_activation = attention_activation
    self._attention_dropout = attention_dropout
    self._drop_path = drop_path

    # init and regularizer
    self._init_args = dict(
      kernel_initializer = _get_initializer(kernel_initializer),
      bias_initializer = bias_initializer,
      kernel_regularizer = kernel_regularizer,
      bias_regularizer = bias_regularizer,
      relative_bias_initializer = _get_initializer(relative_bias_initializer)
    )
    return 


  def build(self, input_shape):
    self.attention = WindowedAttention(
      self._num_heads,
      kernel_size=self._kernel_size,
      post_kernel_size=self._post_kernel_size,
      groups=self._groups, 
      strides=self._strides, 
      use_separable_conv=self._use_separable_conv, 
      dilation_rate=self._dilation_rate, 
      qkv_bias=self._qkv_bias, 
      qk_scale=self._qk_scale, 
      use_bn = self._use_bn,
      use_sync_bn = self._use_sync_bn,
      norm_momentum = self._norm_momentum,
      norm_epsilon = self._norm_epsilon,
      attention_dropout=self._attention_dropout, 
      attention_activation=self._attention_activation, 
      project_attention = self._project_attention,
      projection_dropout = self._projection_dropout,
      projection_expansion = self._projection_expansion,
      projection_use_bias = self._projection_use_bias,
      projection_activation = self._projection_activation,
      **self._init_args 
    )

    if self._drop_path > 0.0:
      self.drop_path = nn_layers.StochasticDepth(self._drop_path)
    else:
      self.drop_path = Identity()
    return

  def call(self, x, mask = None, training = None):
    (x, x_windows, x_shifts, 
    x_split_size, H, W, 
    C, Hp, Wp) = pad_and_shift_input(
        x, self._window_size, self._shift_size, shift = self._shift)

    attn_windows, attn = self.attention(x_windows, x_windows, mask = mask, 
        training = training) # output is in the queries frame of ref

    x_output = upad_and_unshift(attn_windows, x_split_size, Hp,Wp, 
        self._window_size, x_shifts, H, W)
    return x_output

class ExpandedWindowSelfAttention(tf.keras.layers.Layer):

  def __init__(self, 
               window_size, 
               num_heads, 
               tlbr = False,
               shift = False, 
               kernel_size = 1,
               post_kernel_size = None,
               groups = 1, 
               strides = 1, 
               use_separable_conv = True, 
               use_bn=True,
               use_sync_bn=False,
               norm_momentum=0.99,
               norm_epsilon=0.001,
               dilation_rate = 1, 
               qkv_bias = True, 
               qk_scale = None, 
               drop_path = 0.0,
               attention_dropout = 0.0, 
               attention_activation = 'softmax', # typically jsut soft max, more for future developments of something better 
               project_attention = True, 
               projection_dropout = 0.0, 
               projection_expansion = 1.0,
               projection_use_bias = False, 
               projection_activation = None, 
               relative_bias_initializer='TruncatedNormal', 
               kernel_initializer='TruncatedNormal',
               kernel_regularizer=None,
               bias_initializer='zeros',
               bias_regularizer=None, 
               **kwargs):

    super().__init__(**kwargs)
    self._kernel_size = kernel_size
    self._post_kernel_size = post_kernel_size or kernel_size
    self._groups = groups
    self._strides = strides
    self._use_separable_conv = use_separable_conv
    self._dilation_rate = dilation_rate

    if isinstance(window_size, int):
      window_size = (window_size, window_size)
    self._window_size = window_size

    self._tlbr = tlbr

    self._num_heads = num_heads
    self._qkv_bias = qkv_bias
    self._qk_scale = qk_scale 
    self._shift = shift

    self._use_bn = use_bn
    self._use_sync_bn = use_sync_bn
    self._norm_momentum = norm_momentum
    self._norm_epsilon = norm_epsilon

    self._project_attention = project_attention
    self._projection_dropout = projection_dropout
    self._projection_expansion = projection_expansion
    self._projection_use_bias = projection_use_bias
    self._projection_activation = projection_activation

    # dropout
    self._attention_activation = attention_activation
    self._attention_dropout = attention_dropout
    self._drop_path = drop_path

    # init and regularizer
    self._init_args = dict(
      kernel_initializer = _get_initializer(kernel_initializer),
      bias_initializer = bias_initializer,
      kernel_regularizer = kernel_regularizer,
      bias_regularizer = bias_regularizer,
      relative_bias_initializer = _get_initializer(relative_bias_initializer)
    )
    return 


  def build(self, input_shape):
    self.attention = WindowedAttention(
      self._num_heads,
      kernel_size=self._kernel_size,
      post_kernel_size=self._post_kernel_size,
      groups=self._groups, 
      strides=self._strides, 
      use_separable_conv=self._use_separable_conv, 
      dilation_rate=self._dilation_rate, 
      qkv_bias=self._qkv_bias, 
      qk_scale=self._qk_scale, 
      use_bn = self._use_bn,
      use_sync_bn = self._use_sync_bn,
      norm_momentum = self._norm_momentum,
      norm_epsilon = self._norm_epsilon,
      attention_dropout=self._attention_dropout, 
      attention_activation=self._attention_activation, 
      project_attention = self._project_attention,
      projection_dropout = self._projection_dropout,
      projection_expansion = self._projection_expansion,
      projection_use_bias = self._projection_use_bias,
      projection_activation = self._projection_activation,
      **self._init_args 
    )

    if self._drop_path > 0.0:
      self.drop_path = nn_layers.StochasticDepth(self._drop_path)
    else:
      self.drop_path = Identity()
    return

  def call(self, x, mask = None, training = None):
    x, H, W, C, Hp, Wp = pad(x, self._window_size)

    if self._tlbr:
      source_windows, _, _ = window_partition_overlaps_tlbr(x, self._window_size)
    else:
      source_windows, _, _ = window_partition_overlaps_br(x, self._window_size)
    
    # source_windows, _, _ = window_partition(x, self._window_size) 
    query_windows, _, _ = window_partition(x, self._window_size) 
    attn_windows, attn = self.attention(query_windows, 
        source_windows, mask = mask, training = training)
    
    x_output = window_reverse(attn_windows, self._window_size, Hp, Wp)
    if Hp != H or Wp != W:
      x_output = x_output[:, :H, :W, :]
    return x_output

class MergeShapeToMatch(tf.keras.layers.Layer):
  def __init__(self, 
               window_size, 
               num_heads, 
               mlp_ratio = 1.5, 
               shift = False, 
               kernel_size = 1,
               post_kernel_size = None,
               expansion_kernel_size = 3, 
               groups = 1, 
               strides = 1, 
               use_separable_conv = False, 
               use_bn=True,
               use_sync_bn=False,
               norm_momentum=0.99,
               norm_epsilon=0.001,
               dilation_rate = 1, 
               qkv_bias = True, 
               qk_scale = None, 
               drop_path = 0.0,
               attention_dropout = 0.0, 
               attention_activation = "softmax", # typically jsut soft max, more for future developments of something better 
               expansion_activation = "mish", 
               project_attention = True, 
               projection_dropout = 0.0, 
               projection_expansion = 1.0,
               projection_use_bias = False, 
               projection_use_bn = False, 
               projection_activation = None, 
               relative_bias_initializer='TruncatedNormal', 
               kernel_initializer='TruncatedNormal',
               kernel_regularizer=None,
               bias_initializer='zeros',
               bias_regularizer=None, 
               **kwargs):

    super().__init__(**kwargs)
    self._kernel_size = kernel_size
    self._post_kernel_size = post_kernel_size or kernel_size
    self._groups = groups
    self._strides = strides
    self._use_separable_conv = use_separable_conv
    self._dilation_rate = dilation_rate

    if isinstance(window_size, int):
      window_size = (window_size, window_size)
    self._window_size = window_size

    self._num_heads = num_heads
    self._qkv_bias = qkv_bias
    self._qk_scale = qk_scale 
    self._shift = shift
    self._mlp_ratio = mlp_ratio

    self._use_bn = use_bn
    self._use_sync_bn = use_sync_bn
    self._norm_momentum = norm_momentum
    self._norm_epsilon = norm_epsilon

    self._project_attention = project_attention
    self._projection_dropout = projection_dropout
    self._projection_expansion = projection_expansion
    self._projection_use_bias = projection_use_bias
    self._projection_activation = projection_activation
    self._projection_use_bn = projection_use_bn

    self._expansion_activation = expansion_activation
    self._expansion_kernel_size = expansion_kernel_size
    # dropout
    self._attention_activation = attention_activation
    self._attention_dropout = attention_dropout
    self._drop_path = drop_path

    # init and regularizer
    self._relative_bias_initializer = _get_initializer(relative_bias_initializer)
    self._init_args = dict(
      kernel_initializer = _get_initializer(kernel_initializer),
      bias_initializer = bias_initializer,
      kernel_regularizer = kernel_regularizer,
      bias_regularizer = bias_regularizer,
    )
    return 

  def build(self, input_shape):
    source_shape, query_shape = input_shape # [upsample, match]
    self.dims = int(query_shape[-1] * self._projection_expansion)
    
    self._sample_size = (query_shape[1]/source_shape[1], 
                         query_shape[2]/source_shape[2])
    self._query_window_size = (int(self._window_size[0] * self._sample_size[0]),
                               int(self._window_size[1] * self._sample_size[0]))

    if max(self._sample_size) == min(self._sample_size) and self._sample_size[0] > 1:
      self._upsample = True 
      self._sample_size = (int(self._sample_size[0]), int(self._sample_size[1]))
      self._query_window_size = (int(self._window_size[0] * self._sample_size[0]),
                                int(self._window_size[1] * self._sample_size[0]))
    elif max(self._sample_size) == min(self._sample_size) and self._sample_size[0] < 1:
      self._upsample = False
      self._sample_size = (int(source_shape[1]/query_shape[1]), 
                           int(source_shape[2]/query_shape[2]))
      self._query_window_size = (int(self._window_size[0] * self._sample_size[0]),
                                int(self._window_size[1] * self._sample_size[0]))
      self._query_window_size, self._window_size = self._window_size, self._query_window_size
    else:
      self._upsample = None

    self.mapping_attention = ShiftedWindowAttention(
        self._query_window_size,
        self._window_size, 
        shift = self._shift,
        num_heads = self._num_heads,
        kernel_size=self._kernel_size,
        post_kernel_size=self._post_kernel_size,
        groups=self._groups, 
        strides=self._strides, 
        use_separable_conv=self._use_separable_conv, 
        dilation_rate=self._dilation_rate, 
        qkv_bias=self._qkv_bias, 
        qk_scale=self._qk_scale, 
        use_bn = self._projection_use_bn,
        use_sync_bn = self._use_sync_bn,
        norm_momentum = self._norm_momentum,
        norm_epsilon = self._norm_epsilon,
        attention_dropout=self._attention_dropout, 
        attention_activation=self._attention_activation, 
        project_attention = self._project_attention,
        projection_dropout = self._projection_dropout,
        projection_expansion = self._projection_expansion,
        projection_use_bias = self._projection_use_bias,
        projection_activation = self._projection_activation,
        relative_bias_initializer = self._relative_bias_initializer,
        **self._init_args)


    if self._upsample == False:
      # self.pool = tf.keras.layers.MaxPool2D(
      #         pool_size=self._sample_size,
      #         strides=self._sample_size,
      #         padding='same',
      #         data_format=None)
      self.spatial_info = tf.keras.layers.DepthwiseConv2D(
                                3, strides = self._sample_size, 
                                padding = "same", 
                                use_bias = False, 
                                **self._init_args)
      self.bn = tf.keras.layers.BatchNormalization(
        momentum = self._norm_momentum,
        epsilon = self._norm_epsilon,
      )
      self.act = _get_activation_fn(self._expansion_activation)

    if self.dims != source_shape[-1]:
      self._channel_match = True
      self.channel_match = nn_blocks.ConvBN(
        filters = self.dims,
        kernel_size = 1, 
        strides = 1, 
        padding = "same", 
        use_bn = self._use_bn,
        use_sync_bn = self._use_sync_bn,
        norm_momentum = self._norm_momentum,
        norm_epsilon = self._norm_epsilon,
        activation = self._expansion_activation, 
        use_separable_conv = self._use_separable_conv, 
        dilation_rate = self._dilation_rate,
        **self._init_args
      )
    else:
      self._channel_match = False

    self._s_layer_norm = _get_norm_fn(
        "layer_norm", epsilon=self._norm_epsilon)(axis = -1)
    self._q_layer_norm = _get_norm_fn(
        "layer_norm", epsilon=self._norm_epsilon)(axis = -1)

    self._ffn = FFN(
      hidden_features=int(self.dims * self._mlp_ratio), 
      kernel_size = self._expansion_kernel_size, 
      strides = 1, 
      activation = self._expansion_activation, 
      groups = self._groups, 
      use_separable_conv = self._use_separable_conv, 
      use_bn = self._use_bn,
      use_sync_bn = self._use_sync_bn,
      norm_momentum = self._norm_momentum,
      norm_epsilon = self._norm_epsilon,
      dilation_rate = self._dilation_rate,
      **self._init_args
    ) 
    self.act = _get_activation_fn(self._expansion_activation)
    return
  
  def s_layer_norm(self, x):
    _, H, W, C = x.shape
    x = tf.reshape(x, [-1, H * W, C])
    x = self._s_layer_norm(x) 
    x = tf.reshape(x, [-1, H , W, C])
    return x

  def q_layer_norm(self, x):
    _, H, W, C = x.shape
    x = tf.reshape(x, [-1, H * W, C])
    x = self._q_layer_norm(x) 
    x = tf.reshape(x, [-1, H , W, C])
    return x

  def pool(self, x):
    x = self.spatial_info(x)
    x = self.bn(x)
    x = self.act(x)
    return x

  def call(self, inputs, mask = None, training = None):
    # maps px onto p(x+1) then adds it to the resampled source
    source, query = inputs 

    ns = self.s_layer_norm(source)
    nq = self.q_layer_norm(query)
    x = self.mapping_attention(nq, ns, mask = mask, training = training)
    if self._upsample == True:
      source = spatial_transform_ops.nearest_upsampling(source, self._sample_size[0])
    elif self._upsample == False:
      source = self.pool(source)
    
    if self._channel_match:
      source = self.channel_match(source)

    return self._ffn(x + source)

# once patched channel tokens are established and preserved so K > 1 is ok

# once unpatched channel tokens are no longer preserved so depth wise 
# combining operations need to be done with K = 1 and spatial operations need 
# to be restricted to each cahnnel map seperately
 
# distort spatial tokens byl merging across spatial tokens

class FFN(tf.keras.layers.Layer):

  def __init__(self, 
               hidden_features = None, 
               out_features = None, 
               kernel_size = 1,
               strides = 1, 
               invert = None,
               activation = "mish", 
               groups = 1, 
               use_bn=True,
               use_sync_bn=False,
               norm_momentum=0.99,
               norm_epsilon=0.001,
               dilation_rate = 1, 
               leaky_alpha=0.1,
               cat_input = True,
               use_separable_conv = False, 
               kernel_initializer='TruncatedNormal',
               kernel_regularizer=None,
               bias_initializer='zeros',
               bias_regularizer=None, 
               **kwargs):
    super().__init__(**kwargs)

    # features 
    self._invert = invert or kernel_size < 0
    self._hidden_features = hidden_features
    self._out_features = out_features
    self._cat_input = cat_input
    self._kernel_size = abs(kernel_size)
    self._strides = strides
    self._norm_momentum = norm_momentum
    self._norm_epsilon = norm_epsilon
    self._activation = activation

    # init and regularizer
    self._init_args = dict(
      activation = activation,
      groups = groups,
      use_separable_conv = use_separable_conv,
      use_bn = use_bn,
      use_sync_bn = use_sync_bn,
      norm_momentum = norm_momentum,
      norm_epsilon = norm_epsilon,
      dilation_rate = dilation_rate,
      kernel_initializer = _get_initializer(kernel_initializer),
      bias_initializer = bias_initializer,
      kernel_regularizer = kernel_regularizer,
      bias_regularizer = bias_regularizer,
      leaky_alpha = leaky_alpha)

    self._dw_init = dict(
      kernel_initializer = _get_initializer(kernel_initializer),
      bias_initializer = bias_initializer,
      kernel_regularizer = kernel_regularizer,
      bias_regularizer = bias_regularizer,
    )

  def build(self, input_shape):
    hidden_features = self._hidden_features or input_shape[-1]
    out_features = self._out_features or input_shape[-1]

    self.spatial_info = tf.keras.layers.DepthwiseConv2D(
                              3, strides = self._strides, 
                              padding = "same", 
                              use_bias = False, 
                              **self._dw_init)
    self.bn = tf.keras.layers.BatchNormalization(
      momentum = self._norm_momentum,
      epsilon = self._norm_epsilon,
    )
    self.act = _get_activation_fn(self._activation)

    self.fc_expand = nn_blocks.ConvBN(filters = hidden_features, 
                                      kernel_size = 1,
                                      strides = 1, 
                                      **self._init_args)
    self.fc_compress = nn_blocks.ConvBN(filters = out_features, 
                                        kernel_size = 1,
                                        strides = 1, 
                                        **self._init_args)

    # self.fc_expand = nn_blocks.ConvBN(filters = hidden_features, 
    #                                   kernel_size = ks1,
    #                                   strides = s1, 
    #                                   **self._init_args)
    # self.fc_compress = nn_blocks.ConvBN(filters = out_features, 
    #                                     kernel_size = ks2,
    #                                     strides = s2, 
    #                                     **self._init_args)
    return 

  def dw_reasoning(self, x):
    x = self.spatial_info(x)
    x = self.bn(x)
    x = self.act(x)
    return x

  def call(self, x):
    if self._invert:
      x = self.dw_reasoning(x)
      x = self.fc_expand(x)
    else:
      x = self.fc_expand(x)
      x = self.dw_reasoning(x)
    x = self.fc_compress(x)
    return x


class SPP(tf.keras.layers.Layer):
  """Spatial Pyramid Pooling.

  A non-agregated SPP layer that uses Pooling.
  """

  def __init__(self, sizes, weighted, strides = (1, 1), **kwargs):
    self._sizes = list(reversed(sizes))
    if not sizes:
      raise ValueError('More than one maxpool should be specified in SSP block')
    self._strides = strides
    self._weighted = weighted
    super().__init__(**kwargs)

  def weighted(self, size):
    return tf.keras.layers.DepthwiseConv2D(
                              (size, size), # spatial embedding combination and reduction
                              strides = self._strides, 
                              padding = "same", 
                              use_bias = False)

  def unweighted(self, size):
    return tf.keras.layers.MaxPool2D(
              pool_size=(size, size),
              strides=self._strides,
              padding='same',
              data_format=None)

  def build(self, input_shape):
    maxpools = []
    for size, weighted in zip(self._sizes, self._weighted):
      if weighted:
        maxpools.append(self.weighted(size))
      else:
        maxpools.append(self.unweighted(size))
    self._maxpools = maxpools
    super().build(input_shape)

  def call(self, inputs, training=None):
    outputs = []
    for maxpool in self._maxpools:
      outputs.append(maxpool(inputs))
    outputs.append(inputs)
    concat_output = tf.keras.layers.concatenate(outputs)
    return concat_output

  def get_config(self):
    layer_config = {'sizes': self._sizes}
    layer_config.update(super().get_config())
    return layer_config

class TBiFPN(tf.keras.Model):

  def __init__(self, 
               input_specs, 

               use_patch_expansion = True, 

               # catch
               include_catch = True,
               catch_transformer = True, 
               # spp_keys = [3, 5, 7], 
               spp_keys = [5, 9, 13], 
               weighted = [False, False, False],
               # spp_keys = [3, 5, 7], 
               # weighted = [True, True, True],
               catch_projection_expansion = 1.0, 
              
               repititions = 1, 
               fpn_only = False, 
               include_detokenization = True, 

               # conv
               kernel_size = 1,
               groups = 1, 
               strides = 1, 
               use_bn=True,
               
               # conv
               use_separable_conv = True, 
               use_sync_bn=False,
               norm_momentum=0.99,
               norm_epsilon=0.001,
               dilation_rate = 1, 
               conv_activation = "mish",
               
               # transformer
               transformers_kernel_size = 1, 
               transformers_projection_kernel_size = 1, 
               window_size = 16, 
               token_size = 32, 
               mlp_ratio = 1, 
               shift = False, 
               post_kernel_size = None,
               qkv_bias = True, 
               qk_scale = None, 
               drop_path = 0.0,
               attention_dropout = 0.0,                
               attention_activation = 'softmax', # typically jsut soft max, more for future developments of something better 
               projection_activation = None, 
               project_attention = True, 
               projection_dropout = 0.0, 
               projection_expansion = 1.0,
               projection_use_bias = False, 
               projection_use_bn = False, 
               expansion_kernel_size = 1, 
               
               # shared
               kernel_initializer='TruncatedNormal',
               kernel_regularizer=None,
               bias_initializer='zeros',
               bias_regularizer=None, 
               **kwargs) -> None:

    self._kernel_size = kernel_size
    self._strides = strides
    self._groups = groups

    self._window_size = window_size
    self._token_size = token_size
    self._norm_epsilon = norm_epsilon
    self._use_bn = use_bn
    self._activation = conv_activation

    self._include_catch = include_catch
    self._catch_transformer = catch_transformer
    self._spp_keys = spp_keys
    self._weighted = weighted
    self._catch_projection_expansion = catch_projection_expansion
    self._use_patch_expansion = use_patch_expansion

    self._repititions = repititions
    self._expansion_kernel_size = expansion_kernel_size

    self._init_args = dict(
      kernel_initializer = _get_initializer(kernel_initializer),
      bias_initializer = bias_initializer,
      kernel_regularizer = kernel_regularizer,
      bias_regularizer = bias_regularizer
    )

    self._conv_init_args = dict(
      groups = self._groups,
      use_separable_conv = use_separable_conv,
      use_sync_bn = use_sync_bn,
      norm_momentum = norm_momentum,
      norm_epsilon = norm_epsilon,
      dilation_rate = dilation_rate)

    self._transformers_kernel_size = transformers_kernel_size
    self._transformers_projection_kernel_size = transformers_projection_kernel_size
    self._transformer_init_args = dict(
      mlp_ratio = mlp_ratio, 
      shift = shift, 
      post_kernel_size = post_kernel_size,
      qkv_bias = qkv_bias, 
      qk_scale = qk_scale, 
      drop_path = drop_path,
      attention_dropout = attention_dropout, 
      attention_activation = attention_activation, 
      expansion_activation = self._activation, # ffn activation
      projection_activation = projection_activation,
      project_attention = project_attention, 
      projection_expansion = projection_expansion,
      projection_dropout = projection_dropout, 
      projection_use_bias = projection_use_bias, 
      use_bn = projection_use_bn)

    inputs = {
        key: tf.keras.layers.Input(shape=value[1:]) for key, value in input_specs.items()
    }
    if self._include_catch:
      outputs = self.build_catch(inputs)
    else:
      outputs = inputs

    for i in range(self._repititions):
      fpn_only_ = False
      if i == self._repititions - 1:
        fpn_only_ = fpn_only
      outputs = self.build_merging(outputs, fpn_only_)


    if include_detokenization:
      outputs = self.build_output(outputs)

    self._output_specs = {key: value.shape for key, value in outputs.items()}
    super().__init__(inputs=inputs, outputs=outputs, name='YoloDecoder')
    return

  def layer_norm(self, x):
    _, H, W, C = x.shape
    x = tf.reshape(x, [-1, H * W, C])
    x = _get_norm_fn("layer_norm", epsilon=self._norm_epsilon)(axis = -1)(x) 
    x = tf.reshape(x, [-1, H , W, C])
    return x
  
  def activation(self, x):
    return _get_activation_fn(self._activation)(x)

  def catch(self, x):
    init_args = self._init_args.copy()
    conv_args = self._conv_init_args.copy()
    transformer_args = self._transformer_init_args.copy()

    filters = x.shape[-1]
    point_wise_compress = nn_blocks.ConvBN(filters = filters//2, 
                                           kernel_size = 1, 
                                           strides = 1, 
                                           use_bn = self._use_bn,
                                           activation = self._activation,
                                           **conv_args, 
                                           **init_args)
    spatial_binning = SPP(self._spp_keys, self._weighted)
    point_wise_merge_bins = nn_blocks.ConvBN(filters = filters//2, 
                                           kernel_size = 1, 
                                           strides = 1, 
                                           use_bn = self._use_bn,
                                           activation = self._activation,
                                           **conv_args, 
                                           **init_args)
    point_wise_merge = nn_blocks.ConvBN(filters = filters, # maybe make a 3x3
                                          kernel_size = 1, 
                                          strides = 1, 
                                          use_bn = self._use_bn,
                                          activation = self._activation,
                                          **conv_args, 
                                          **init_args)


    x = point_wise_compress(x)
    x_bins = spatial_binning(x)
    x_bins = point_wise_merge_bins(x_bins)
    x = point_wise_merge(self.activation(x + x_bins))

    if self._catch_transformer:
      transformer_args["projection_expansion"] = self._catch_projection_expansion
      transformer_args["projection_activation"] = self._activation
      transformer_args["post_kernel_size"] = self._transformers_projection_kernel_size
      del transformer_args["mlp_ratio"], transformer_args["expansion_activation"]

      num_heads = x.shape[-1]//self._token_size

      if self._use_patch_expansion:
        attention = ExpandedWindowSelfAttention(
              window_size = self._window_size, 
              num_heads = num_heads, 
              tlbr=True,
              kernel_size=self._transformers_kernel_size, 
              **conv_args, 
              **transformer_args, 
              **init_args
          )
      else:
        attention = ShiftedWindowSelfAttention(
              window_size = self._window_size, 
              num_heads = num_heads, 
              kernel_size=self._transformers_kernel_size, 
              **conv_args, 
              **transformer_args, 
              **init_args
          )

      forward = nn_blocks.ConvBN(filters = filters, 
                                  kernel_size = 1, 
                                  strides = 1, 
                                  use_bn = self._use_bn,
                                  activation = self._activation,
                                  **conv_args, 
                                  **init_args)
      x += attention(self.layer_norm(x))
      x = forward(x)
    return x

  def build_catch(self, inputs): # 17,254,328 params
    outputs = {}
    for k, v in inputs.items():
      outputs[k] = self.catch(v)
    return outputs

  def merge_levels(self, source, query):
    init_args = self._init_args.copy()
    conv_args = self._conv_init_args.copy()
    transformer_args = self._transformer_init_args.copy()

    project_use_bn = transformer_args["use_bn"]
    del transformer_args["use_bn"]
    transformer_args["projection_use_bn"] = project_use_bn

    num_heads = query.shape[-1]//self._token_size
    merge_shapes = MergeShapeToMatch(
      window_size=self._window_size, 
      num_heads=num_heads, 
      expansion_kernel_size = self._expansion_kernel_size,
      **conv_args, 
      **transformer_args,
      **init_args
    )
    return merge_shapes([source, query])
  
  def build_merging(self, inputs, fpn_only):
    outputs = inputs
    min_level = int(min(outputs.keys()))
    max_level = int(max(outputs.keys()))

    #up_merge
    for i in reversed(range(min_level, max_level)):
      outputs[str(i)] = self.merge_levels(outputs[str(i + 1)], outputs[str(i)])

    # down_merge
    if not fpn_only:
      for i in range(min_level, max_level):
        outputs[str(i + 1)] = self.merge_levels(outputs[str(i)], outputs[str(i + 1)])

    return outputs

  def output_node(self, x):
    init_args = self._init_args.copy()
    conv_args = self._conv_init_args.copy()

    filters = x.shape[-1]

    # # detokenization
    # conv_args["groups"] = filters//self._token_size
    # spatial_reasoning = nn_blocks.ConvBN(filters = filters, # maybe make a 3x3
    #                                       kernel_size = 3, 
    #                                       strides = 1, 
    #                                       use_bn = self._use_bn,
    #                                       activation = self._activation,
    #                                       **conv_args, 
    #                                       **init_args)
    
    # conv_args["groups"] = self._groups
    point_wise_compress = nn_blocks.ConvBN(filters = filters//2, 
                                           kernel_size = 1, 
                                           strides = 1, 
                                           use_bn = self._use_bn,
                                           activation = self._activation,
                                           **conv_args, 
                                           **init_args)

    spatial_re_expantion = nn_blocks.ConvBN(filters = filters, 
                                            kernel_size = 3, 
                                            strides = 1, 
                                            use_bn = self._use_bn,
                                            activation = self._activation,
                                            **conv_args, 
                                            **init_args)
    
    # x = spatial_reasoning(x)
    x = point_wise_compress(x)
    x = spatial_re_expantion(x)
    return x

  def build_output(self, inputs): # 17,254,328 params
    outputs = {}
    for k, v in inputs.items():
      outputs[k] = self.output_node(v)
    return outputs

  @property
  def output_specs(self):
    return self._output_specs

  def get_config(self):
    config = dict(
        input_specs=self._input_specs,
        use_fpn=self._use_fpn,
        fpn_depth=self._fpn_depth,
        **self._decoder_config)
    return config

  @classmethod
  def from_config(cls, config, custom_objects=None):
    return cls(**config)
      

if __name__ == "__main__":
  inputs = {"3": [2, 80, 80, 256], "4": [2, 40, 40, 512], "5": [2, 20, 20, 1024]}

  m = TBiFPN(inputs, 
                  repititions=1, 
                  fpn_only=False, 
                  include_catch=True, 
                  include_detokenization=True)
  m.summary()

  built = {}
  for k, s in inputs.items():
    built[k] = tf.ones(inputs[k])

  m(built)
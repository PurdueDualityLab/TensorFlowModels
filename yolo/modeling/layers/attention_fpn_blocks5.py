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

from tensorflow.python.eager.backprop import _num_elements
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
               use_sync_bn=True,
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
               use_sync_bn=True,
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

  def call(self, inputs, mask = None, training = None):
    query, source = inputs

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
               use_sync_bn=True,
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

class TokenizerMlp(tf.keras.layers.Layer):
  def __init__(self, dims, activation = "mish", mlp_ratio = 1.5, **kwargs) -> None:
    super().__init__(**kwargs)
    self.dims = dims
    self.activation = activation
    self.mlp_ratio = mlp_ratio
  
  def build(self, input_shapes):
    self.expand = tf.keras.layers.Dense(
      self.dims * self.mlp_ratio,
      kernel_initializer='TruncatedNormal',
      kernel_regularizer=None,
      bias_initializer='zeros',
      bias_regularizer=None, 
    )
    self.act = _get_activation_fn(self.activation)
    self.project = tf.keras.layers.Dense(
      self.dims,
      kernel_initializer='TruncatedNormal',
      kernel_regularizer=None,
      bias_initializer='zeros',
      bias_regularizer=None, 
      use_bias = False,
    )
    return 
  
  def call(self, x):
    x = self.expand(x)
    x = self.act(x)
    x = self.project(x)
    return x

class Tokenizer(tf.keras.layers.Layer):

  def __init__(self, embedding_dims = None, **kwargs) -> None:
    super().__init__(**kwargs)
    self._embedding_dims = embedding_dims


  def build(self, input_shapes):
    conv_shape = input_shapes[0]
    num_elements = self._embedding_dims or conv_shape[-1] 

    # if len(input_shapes) == 3:
    #   self._merge_token_instructions = True
    #   self.impa_mlp = TokenizerMlp(num_elements)
    #   self.impm_mlp = TokenizerMlp(num_elements)
    # else:
    #   self._merge_token_instructions = False

    # learned keys indicating HOW the tokenization is to be done
    # each tokenizer gets its own but also needs to account for how the 
    # other levels were tokenized, each channel gets a tokenization key
    # self.impa = self.add_weight(
    #   name = '{}_tokenization_key_shift'.format(self.name),
    #   shape = [1, 1, 1, num_elements], 
    #   initializer = "TruncatedNormal", 
    # )
    # self.impm = self.add_weight(
    #   name = '{}_tokenization_key_scale'.format(self.name),
    #   shape = [1, 1, 1, num_elements], 
    #   initializer = "TruncatedNormal", 
    # )

    self.projection = nn_blocks.ConvBN( # after shift, project the channels to similar arangement 
        filters = num_elements,
        groups = 1, 
        kernel_size = 1, 
        strides = 1, 
        padding = "same", 
        use_bias = False,
        use_bn = True,
        use_sync_bn = True,
        norm_momentum = 0.97,
        norm_epsilon = 0.0001,
        activation = None, #"mish", 
        use_separable_conv = False, 
        kernel_initializer='TruncatedNormal',
        kernel_regularizer=None,
        bias_initializer='zeros',
        bias_regularizer=None, 
      )

    self.act = _get_activation_fn("mish")
    return 
  
  def call(self, inputs):
    # if self._merge_token_instructions:
    #   lp_impa = self.impa_mlp(tf.concat([inputs[1], self.impa], axis = -1))
    #   lp_impm = self.impa_mlp(tf.concat([inputs[2], self.impm], axis = -1))
    # else:
    # lp_impa = self.impa
    # lp_impm = self.impm

    x = inputs[0]
    #x = x + lp_impa
    x = self.projection(x) # 
    #x = x * lp_impm
    x = self.act(x)
    return x, x, x #lp_impa, lp_impm

class DeTokenizer(Tokenizer):
  
  def call(self, inputs):
    #if self._merge_token_instructions:
    y = inputs[1]
    # lp_impa = self.impa_mlp(tf.concat(inputs[1], axis = -1))
    # lp_impm = self.impa_mlp(tf.concat(inputs[2], axis = -1))
    # else:
    #   lp_impa = self.impa
    #   lp_impm = self.impm

    x = inputs[0]

    x = x + y
    #x = x * lp_impm
    x = self.projection(x)
    #x = x + lp_impa
    x = self.act(x)
    return x

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
               use_sync_bn=True,
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


class LayerNorm(tf.keras.layers.Layer):

  def build(self, input_shape):
    self.norm = _get_norm_fn("layer_norm", epsilon=0.0001, axis = -1)()
    return
  
  def call(self, x):
    _, H, W, C = x.shape
    x = tf.reshape(x, [-1, H * W, C])
    x = self.norm(x) 
    x = tf.reshape(x, [-1, H , W, C])
    return x

class TBiFPN(tf.keras.Model):

  def __init__(self, 
               input_specs, 
               embeding_dims = None, 
               window_size = 8.0, 
               mlp_ratio = 2.0,
               token_size = 32, 
               activation = "mish",
               repititions = 1, 
               fpn_only = False,
               tokenize = True, 
               **kwargs) -> None:
    self._embeding_dims = embeding_dims
    self._window_size = window_size or input_specs[str(max(input_specs.keys()))][1]
    self._mlp_ratio = mlp_ratio
    self._token_size = token_size
    self._activation = activation
    self._repititions = repititions
    self._tokenize = tokenize

    inputs = {
        key: tf.keras.layers.Input(shape=value[1:]) for key, value in input_specs.items()
    }

    outputs = inputs
    for i in range(self._repititions):
      outputs, implicits = self.build_tokenizer(outputs)
      fpn_only_ = False
      if i == self._repititions - 1:
        fpn_only_ = fpn_only
      outputs = self.build_merging(outputs, fpn_only_)
      outputs = self.build_detokenizer(outputs, implicits)

    self._output_specs = {key: value.shape for key, value in outputs.items()}
    super().__init__(inputs=inputs, outputs=outputs, name='YoloDecoder')
    return

  def layer_norm(self, x):
    # _, H, W, C = x.shape
    # x = tf.reshape(x, [-1, H * W, C])
    # x = _get_norm_fn("layer_norm", epsilon=0.0001)(axis = -1)(x) 
    # x = tf.reshape(x, [-1, H , W, C])
    return LayerNorm()(x)
  
  def activation(self, x):
    return _get_activation_fn(self._activation)(x)


  def conv(self, x, output_features):
    x = nn_blocks.ConvBN( # after shift, project the channels to similar arangement 
        filters = output_features,
        groups = 1, 
        kernel_size = 1, 
        strides = 1, 
        padding = "same", 
        use_bias = False,
        use_bn = True,
        use_sync_bn = True,
        norm_momentum = 0.97,
        norm_epsilon = 0.0001,
        activation = self._activation, 
        use_separable_conv = False, 
        kernel_initializer='TruncatedNormal',
        kernel_regularizer=None,
        bias_initializer='zeros',
        bias_regularizer=None, 
      )(x)
    return x

  def dw_conv(self, x, stride):
    x = tf.keras.layers.DepthwiseConv2D(
                              3, strides = stride, 
                              padding = "same", 
                              use_bias = False, 
                              kernel_initializer='TruncatedNormal',
                              kernel_regularizer=None,
                              bias_initializer='zeros',
                              bias_regularizer=None)(x)
    x = tf.keras.layers.BatchNormalization(
      momentum = 0.97,
      epsilon = 0.0001,
    )(x)
    return self.activation(x)

  def squeeze_expand(self, x):
    num_elements = x.shape[-1]
    x = nn_blocks.ConvBN( # after shift, project the channels to similar arangement 
        filters = num_elements//2,
        groups = 1, 
        kernel_size = 1, 
        strides = 1, 
        padding = "same", 
        use_bias = False,
        use_bn = True,
        use_sync_bn = True,
        norm_momentum = 0.97,
        norm_epsilon = 0.0001,
        activation = self._activation, 
        use_separable_conv = False, 
        kernel_initializer='TruncatedNormal',
        kernel_regularizer=None,
        bias_initializer='zeros',
        bias_regularizer=None, 
      )(x)
    x = nn_blocks.ConvBN( # after shift, project the channels to similar arangement 
        filters = num_elements,
        groups = 1, 
        kernel_size = 3, 
        strides = 1, 
        padding = "same", 
        use_bias = False,
        use_bn = True,
        use_sync_bn = True,
        norm_momentum = 0.97,
        norm_epsilon = 0.0001,
        activation = self._activation, 
        use_separable_conv = False, 
        kernel_initializer='TruncatedNormal',
        kernel_regularizer=None,
        bias_initializer='zeros',
        bias_regularizer=None, 
      )(x)
    return x

  def tokenize(self, x, impa, impm):
    num_elements = x.shape[-1]
    x = self.squeeze_expand(x)

    tokenizer = Tokenizer(embedding_dims=self._embeding_dims)
    if impa is None and impm is None:
      x, impa, impm = tokenizer([x])
    else:
      x, impa, impm = tokenizer([x, impa, impm])

    # x = self.activation(x)
    return x, impa, impm
  
  def build_tokenizer(self, inputs):
    outputs = {}
    implicits = {}
    impa = None
    impm = None 
    for k, v in inputs.items():
      if self._tokenize:
        v, impa, impm = self.tokenize(v, impa, impm)
      else:
        impa, impm = v, v
      outputs[k] = v
      implicits[k] = (impa, impm)
    return outputs, implicits

  def detokenize(self, x, imp):
    num_elements = x.shape[-1]
    detokenizer = DeTokenizer(embedding_dims=self._embeding_dims)
    x = detokenizer([x, imp[0], imp[1]])
    x = self.squeeze_expand(x)
    # x = self.activation(x)
    return x
  
  def build_detokenizer(self, inputs, imps):
    outputs = {}
    for k, v in inputs.items():
      outputs[k] = self.detokenize(v, imps[k])
    return outputs

  def ffn(self, x, output_features = None):
    return FFN(hidden_features=int(x.shape[-1] * self._mlp_ratio),
                out_features=output_features, 
                strides = 1, 
                kernel_size = 1, 
                activation = self._activation,
                groups = 1, 
                use_separable_conv = False, 
                use_bn = True,
                use_sync_bn = True,
                norm_momentum = 0.97,
                norm_epsilon = 0.001)(x)

  # # upsample self attention
  # def merge_levels(self, source, query, oquery = None):
  #   num_heads = query.shape[-1]//self._token_size
  #   exp_features = query.shape[-1]
    
  #   sample_size = query.shape[1]/source.shape[1]
  #   source_window_size = self._window_size

  #   if sample_size > 1:
  #     sample_size = int(sample_size)
  #     upsample = True
  #   elif sample_size < 1:
  #     sample_size = int(source.shape[1]/query.shape[1])
  #     query_window_size = int(self._window_size * sample_size)
  #     query_window_size, source_window_size = source_window_size, query_window_size
  #     upsample = False
  #   else:
  #     upsample = None

  #   attention = ShiftedWindowSelfAttention( 
  #     source_window_size, 
  #     shift = False, 
  #     num_heads = num_heads, 
  #     kernel_size = 1, 
  #     use_separable_conv = False, 
  #     qkv_bias=True, 
  #     qk_scale=None, 
  #     project_attention = False,
  #   )

  #   if upsample == True:
  #     source = tf.keras.layers.UpSampling2D(sample_size)(source)
  #   elif upsample == False:
  #     source = self.dw_conv(source, sample_size)
  #   else:
  #     source = source

  #   source = self.conv(source, exp_features)
  #   x = query + source

  #   ns = self.layer_norm(x)
  #   x = attention(ns) + x 
  #   x = self.ffn(x)
  #   return x
  
  # # upsample mapping 
  # def merge_levels(self, source, query, oquery = None):
  #   num_heads = query.shape[-1]//self._token_size
  #   exp_features = query.shape[-1]
    
  #   sample_size = query.shape[1]/source.shape[1]
  #   query_window_size = int(self._window_size * sample_size)
  #   source_window_size = self._window_size

  #   if sample_size > 1:
  #     sample_size = int(sample_size)
  #     upsample = True
  #   elif sample_size < 1:
  #     sample_size = int(source.shape[1]/query.shape[1])
  #     query_window_size = int(self._window_size * sample_size)
  #     query_window_size, source_window_size = source_window_size, query_window_size
  #     upsample = False
  #   else:
  #     upsample = None

  #   attention = ShiftedWindowAttention(
  #     query_window_size, 
  #     source_window_size, 
  #     shift = False, 
  #     num_heads = num_heads, 
  #     kernel_size = 1, 
  #     use_separable_conv = False, 
  #     qkv_bias=True, 
  #     qk_scale=None, 
  #     project_attention = False,
  #   )

  #   source = self.conv(source, exp_features)

  #   nq = self.layer_norm(query)
  #   ns = self.layer_norm(source)
  #   x = attention([nq, ns])

  #   if upsample == True:
  #     source = tf.keras.layers.UpSampling2D(sample_size)(source)
  #   elif upsample == False:
  #     source = self.dw_conv(source, sample_size)
  #   else:
  #     source = source
  #   return self.ffn(x + source)

  # upsample mapping 
  def merge_levels(self, source, query, oquery = None):
    num_heads = query.shape[-1]//self._token_size
    exp1_features = query.shape[-1]
    exp2_features = source.shape[-1]
    
    sample_size = query.shape[1]/source.shape[1]
    query_window_size = int(self._window_size * sample_size)
    source_window_size = self._window_size

    if sample_size > 1:
      sample_size = int(sample_size)
      upsample = True
    elif sample_size < 1:
      sample_size = int(source.shape[1]/query.shape[1])
      query_window_size = int(self._window_size * sample_size)
      query_window_size, source_window_size = source_window_size, query_window_size
      upsample = False
    else:
      upsample = None

    attention = ShiftedWindowAttention(
      query_window_size, 
      source_window_size, 
      shift = False, 
      num_heads = num_heads, 
      kernel_size = 1, 
      use_separable_conv = False, 
      qkv_bias=True, 
      qk_scale=None, 
      project_attention = False,
    )

    if exp1_features != exp2_features:
      query = self.conv(query, exp2_features)

    nq = self.layer_norm(query)
    ns = self.layer_norm(source)
    x = attention([nq, ns])
    
    # # config one 
    if upsample == True:
      source = tf.keras.layers.UpSampling2D(sample_size)(source)
    elif upsample == False:
      source = self.dw_conv(source, sample_size)
    else:
      source = source
    x = tf.keras.layers.Add()([x, source])

    # config 2 and 3
    # x = tf.keras.layers.Add()([x, query])
    return self.ffn(x, output_features=exp1_features)

  # upsample mapping 
  def merge_levels_down(self, source, query, oquery = None):
    num_heads = query.shape[-1]//self._token_size
    exp1_features = query.shape[-1]
    exp2_features = source.shape[-1]
    
    sample_size = query.shape[1]/source.shape[1]
    query_window_size = int(self._window_size * sample_size)
    source_window_size = self._window_size

    if sample_size > 1:
      sample_size = int(sample_size)
      upsample = True
    elif sample_size < 1:
      sample_size = int(source.shape[1]/query.shape[1])
      query_window_size = int(self._window_size * sample_size)
      query_window_size, source_window_size = source_window_size, query_window_size
      upsample = False
    else:
      upsample = None

    attention = ShiftedWindowAttention(
      query_window_size, 
      source_window_size, 
      shift = False, 
      num_heads = num_heads, 
      kernel_size = 1, 
      use_separable_conv = False, 
      qkv_bias=True, 
      qk_scale=None, 
      project_attention = False,
    )

    if exp1_features != exp2_features:
      query = self.conv(query, exp2_features)

    nq = self.layer_norm(query)
    ns = self.layer_norm(source)
    x = attention([nq, ns])
    
    # # config one 
    # if upsample == True:
    #   source = tf.keras.layers.UpSampling2D(sample_size)(source)
    # elif upsample == False:
    #   source = self.dw_conv(source, sample_size)
    # else:
    #   source = source
    x = tf.keras.layers.Add()([x, query])

    # config 2 and 3
    # x = tf.keras.layers.Add()([x, query])
    return self.ffn(x, output_features=exp1_features)

  def build_merging(self, inputs, fpn_only):
    outputs = inputs
    min_level = int(min(outputs.keys()))
    max_level = int(max(outputs.keys()))

    #if fpn_only:
    outputs[str(max_level)] = self.merge_levels(outputs[str(max_level)], outputs[str(max_level)])
      
    #up_merge
    for i in reversed(range(min_level, max_level)):
      level = outputs[str(i)]
      value = self.merge_levels(outputs[str(i + 1)], outputs[str(i)])
      # if i != min_level and i != max_level:
      #   print("here", i)
      #   outputs[str(i)] = self.merge_levels(level, outputs[str(i)])
      outputs[str(i)] = value

    # down_merge
    if not fpn_only:
      for i in range(min_level, max_level):
        outputs[str(i + 1)] = self.merge_levels(outputs[str(i)], outputs[str(i + 1)])


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

  built = {}
  for k, s in inputs.items():
    built[k] = tf.ones(inputs[k])

  # layer = Tokenizer()
  # p = layer([built['3']])

  # layer = Tokenizer()
  # p = layer([built['4'], p[1], p[2]])

  # layer = DeTokenizer()
  # p = layer(p)

  # print(p)
  m = TBiFPN(inputs)
  m.summary()



  # m(built)
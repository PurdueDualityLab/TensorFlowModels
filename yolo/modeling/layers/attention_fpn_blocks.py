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


class WindowedMultiHeadSelfAttention(tf.keras.layers.Layer):

  """Windowd multi head attention with a location based relative bias that is learned. 
  
  Functionally equivalent to multi head self attention but adds a gather for the 
  positional bias. 
  """

  def __init__(self, 
               window_size, 
               num_heads, 
               qkv_bias = True, 
               qk_scale = None, 
               project_attention = True, 
               attention_dropout = 0.0, 
               projection_dropout = 0.0, 
               kernel_initializer='TruncatedNormal',
               kernel_regularizer=None,
               bias_initializer='zeros',
               bias_regularizer=None, 
               relative_bias_initializer='TruncatedNormal', 
               attention_activation = 'softmax', # typically jsut soft max, more for future developments of something better 
               **kwargs):
    super().__init__(**kwargs)

    if isinstance(window_size, int):
      window_size = (window_size, window_size)

    self._window_size = window_size # wH, wW
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
    self.dim = input_shape[-1]
    head_dims = self.dim//self._num_heads
    self.scale = self._qk_scale or head_dims ** -0.5 # x/sqrt(num_heads) = x/sqrt(d_k)

    # biases to apply to each "position" learned 
    num_elements = (2*self._window_size[0] - 1) * (2*self._window_size[1] - 1)
    self._realtive_positional_bias_table = self.add_weight(
      name = '{}_realtive_positional_bias_table'.format(self.name), 
      shape = [num_elements, self._num_heads], 
      initializer = self._relative_bias_initializer, 
      regularizer = self._relative_bias_regularizer, 
      trainable = True)

    # get the postions to associate the above bais table with    
    coords_h = np.arange(self._window_size[0])
    coords_w = np.arange(self._window_size[1])
    coords = np.stack(np.meshgrid(coords_h, coords_w, indexing='ij')) # 2, Wh, Ww
    coords_flatten = coords.reshape(2, -1) # 2, Wh*Ww
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :] # 2, Wh*Ww, Wh*Ww
    relative_coords = relative_coords.transpose([1, 2, 0]) # Wh*Ww, Wh*Ww, 2
    relative_coords[:, :, 0] += self._window_size[0] - 1 # shift to start from 0
    relative_coords[:, :, 1] += self._window_size[1] - 1
    relative_coords[:, :, 0] *= 2 * self._window_size[1] - 1
    relative_positional_indexes = relative_coords.sum(-1) # Wh*Ww, Wh*Ww
    self._relative_positional_indexes = tf.Variable(
      initial_value=tf.convert_to_tensor(relative_positional_indexes), 
      trainable=False, 
      name='{}_realtive_indexes'.format(self.name))

    self.qkv = tf.keras.layers.Dense(
        self.dim * 3, use_bias = self._qkv_bias, **self._init_args)

    self.attn_drop = tf.keras.layers.Dropout(self._attention_dropout)
    self.act = _get_activation_fn(self._attention_activation) # softmax

    # per attention projection
    if self._project_attention:
      self.proj = tf.keras.layers.Dense(self.dim, **self._init_args)
      self.proj_drop = tf.keras.layers.Dropout(self._projection_dropout)
    return 
  
  def get_indexed_bias(self):
    # compute the relative poisiton bias
    num_elems = self._window_size[0] * self._window_size[1]
    indexes = tf.reshape(self._relative_positional_indexes, [-1])
    relative_position_bias = tf.gather(self._realtive_positional_bias_table, indexes)
    relative_position_bias = tf.reshape(relative_position_bias, [num_elems, num_elems, -1]) # Wh*Ww,Wh*Ww,nH
    relative_position_bias = tf.transpose(relative_position_bias, perm=(2, 0, 1)) # nH, Wh*Ww, Wh*Ww
    return tf.expand_dims(relative_position_bias, axis = 0)

  def get_embedding(self, x, N, C):
    qkv = self.qkv(x)
    qkv = tf.reshape(qkv, [-1, N, 3, self._num_heads, C // self._num_heads])
    qkv = tf.einsum("bnthc->tbhnc", qkv)
    return qkv[0], qkv[1], qkv[2]

  def call(self, x, mask = None, training = None):
    _, N, C = x.shape

    q, k, v = self.get_embedding(x, N, C)

    # compute the matrix mul attention
    q = q * self.scale
    attn = tf.matmul(q, k, transpose_b = True)

    # compute the relative poisiton bias
    relative_position_bias = self.get_indexed_bias()
    attn = attn + relative_position_bias

    if mask is not None:
      num_windows = mask.shape[0]
      mask = tf.cast(tf.expand_dims(tf.expand_dims(mask, axis = 1), axis = 0), attn.dtype)
      attn = tf.reshape(attn, [-1, num_windows, self._num_heads, N, N]) + mask
      attn = tf.reshape(attn, [-1, self._num_heads, N, N])

    attn = self.act(attn)

    if training:
      attn = self.attn_drop(attn)

    x = tf.einsum("bhij,bhjk->bihk", attn, v)
    x = tf.reshape(x, [-1, N, C])

    if self._project_attention:
      x = self.proj(x)
      if training:
        x = self.proj_drop(x)
    return x

class WindowedMultiHeadAttention(tf.keras.layers.Layer):

  """Windowd multi head attention with a location based relative bias that is learned. 
  
  Functionally equivalent to multi head self attention but adds a gather for the 
  positional bias. 
  """

  def __init__(self, 
               window_size, 
               num_heads, 
               qkv_bias = True, 
               qk_scale = None, 
               project_attention = True, 
               attention_dropout = 0.0, 
               projection_dropout = 0.0, 
               kernel_initializer='TruncatedNormal',
               kernel_regularizer=None,
               bias_initializer='zeros',
               bias_regularizer=None, 
               relative_bias_initializer='TruncatedNormal', 
               attention_activation = 'softmax', # typically jsut soft max, more for future developments of something better 
               **kwargs):
    super().__init__(**kwargs)

    if isinstance(window_size, int):
      window_size = (window_size, window_size)

    self._window_size = window_size # wH, wW
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

    query_shape = input_shape[0]
    self.dim = query_shape[-1]
    head_dims = self.dim//self._num_heads
    self.scale = self._qk_scale or head_dims ** -0.5 # x/sqrt(num_heads) = x/sqrt(d_k)

    # biases to apply to each "position" learned 
    num_elements = (2*self._window_size[0] - 1) * (2*self._window_size[1] - 1)
    self._realtive_positional_bias_table = self.add_weight(
      name = '{}_realtive_positional_bias_table'.format(self.name), 
      shape = [num_elements, self._num_heads], 
      initializer = self._relative_bias_initializer, 
      regularizer = self._relative_bias_regularizer, 
      trainable = True)

    # get the postions to associate the above bais table with    
    coords_h = np.arange(self._window_size[0])
    coords_w = np.arange(self._window_size[1])
    coords = np.stack(np.meshgrid(coords_h, coords_w, indexing='ij')) # 2, Wh, Ww
    coords_flatten = coords.reshape(2, -1) # 2, Wh*Ww
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :] # 2, Wh*Ww, Wh*Ww
    relative_coords = relative_coords.transpose([1, 2, 0]) # Wh*Ww, Wh*Ww, 2
    relative_coords[:, :, 0] += self._window_size[0] - 1 # shift to start from 0
    relative_coords[:, :, 1] += self._window_size[1] - 1
    relative_coords[:, :, 0] *= 2 * self._window_size[1] - 1
    relative_positional_indexes = relative_coords.sum(-1) # Wh*Ww, Wh*Ww
    self._relative_positional_indexes = tf.Variable(
      initial_value=tf.convert_to_tensor(relative_positional_indexes), 
      trainable=False, 
      name='{}_realtive_indexes'.format(self.name))

    self.q = tf.keras.layers.Dense(
        self.dim, use_bias = self._qkv_bias, **self._init_args)
    self.kv = tf.keras.layers.Dense(
        self.dim * 2, use_bias = self._qkv_bias, **self._init_args)

    self.attn_drop = tf.keras.layers.Dropout(self._attention_dropout)
    self.act = _get_activation_fn(self._attention_activation) # softmax

    # per attention projection
    if self._project_attention:
      self.proj = tf.keras.layers.Dense(self.dim, **self._init_args)
      self.proj_drop = tf.keras.layers.Dropout(self._projection_dropout)
    return 
  
  def get_indexed_bias(self):
    # compute the relative poisiton bias
    num_elems = self._window_size[0] * self._window_size[1]
    indexes = tf.reshape(self._relative_positional_indexes, [-1])
    relative_position_bias = tf.gather(self._realtive_positional_bias_table, indexes)
    relative_position_bias = tf.reshape(relative_position_bias, [num_elems, num_elems, -1]) # Wh*Ww,Wh*Ww,nH
    relative_position_bias = tf.transpose(relative_position_bias, perm=(2, 0, 1)) # nH, Wh*Ww, Wh*Ww
    return tf.expand_dims(relative_position_bias, axis = 0)

  def get_embedding(self, query, source, N, C):
    q = self.q(query)
    kv = self.kv(source)
    qkv = tf.concat([q, kv], axis = -1)
    qkv = tf.reshape(qkv, [-1, N, 3, self._num_heads, C // self._num_heads])
    qkv = tf.einsum("bnthc->tbhnc", qkv)
    return qkv[0], qkv[1], qkv[2]

  def call(self, query, source, mask = None, training = None):
    _, N, C = query.shape

    q, k, v = self.get_embedding(query, source, N, C)

    # compute the matrix mul attention
    q = q * self.scale
    attn = tf.matmul(q, k, transpose_b = True)

    # compute the relative poisiton bias
    relative_position_bias = self.get_indexed_bias()
    attn = attn + relative_position_bias

    if mask is not None:
      num_windows = mask.shape[0]
      mask = tf.cast(tf.expand_dims(tf.expand_dims(mask, axis = 1), axis = 0), attn.dtype)
      attn = tf.reshape(attn, [-1, num_windows, self._num_heads, N, N]) + mask
      attn = tf.reshape(attn, [-1, self._num_heads, N, N])

    attn = self.act(attn)

    if training:
      attn = self.attn_drop(attn)

    x = tf.einsum("bhij,bhjk->bihk", attn, v)
    x = tf.reshape(x, [-1, N, C])

    if self._project_attention:
      x = self.proj(x)
      if training:
        x = self.proj_drop(x)
    return x

class ShiftedWindowMultiHeadAttention(tf.keras.layers.Layer):

  def __init__(self, 
               window_size, 
               num_heads, 
               use_shortcut = True, 
               shift = SHIFT, 
               pre_norm = None, 
               shift_size = None, 
               qkv_bias = True, 
               qk_scale = None, 
               project_attention = True, 
               attention_dropout = 0.0, 
               projection_dropout = 0.0, 
               drop_path = 0.0, 
               kernel_initializer='TruncatedNormal',
               kernel_regularizer=None,
               bias_initializer='zeros',
               bias_regularizer=None, 
               relative_bias_initializer='TruncatedNormal', 
               attention_activation = 'softmax', # typically jsut soft max, more for future developments of something better 
               **kwargs):

    super().__init__(**kwargs)

    if isinstance(window_size, int):
      window_size = (window_size, window_size)

    self._window_size = window_size # wH, wW
    self._num_heads = num_heads
    self._qkv_bias = qkv_bias
    self._qk_scale = qk_scale 
    self._shift = shift 

    self._project_attention = project_attention

    # dropout
    self._attention_dropout = attention_dropout
    self._projection_dropout = projection_dropout

    # activation 
    self._attention_activation = attention_activation

    self._shift_size = shift_size or min(window_size) // 2
    self._use_shortcut = use_shortcut
    self._drop_path = drop_path
    self._pre_norm = _get_norm_fn(pre_norm)

    # init and regularizer
    self._relative_bias_initializer = _get_initializer(relative_bias_initializer)
    self._relative_bias_regularizer = bias_regularizer
    self._init_args = dict(
      kernel_initializer = _get_initializer(kernel_initializer),
      bias_initializer = bias_initializer,
      kernel_regularizer = kernel_regularizer,
      bias_regularizer = bias_regularizer,
    )
  
  def build(self, input_shape):

    if min(input_shape[1:-1]) <= min(self._window_size):
      # we cannot shift and the eindow size must be the input resolution 
      self._shift_size = 0
      self._window_size = [min(input_shape[1:-1]), min(input_shape[1:-1])]

    if not (0 <= self._shift_size < min(self._window_size)):
      raise ValueError("the shift must be contained between 0 and wndow_size "
                        "or the image will be shifted outside of the partition " 
                        "window. ")

    if self._drop_path > 0.0:
      self.drop_path = nn_layers.StochasticDepth(self._drop_path)
    else:
      self.drop_path = Identity()

    self.pre_norm = self._pre_norm
    if self._pre_norm is not None:
      self.pre_norm = self._pre_norm()

    self._attention_layer = WindowedMultiHeadSelfAttention(
      window_size=self._window_size, 
      num_heads=self._num_heads, 
      qkv_bias=self._qkv_bias, 
      qk_scale=self._qk_scale, 
      project_attention=True, 
      attention_dropout=self._attention_dropout, 
      projection_dropout=self._projection_dropout, 
      relative_bias_initializer=self._relative_bias_initializer, 
      attention_activation=self._attention_activation, 
      **self._init_args
    )

    return 

  def call(self, x, mask = None, training = None):
    shortcut = x 
    if self.pre_norm is not None: # pre norm
      if isinstance(self._pre_norm, tf.keras.layers.LayerNormalization):
        _, H, W, C = x.shape
        x = tf.reshape(x, [-1, H * W, C])
        x = self.pre_norm(x)
        x = tf.reshape(x, [-1, H, W, C])
      else:
        x = self.pre_norm(x)

    _, H, W, C = x.shape
    pad_l = pad_t = 0 
    pad_r = (self._window_size[1] - W % self._window_size[1]) % self._window_size[1]
    pad_b = (self._window_size[0] - H % self._window_size[0]) % self._window_size[0]  
    x = tf.pad(x, [[0,0], [pad_t, pad_b], [pad_l, pad_r], [0, 0]]) 
    _, Hp, Wp, _ = x.shape

    if self._shift  == True:
      shifts = [(0, 0), (self._shift_size, self._shift_size)] # 6 ms latency, 9 ms latency 
    elif self._shift is None:
      shifts = [(self._shift_size, self._shift_size)]
    else:
      shifts = [(0, 0)] # 6 ms latency

    x_output = None
    windows = []
    bsize = []
    for shift in shifts:
      # cyclic shift 
      if shift[0] != 0 or shift[1] != 0:
        shifted_x = x[:, (shift[0]):(Hp - (self._window_size[0] - shift[0])), (shift[1]):(Wp - (self._window_size[1] - shift[1])), :]
        attn_mask = None
      else:
        shifted_x = x
        attn_mask = None 

      x_windows = window_partition(shifted_x, self._window_size) # nW*B, window_size, window_size, C
      windows.append(x_windows)

      nwin = tf.shape(x_windows)
      bsize.append(nwin[0])

    x_windows = tf.concat(windows, axis = 0)
    x_windows = tf.reshape(x_windows, [-1, self._window_size[0] * self._window_size[1], C]) # nW*B, window_size*window_size, C
    attn_windows = self._attention_layer(x_windows, mask=attn_mask, training=training)
    attn_windows = tf.reshape(attn_windows, [-1, self._window_size[0], self._window_size[1], C]) # nW*B, window_size*window_size, C
    windows = tf.split(attn_windows, bsize, axis = 0)

    for shift, attn_windows in zip(shifts, windows):
      if shift[0] != 0 or shift[1] != 0:
        shifted_x = window_reverse(attn_windows, self._window_size, (Hp - self._window_size[0]), (Wp - self._window_size[1])) # B H' W' C
        shifted_x = tf.pad(shifted_x, [[0,0], [shift[0], self._window_size[0] - shift[0]], [shift[1], self._window_size[1] - shift[1]], [0, 0]]) 
      else:
        shifted_x = window_reverse(attn_windows, self._window_size, Hp, Wp) # B H' W' C

      if x_output is None:
        x_output = shifted_x
      else:
        x_output = x_output + shifted_x

    if pad_r > 0 or pad_b > 0:
      x_output = x_output[:, :H, :W, :]

    if self._use_shortcut:
      if training:
        x_output = self.drop_path(x_output)
      x = shortcut + x_output
    return x

class FFN(tf.keras.layers.Layer):

  def __init__(self, 
               kernel_size = (3,3),
               strides = (1, 1), 
               num_heads = 1, 
               hidden_features = None, 
               out_features = None, 
               kernel_initializer='TruncatedNormal',
               kernel_regularizer=None,
               bias_initializer='zeros',
               bias_regularizer=None,
               activation = "gelu", 
               leaky_alpha=0.1,
               dropout = 0.0, 
               cat_input = True, 
               **kwargs):
    super().__init__(**kwargs)

    # features 
    self._kernel_size = kernel_size
    self._strides = strides
    self._num_heads = num_heads
    self._hidden_features = hidden_features
    self._out_features = out_features
    self._cat_input = cat_input

    if USE_SYNC_BN:
      self._norm_layer = _get_norm_fn("sync_batch_norm")
    else:
      self._norm_layer = _get_norm_fn("batch_norm")

    # init and regularizer
    self._init_args = dict(
      kernel_initializer = _get_initializer(kernel_initializer),
      bias_initializer = bias_initializer,
      kernel_regularizer = kernel_regularizer,
      bias_regularizer = bias_regularizer,
    )

    # activation
    self._activation = activation
    self._leaky = leaky_alpha
    self._dropout = dropout

  def build(self, input_shape):
    hidden_features = self._hidden_features or input_shape[-1]
    out_features = self._out_features or input_shape[-1]

    self.dw_spatial_sample = nn_blocks.ConvBN(
                                      filters = hidden_features,
                                      groups = self._num_heads, # attention seperation
                                      kernel_size = self._kernel_size, 
                                      strides = self._strides, 
                                      padding = "same", 
                                      activation = None,
                                      use_bias = True,  
                                      use_bn = True,
                                      use_sync_bn = USE_SYNC_BN,
                                      **self._init_args)
    self.norm_spatial_sample = self._norm_layer()

    self.fc_expand = nn_blocks.ConvBN(filters = hidden_features,
                                      kernel_size = (1, 1), 
                                      strides = (1, 1), 
                                      padding = "same", 
                                      activation = None,
                                      use_bias = True,  
                                      use_bn = True,
                                      use_sync_bn = USE_SYNC_BN,
                                      **self._init_args)

    self.fc_compress = nn_blocks.ConvBN(filters = out_features, 
                                        kernel_size = (1, 1), 
                                        strides = (1, 1), 
                                        padding = "same", 
                                        activation = None, 
                                        use_bias = True,  
                                        use_bn = True,
                                        use_sync_bn = USE_SYNC_BN,
                                        **self._init_args)
    self.act = _get_activation_fn(self._activation, leaky_alpha=self._leaky)
    return 

  def call(self, x_in):
    x = self.dw_spatial_sample(x_in)
    x = self.norm_spatial_sample(x)
    x = self.act(x)

    x = self.fc_expand(x)
    x = self.act(x)

    x = self.fc_compress(x)
    x = self.act(x)
    return x

class SwinTransformerLayer(tf.keras.layers.Layer):

  def __init__(self, 
               num_heads, 
               window_size = 7, 
               shift_size = 0, 
               shift = SHIFT, 
               mlp_ratio = 4, 
               qkv_bias = True, 
               qk_scale = None, 
               dropout = 0.0, 
               attention_dropout = 0.0, 
               drop_path = 0.0, 
               activation = 'gelu',
               pre_norm = "layer_norm", 
               norm_layer = 'layer_norm',
               kernel_initializer='TruncatedNormal',
               kernel_regularizer=None,
               bias_initializer='zeros',
               bias_regularizer=None, 
               cat_input = True, 
               **kwargs):
    super().__init__(**kwargs)

    self._num_heads = num_heads
    self._window_size = window_size
    self._shift_size = shift_size
    self._shift = shift
    
    self._mlp_ratio = mlp_ratio
    self._qkv_bias = qkv_bias
    self._qk_scale = qk_scale
    
    self._dropout = dropout 
    self._attention_dropout = attention_dropout
    self._drop_path = drop_path
    self._cat_input = cat_input

    self._activation = activation
    if USE_SYNC_BN:
      self._norm_layer_key = "sync_batch_norm"
      self._norm_layer = _get_norm_fn("sync_batch_norm")
    else:
      self._norm_layer_key = "batch_norm"
      self._norm_layer = _get_norm_fn("batch_norm")
    self._pre_norm_layer_key = pre_norm
    self._pre_norm_layer = _get_norm_fn(pre_norm)

    # init and regularizer
    self._init_args = dict(
      kernel_initializer = _get_initializer(kernel_initializer),
      bias_initializer = bias_initializer,
      kernel_regularizer = kernel_regularizer,
      bias_regularizer = bias_regularizer,
    )
    
  def build(self, input_shape):
    self._dims = input_shape[-1]

    self.attention = ShiftedWindowMultiHeadAttention(
      window_size=self._window_size, 
      num_heads=self._num_heads, 
      use_shortcut=True, 
      pre_norm=self._pre_norm_layer_key, 
      shift_size=self._shift_size, 
      qkv_bias=self._qkv_bias, 
      qk_scale=self._qk_scale, 
      shift=self._shift,
      attention_dropout=self._attention_dropout, 
      projection_dropout=self._dropout
    )

    if self._drop_path > 0.0:
      self.drop_path = nn_layers.StochasticDepth(self._drop_path)
    else:
      self.drop_path = Identity()
    
    self.prenorm = self._pre_norm_layer
    if self._pre_norm_layer is not None:
      self.prenorm = self._pre_norm_layer()
    
    
    mlp_hidden_dim = int(self._dims * self._mlp_ratio)
    self.mlp = LeFFN(hidden_features=mlp_hidden_dim, activation=self._activation, 
                   dropout=self._dropout, cat_input=self._cat_input, **self._init_args)
  
  def call(self, x, mask_matrix = None, training = None):
    x = self.attention(x, 
            mask = mask_matrix, 
            training = training) # empahsize significant items

    if self.prenorm is not None:
      _, H, W, C = x.shape
      x_interem = tf.reshape(x, [-1, H * W, C])
      x_interem = self.prenorm(x_interem)
      x_interem = tf.reshape(x_interem, [-1, H, W, C])
    else:
      x_interem = x

    x_interem = self.mlp(x_interem)
    if training:
      x_interem = self.drop_path(x_interem)
    x = x + x_interem
    return x





if __name__ == "__main__":
  inputs = {"3": [2, 56, 56, 256], "4": [2, 28, 28, 512], "5": [2, 14, 14, 1024]}

  built = {}
  for k, s in inputs.items():
    built[k] = tf.ones(inputs[k])

  # input1 = tf.reshape(built["5"], [1, -1, 256])
  # layer = MLP(10, 10, activation="relu")
  # output = layer(input1)
  # print(output.shape)
  
  # input1 = built["4"]
  # layer = WindowPartition(7)
  # output = layer(input1)
  # print(output.shape)

  # output = tf.reshape(output, [-1, 7 * 7, output.shape[-1]])
  # layer = WindowedMultiHeadAttention(7, 32)
  # output, _ = layer(output)
  # output = tf.reshape(output, [-1, 7, 7, output.shape[-1]])
  # print(output.shape)

  # layer = WindowReverse(28, 28)
  # output = layer(output)
  # print(output.shape)


  # input1 = built["5"] #tf.reshape(built["5"], [2, -1, 1024])
  # layer = SwinTransformerLayer(8, shift_size=3)
  # output = layer(input1)
  # print(output.shape)

  # input1 = built["5"]
  # layer = PatchMerge()
  # output = layer(input1)
  # print(output.shape)

  # input1 = built["4"]
  # layer = PatchEmbed()
  # output = layer(input1)
  # print(output.shape)

  input1 = built["5"]
  layer = SwinTransformerBlock(2, 8, window_size=7)
  output = layer(input1)
  print(output.shape)
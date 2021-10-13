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
from typing import List, Tuple
from yolo.modeling.layers import nn_blocks
import tensorflow as tf
import numpy as np
from official.modeling import tf_utils
from official.vision.beta.modeling.layers import nn_layers
from functools import partial

USE_SYNC_BN = True
SHIFT = True
ALT_SHIFTS = False

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

  def call(self, x, mask = None, training = None):
    _, N, C = x.shape

    qkv = self.qkv(x)
    qkv = tf.reshape(qkv, [-1, N, 3, self._num_heads, C // self._num_heads])
    qkv = tf.transpose(qkv, perm=(2, 0, 3, 1, 4))
    q, k, v = qkv[0], qkv[1], qkv[2]
    
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

    x = tf.matmul(attn, v)
    x = tf.transpose(x, perm = (0, 2, 1, 3)) # move heads to be merged with features
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

    self._attention_layer = WindowedMultiHeadAttention(
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

    H, W = input_shape[1:-1]
    self.Hp = int(np.ceil(H / self._window_size[0]) * self._window_size[0])
    self.Wp = int(np.ceil(W / self._window_size[1]) * self._window_size[1])
    img_mask = np.zeros((1, self.Hp, self.Wp, 1))

    h_slices = (
      slice(0, -self._window_size[0]), 
      slice(-self._window_size[0], -self._shift_size), 
      slice(-self._shift_size, None)
    )
    w_slices = (
      slice(0, -self._window_size[1]), 
      slice(-self._window_size[1], -self._shift_size), 
      slice(-self._shift_size, None)
    )

    cnt = 0 
    for h in h_slices: 
      for w in w_slices:
        img_mask[:, h, w, :] = cnt 
        cnt += 1

    img_mask = tf.convert_to_tensor(img_mask)
    mask_windows = window_partition(img_mask, self._window_size)
    mask_windows = tf.reshape(mask_windows, [-1, self._window_size[0]*self._window_size[1]])

    attn_mask = tf.expand_dims(mask_windows, axis = 1) - tf.expand_dims(mask_windows, axis = 2)
    attn_mask = tf.where(attn_mask == 0, tf.cast(0.0, img_mask.dtype), attn_mask)
    attn_mask = tf.where(attn_mask != 0, tf.cast(-100.0, img_mask.dtype), attn_mask)

    self._attn_mask = attn_mask
    return 

  def _build_mask(self, dtype = 'float32'):
    return tf.cast(self._attn_mask, dtype)

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

    if self._shift:
      shifts = [(0, 0), (-self._shift_size, -self._shift_size)] # 6 ms latency, 9 ms latency 
    else:
      shifts = [(0, 0)] # 6 ms latency

    x_output = None
    for shift in shifts:
      # cyclic shift 
      if shift[0] != 0 or shift[1] != 0:
        shifted_x = tf.roll(x, shift=[-shift[0], -shift[1]], axis = [1, 2])
        attn_mask = self._build_mask(dtype = x.dtype) if mask is None else mask
      else:
        shifted_x = x
        attn_mask = None 

      x_windows = window_partition(shifted_x, self._window_size) # nW*B, window_size, window_size, C
      x_windows = tf.reshape(x_windows, [-1, self._window_size[0] * self._window_size[1], C]) # nW*B, window_size*window_size, C

      attn_windows = self._attention_layer(x_windows, mask=attn_mask, training=training)

      attn_windows = tf.reshape(attn_windows, [-1, self._window_size[0], self._window_size[1], C]) # nW*B, window_size*window_size, C
      shifted_x = window_reverse(attn_windows, self._window_size, Hp, Wp) # B H' W' C

      if shift[0] != 0 or shift[1] != 0:
        shifted_x = tf.roll(shifted_x, shift=[shift[0], shift[1]], axis = [1, 2])

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


class SPP(tf.keras.layers.Layer):
  """Spatial Pyramid Pooling.

  A non-agregated SPP layer that uses Pooling.
  """

  def __init__(self, 
               sizes, 
               strides = (1, 1),
               cat_input = True, 
               unweighted = False, 
               kernel_initializer='TruncatedNormal',
               kernel_regularizer=None,
               bias_initializer='zeros',
               bias_regularizer=None, 
               use_bn = False, 
               **kwargs):
    self._sizes = list(reversed(sizes))
    if not sizes:
      raise ValueError('More than one maxpool should be specified in SSP block')
    self._strides = strides
    self._cat_input = cat_input
    self._unweighted = unweighted
    self._use_bn = use_bn

    if use_bn:
      if USE_SYNC_BN:
        self._norm_layer = _get_norm_fn("sync_batch_norm")
      else:
        self._norm_layer = _get_norm_fn("batch_norm")

    self._init_args = dict(
      kernel_initializer = _get_initializer(kernel_initializer),
      bias_initializer = bias_initializer,
      kernel_regularizer = kernel_regularizer,
      bias_regularizer = bias_regularizer,
    )
    super().__init__(**kwargs)

  def _weighted(self, size):
    if self._use_bn:
      return  tf.keras.Sequential([tf.keras.layers.DepthwiseConv2D(
                                        (size, size), # spatial embedding combination and reduction
                                        strides = self._strides, 
                                        padding = "same", 
                                        use_bias = False, 
                                        **self._init_args), 
                                    self._norm_layer()])
    else:
      return  tf.keras.Sequential([tf.keras.layers.DepthwiseConv2D(
                                              (size, size), # spatial embedding combination and reduction
                                              strides = self._strides, 
                                              padding = "same", 
                                              use_bias = False, 
                                              **self._init_args)])

  def _pooling(self, size):
    return tf.keras.layers.MaxPool2D(
              pool_size=(size, size),
              strides=self._strides,
              padding='same',
              data_format=None)

  def build(self, input_shape):
    maxpools = []
    for size in self._sizes:
      if self._unweighted:
        maxpools.append(self._pooling(size))
      else:
        maxpools.append(self._weighted(size))
    self._maxpools = maxpools
    super().build(input_shape)

  def call(self, inputs, training=None):
    outputs = []
    for maxpool in self._maxpools:
      outputs.append(maxpool(inputs))
    if self._cat_input:
      outputs.append(inputs)
    concat_output = tf.keras.layers.concatenate(outputs)
    return concat_output

  def get_config(self):
    layer_config = {'sizes': self._sizes}
    layer_config.update(super().get_config())
    return layer_config

class LeFFN(tf.keras.layers.Layer):

  def __init__(self, 
               strides = (1, 1), 
               hidden_features = None, 
               out_features = None, 
               kernel_initializer='TruncatedNormal',
               kernel_regularizer=None,
               bias_initializer='zeros',
               bias_regularizer=None,
               activation = "gelu", 
               leaky_alpha=0.1,
               dropout = 0.0, 
               **kwargs):
    super().__init__(**kwargs)

    # features 
    self._strides = strides
    self._hidden_features = hidden_features
    self._out_features = out_features

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

    self.dw_spatial_sample = SPP([3], strides = 1, cat_input = True, **self._init_args) # per layer spatial binning
    self.norm_spatial_sample = self._norm_layer()

    self.fc_expand = nn_blocks.ConvBN(filters = hidden_features,
                                      kernel_size = (1, 1), 
                                      strides = (1, 1), 
                                      padding = "same", 
                                      activation = None,
                                      use_bias = True,  
                                      use_sync_bn = USE_SYNC_BN,
                                      **self._init_args)

    self.fc_compress = nn_blocks.ConvBN(filters = out_features, 
                                        kernel_size = (1, 1), 
                                        strides = (1, 1), 
                                        padding = "same", 
                                        activation = None, 
                                        use_bias = True,  
                                        use_sync_bn = USE_SYNC_BN,
                                        use_separable_conv = True,
                                        **self._init_args)
    self.act = _get_activation_fn(self._activation, leaky_alpha=self._leaky)
    return 

  def call(self, x):
    x = self.dw_spatial_sample(x)
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
                   dropout=self._dropout, **self._init_args)
  
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


class PatchMerge(tf.keras.layers.Layer):

  # reprojection and re-embedding

  def __init__(self,
               norm_layer = 'layer_norm',
               filter_scale = 2, 
               kernel_initializer='TruncatedNormal',
               kernel_regularizer=None,
               bias_initializer='zeros',
               bias_regularizer=None, 
               **kwargs):
    super().__init__(**kwargs)

    #default
    if USE_SYNC_BN:
      self._norm_layer = _get_norm_fn("sync_batch_norm")
    else:
      self._norm_layer = _get_norm_fn("batch_norm")

    self._filter_scale = filter_scale

    # init and regularizer
    self._init_args = dict(
      kernel_initializer = _get_initializer(kernel_initializer),
      bias_initializer = bias_initializer,
      kernel_regularizer = kernel_regularizer,
      bias_regularizer = bias_regularizer,
    )
  
  def build(self, input_shape):
    self._dims = input_shape[-1]

    self.expand = SPP([3, 5], strides = 2, cat_input = False) #, use_bn=True) # per layer spatial binning
    self.reduce = nn_blocks.ConvBN(filters = self._dims * self._filter_scale, # point wise token expantion
                                        kernel_size = (1, 1), 
                                        strides = (1, 1), 
                                        padding = "same", 
                                        activation = None, 
                                        use_bias = True,  
                                        use_bn = False, 
                                        use_separable_conv = True, 
                                        use_sync_bn = USE_SYNC_BN,
                                        **self._init_args)
    self.norm_reduce = self._norm_layer()
    
  
  def call(self, x):
    """Down sample by 2. """
    x = self.expand(x)
    x = self.reduce(x)
    x = self.norm_reduce(x)
    return x

class SwinTransformerBlock(tf.keras.layers.Layer):

  def __init__(self,
               depth, 
               num_heads, 
               window_size = 7, 
               mlp_ratio = 4, 
               qkv_bias = True, 
               qk_scale = None, 
               dropout = 0.0, 
               attention_dropout = 0.0, 
               drop_path = 0.0, 
               norm_layer = 'layer_norm', 
               downsample = 'patch_and_merge',
               activation = 'gelu',
               kernel_initializer='TruncatedNormal',
               kernel_regularizer=None,
               bias_initializer='zeros',
               bias_regularizer=None, 
               ignore_shifts = False, 
               **kwargs):
    super().__init__(**kwargs)

    self._depth = depth
    self._num_heads = num_heads
    
    self._window_size = window_size
    self._shift_size = window_size // 2

    self._norm_layer_key = norm_layer
    self._norm_layer = _get_norm_fn(norm_layer)
    self._downsample = downsample
    self._ignore_shifts = ignore_shifts

    # init and regularizer
    self._swin_args = dict(
       norm_layer=norm_layer,
       mlp_ratio=mlp_ratio,
       qkv_bias=qkv_bias,
       qk_scale=qk_scale,
       dropout=dropout,
       attention_dropout=attention_dropout,
       activation=activation, 
    )

    self._drop_path = drop_path
    self._init_args = dict(
      kernel_initializer = kernel_initializer,
      bias_initializer = bias_initializer,
      kernel_regularizer = kernel_regularizer,
      bias_regularizer = bias_regularizer,
    )
    self._swin_args.update(self._init_args)
  
  def build(self, input_shape):
    
    self.layers = []

    index_drop_path = isinstance(self._drop_path, List)
    for i in range(self._depth):
      if self._ignore_shifts:
        shift_size = 0
        window_size = self._window_size
      else:  
        shift_size = 0 if i % 2 == 0 else self._window_size // 2
        window_size = self._window_size
      
      drop_path = self._drop_path[i] if index_drop_path else self._drop_path

      shift = SHIFT if not ALT_SHIFTS else shift_size != 0
      layer = SwinTransformerLayer(self._num_heads, 
                                   window_size=window_size,
                                   shift_size=shift_size, 
                                   shift = shift, 
                                   drop_path=drop_path, 
                                   **self._swin_args)
      self.layers.append(layer)
    
    if self._downsample == "patch_and_merge":
      self.downsample = PatchMerge(norm_layer=self._norm_layer_key, **self._init_args)
    else:
      self.downsample = None
    return 

  def call(self, x):
    
    for layer in self.layers:
      x = layer(x) 

    x_down = x
    if self.downsample is not None:
      x_down = self.downsample(x)
    return x, x_down

class PatchEmbed(tf.keras.layers.Layer):

  def __init__(self,
               patch_size = 4, 
               embed_dimentions = 96, 
               norm_layer = None, 
               activation = None,
               absolute_positional_embed = False, 
               kernel_initializer='VarianceScaling',
               kernel_regularizer=None,
               bias_initializer='zeros',
               bias_regularizer=None, 
               **kwargs):
    super().__init__(**kwargs)

    self._patch_size = patch_size
    self._embed_dimentions = embed_dimentions
    self._absolute_positional_embed = absolute_positional_embed
    if USE_SYNC_BN:
      self._norm_layer = _get_norm_fn("sync_batch_norm")
    else:
      self._norm_layer = _get_norm_fn("batch_norm")

    # init and regularizer
    self._bias_regularizer = bias_regularizer
    self._init_args = dict(
      activation = activation,
      kernel_initializer = kernel_initializer,
      bias_initializer = kernel_initializer,
      kernel_regularizer = kernel_regularizer,
      bias_regularizer = bias_regularizer,
    )
  
  def build(self, input_shape):
    input_shape = input_shape[1:-1]
    embed_dims = self._embed_dimentions
    patch_size = self._patch_size

    # # TF corner pads by default so no need to do it manually
    self.project = nn_blocks.ConvBN(
      filters = embed_dims, 
      kernel_size = 7, 
      strides = 2, 
      padding = 'same',
      use_bn = True, 
      use_sync_bn = USE_SYNC_BN,
      use_bias = True, 
      **self._init_args
    )
    
    if self._patch_size == 4:
      self.sample = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2,2), padding = "same")
    

    if self._absolute_positional_embed:
      patch_resolution=(input_shape[0]//patch_size, input_shape[1]//patch_size)
      self.ape = self.add_weight(
        name = "absolute_positional_embed", 
        shape = [1, patch_resolution[0], patch_resolution[1], embed_dims],
        initializer = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.02), 
        regularizer = self._bias_regularizer, 
        trainable = True)

  
  def call(self, x):
    x = self.project(x) # wh/2
    if self._patch_size == 4:
      x = self.sample(x) # wh/2

    if self._absolute_positional_embed:
      absolute_positional_embed = self.ape
      x = x + absolute_positional_embed
    return x

# class PatchEmbed(tf.keras.layers.Layer):

#   def __init__(self,
#                patch_size = 4, 
#                embed_dimentions = 96, 
#                norm_layer = None, 
#                activation = None,
#                absolute_positional_embed = False, 
#                kernel_initializer='VarianceScaling',
#                kernel_regularizer=None,
#                bias_initializer='zeros',
#                bias_regularizer=None, 
#                **kwargs):
#     super().__init__(**kwargs)

#     self._patch_size = patch_size
#     self._embed_dimentions = embed_dimentions
#     self._absolute_positional_embed = absolute_positional_embed
#     # self._norm_layer = _get_norm_fn(norm_layer)
#     self._norm_layer = _get_norm_fn("batch_norm")

#     # init and regularizer
#     self._bias_regularizer = bias_regularizer
#     self._init_args = dict(
#       activation = activation,
#       kernel_initializer = kernel_initializer,
#       bias_initializer = kernel_initializer,
#       kernel_regularizer = kernel_regularizer,
#       bias_regularizer = bias_regularizer,
#     )
  
#   def build(self, input_shape):
#     input_shape = input_shape[1:-1]
#     embed_dims = self._embed_dimentions
#     patch_size = self._patch_size

#     # TF corner pads by default so no need to do it manually
#     self.project = tf.keras.layers.Conv2D(
#       filters = embed_dims, 
#       kernel_size = patch_size, 
#       strides = patch_size, 
#       padding = 'same',
#       **self._init_args)

#     self.norm = None
#     if self._norm_layer is not None:
#       self.norm = self._norm_layer()

#     if self._absolute_positional_embed:
#       patch_resolution=(input_shape[0]//patch_size, input_shape[1]//patch_size)
#       self.ape = self.add_weight(
#         name = "absolute_positional_embed", 
#         shape = [1, patch_resolution[0], patch_resolution[1], embed_dims],
#         initializer = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.02), 
#         regularizer = self._bias_regularizer, 
#         trainable = True)

  
#   def call(self, x):
#     x = self.project(x)
#     if self.norm is not None:
#       _, H, W, C = x.shape
#       x = tf.reshape(x, [-1, H * W, C])
#       x = self.norm(x)
#       x = tf.reshape(x, [-1, H, W, C])

#     if self._absolute_positional_embed:
#       # B, W, H, C = embeddings.shape
#       # interpolate positonal embeddings for different shapes, cannot do in TF
#       absolute_positional_embed = self.ape
#       x = x + absolute_positional_embed
#     return x

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
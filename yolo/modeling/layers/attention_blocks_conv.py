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
import functools
from typing import Callable, List, Tuple, Union
import tensorflow as tf
import numpy as np
from tensorflow.python.ops.control_flow_ops import _summarize_eager
from official.modeling import activations, tf_utils
from yolo.modeling.layers import nn_blocks
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
                 momentum = 0.99, 
                 epsilon = 0.001, 
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
  elif norm_layer == 'layer_norm':
    fn = tf.keras.layers.LayerNormalization
    return partial(fn, axis = axis, epsilon = epsilon)
  elif norm_layer == 'sync_batch_norm':
    fn = tf.keras.layers.experimental.SyncBatchNormalization
    return partial(fn, **kwargs)
  fn = tf.keras.layers.BatchNormalization
  return partial(fn, **kwargs)

def window_partition(x, window_size):
  """
  Args:
    x: (B, H, W, C)
    window_size (int): window size

  Returns:
    windows: (num_windows*B, window_size, window_size, C)
  """
  _, H, W, C = x.shape
  x = tf.reshape(x, [-1, H // window_size, window_size, W // window_size, window_size, C])
  x = tf.transpose(x, perm=(0, 1, 3, 2, 4, 5))
  x = tf.reshape(x, [-1, window_size, window_size, C])
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
  x = tf.reshape(x, [-1, H // window_size, window_size, W // window_size, window_size, C])
  x = tf.transpose(x, perm=(0, 1, 3, 2, 4, 5))
  x = tf.reshape(x, [-1, H, W, C])
  return x


class MLP(tf.keras.layers.Layer):

  def __init__(self, 
               hidden_features = None, 
               out_features = None, 
               kernel_initializer='VarianceScaling',
               kernel_regularizer=None,
               bias_initializer='zeros',
               bias_regularizer=None,
               activation = "gelu", 
               leaky_alpha=0.1,
               dropout = 0.0, 
               **kwargs):
    super().__init__(**kwargs)
    # features 
    self._hidden_features = hidden_features
    self._out_features = out_features

    # init and regularizer
    self._init_args = dict(
      kernel_initializer = kernel_initializer,
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

    self.fc1 = tf.keras.layers.Dense(hidden_features, **self._init_args)
    self.fc2 = tf.keras.layers.Dense(out_features, **self._init_args)
    self.drop = tf.keras.layers.Dropout(self._dropout)
    self.act = _get_activation_fn(self._activation, leaky_alpha=self._leaky)
    return 

  def call(self, x):
    x = self.fc1(x)
    x = self.act(x)
    x = self.drop(x)
    x = self.fc2(x)
    x = self.drop(x)
    return x

class FFN(tf.keras.layers.Layer):

  def __init__(self, 
               hidden_features = None, 
               out_features = None, 
               kernel_initializer='VarianceScaling',
               kernel_regularizer=None,
               bias_initializer='zeros',
               bias_regularizer=None,
               activation = "gelu", 
               leaky_alpha=0.1,
               dropout = 0.0, 
               **kwargs):
    super().__init__(**kwargs)
    # features 
    self._hidden_features = hidden_features
    self._out_features = out_features

    # init and regularizer
    self._init_args = dict(
      kernel_initializer = kernel_initializer,
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

    self.conv1 = nn_blocks.ConvBN(
      filters = hidden_features, 
      kernel_size = (3 , 3),
      **self._init_args
    )
    self.conv2 = nn_blocks.ConvBN(
      filters = out_features, 
      kernel_size = (1 , 1),
      **self._init_args
    )

    self.act = _get_activation_fn(self._activation, leaky_alpha=self._leaky)
    return 

  def call(self, x):
    x = self.conv1(x)
    x = self.act(x)
    x = self.conv2(x)
    return x

class FFN2(tf.keras.layers.Layer):

  def __init__(self, 
               hidden_features = None, 
               out_features = None, 
               kernel_initializer='VarianceScaling',
               kernel_regularizer=None,
               bias_initializer='zeros',
               bias_regularizer=None,
               activation = "gelu", 
               leaky_alpha=0.1,
               dropout = 0.0, 
               **kwargs):
    super().__init__(**kwargs)
    # features 
    self._hidden_features = hidden_features
    self._out_features = out_features

    # init and regularizer
    self._init_args = dict(
      kernel_initializer = kernel_initializer,
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

    self.fc1 = tf.keras.layers.Dense(hidden_features, **self._init_args)
    self.fc2 = tf.keras.layers.Dense(out_features, **self._init_args)
    self.conv2 = nn_blocks.ConvBN(
      filters = out_features, 
      kernel_size = (3 , 3),
      **self._init_args
    )

    self.drop = tf.keras.layers.Dropout(self._dropout)
    self.act = _get_activation_fn(self._activation, leaky_alpha=self._leaky)
    return 

  def call(self, x):
    B, H, W, C = x.shape

    x = tf.reshape(x, [-1, H * W, C])
    x = self.fc1(x)
    x = self.act(x)
    x = self.drop(x)
    x = self.fc2(x)
    x = self.act(x)
    x = self.drop(x)

    B, _, C = x.shape
    x = tf.reshape(x, [-1, H, W, C])
    x = self.conv2(x)
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
               attention_dropout = 0.0, 
               projection_dropout = 0.0, 
               kernel_initializer='VarianceScaling',
               kernel_regularizer=None,
               bias_initializer='zeros',
               bias_regularizer=None, 
               relative_bias_initializer='TrucatedNormal', 
               attention_activation = 'softmax', # typically jsut soft max, more for future developments of something better 
               **kwargs):
    super().__init__(**kwargs)

    if isinstance(window_size, int):
      window_size = (window_size, window_size)

    self._window_size = window_size # wH, wW
    self._num_heads = num_heads
    self._qkv_bias = qkv_bias
    self._qk_scale = qk_scale 

    # dropout
    self._attention_dropout = attention_dropout
    self._projection_dropout = projection_dropout

    # activation 
    self._attention_activation = attention_activation

    # init and regularizer
    self._relative_bias_initializer = relative_bias_initializer
    if relative_bias_initializer == "TrucatedNormal":
      self._relative_bias_initializer = tf.keras.initializers.TruncatedNormal(
          mean=0.0, stddev=0.02) 
    self._relative_bias_regularizer = bias_regularizer

    self._init_args = dict(
      kernel_initializer = kernel_initializer,
      bias_initializer = bias_initializer,
      kernel_regularizer = kernel_regularizer,
      bias_regularizer = bias_regularizer,
    )
    return
  
  def build(self, input_shape):
    self.dim = input_shape[-1]
    head_dims = self.dim/self._num_heads
    self.scale = self._qk_scale or head_dims ** -5

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
    coords_matrix = np.meshgrid(coords_h, coords_w, indexing='ij') # 2, Wh, Ww
    coords = np.stack(coords_matrix)
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
    self.proj = tf.keras.layers.Dense(self.dim, **self._init_args)
    self.proj_drop = tf.keras.layers.Dropout(self._projection_dropout)
    self.act = _get_activation_fn(self._attention_activation) # softmax
    return 
  
  def call(self, x, mask = None):
    _, N, C = x.shape

    qkv = self.qkv(x)
    qkv = tf.reshape(qkv, [-1, N, 3, self._num_heads, C // self._num_heads])
    qkv = tf.transpose(qkv, perm=(2, 0, 3, 1, 4))
    q, k, v = qkv[0], qkv[1], qkv[2]
    
    # compute the matrix mul attention
    q = q * self.scale
    attn = tf.matmul(q, k, transpose_b = True)

    # compute the relative poisiton bias
    num_elems = self._window_size[0] * self._window_size[1]
    indexes = tf.reshape(self._relative_positional_indexes, [-1])
    relative_position_bias = tf.gather(self._realtive_positional_bias_table, indexes)
    relative_position_bias = tf.reshape(relative_position_bias, [num_elems, num_elems, -1])
    relative_position_bias = tf.transpose(relative_position_bias, perm=(2, 0, 1))
    attn = attn + tf.expand_dims(relative_position_bias, axis = 0)

    if mask is not None:
      num_windows = mask.shape[0]
      mask = tf.cast(tf.expand_dims(tf.expand_dims(mask, axis = 1), axis = 0), attn.dtype)
      attn = tf.reshape(attn, [-1, num_windows, self._num_heads, N, N])
      attn = tf.reshape(attn + mask, [-1, self._num_heads, N, N])
      attn = self.act(attn)
    else:
      attn = self.act(attn)

    attn = self.attn_drop(attn)

    x = tf.matmul(attn, v)
    x = tf.transpose(x, perm = (0, 2, 1, 3)) # move heads to be merged with features
    x = tf.reshape(x, [-1, N, C])

    x = self.proj(x)
    x = self.proj_drop(x)
    return x
    
class SwinTransformerLayer(tf.keras.layers.Layer):

  def __init__(self, 
               num_heads, 
               window_size = 7, 
               shift_size = 0, 
               mlp_ratio = 4, 
               qkv_bias = True, 
               qk_scale = None, 
               dropout = 0.0, 
               attention_dropout = 0.0, 
               drop_path = 0.0, 
               activation = 'gelu',
               norm_layer = 'batch_norm',
               kernel_initializer='VarianceScaling',
               kernel_regularizer=None,
               bias_initializer='zeros',
               bias_regularizer=None, 
               **kwargs):
    super().__init__(**kwargs)

    self._num_heads = num_heads
    self._window_size = window_size
    self._shift_size = shift_size
    
    self._mlp_ratio = mlp_ratio
    self._qkv_bias = qkv_bias
    self._qk_scale = qk_scale
    
    self._dropout = dropout 
    self._attention_dropout = attention_dropout
    self._drop_path = drop_path

    self._activation = activation
    self._norm_layer = _get_norm_fn(norm_layer)

    # init and regularizer
    self._init_args = dict(
      kernel_initializer = kernel_initializer,
      bias_initializer = bias_initializer,
      kernel_regularizer = kernel_regularizer,
      bias_regularizer = bias_regularizer,
    )
    
  def build(self, input_shape):
    self._dims = input_shape[-1]

    if min(input_shape[1:-1]) <= self._window_size:
      # we cannot shift and the eindow size must be the input resolution 
      self._shift_size = 0
      self._window_size = min(input_shape[1:-1])
    
    if not (0 <= self._shift_size < self._window_size):
      raise ValueError("the shift must be contained between 0 and wndow_size "
                        "or the image will be shifted outside of the partition " 
                        "window. ")
    
    self.attention_layer = WindowedMultiHeadAttention(
        window_size=self._window_size, 
        num_heads=self._num_heads, 
        qkv_bias=self._qkv_bias, 
        qk_scale=self._qk_scale, 
        attention_dropout=self._attention_dropout, 
        projection_dropout=self._dropout, 
        **self._init_args)

    if self._drop_path > 0.0:
      self.drop_path = nn_layers.StochasticDepth(self._drop_path)
    else:
      self.drop_path = nn_blocks.Identity()

    self.act = _get_activation_fn(self._activation)
    
    self.norm1 = self._norm_layer()
    mlp_hidden_dim = max(int(self._dims * self._mlp_ratio), 1)
    self.ffn = FFN(hidden_features=mlp_hidden_dim, activation=self._activation, 
                   dropout=self._dropout, **self._init_args)
  
  def call(self, x, mask_matrix = None, training = None):
    B, H, W, C = x.shape
    
    # save normalsize and reshape
    shortcut = x

    # pad feature maps to multiples of window size
    pad_l = pad_t = 0 
    pad_r = (self._window_size - W % self._window_size) % self._window_size
    pad_b = (self._window_size - H % self._window_size) % self._window_size  
    x = tf.pad(x, [[0,0], [pad_t, pad_b], [pad_l, pad_r], [0, 0]]) 
    _, Hp, Wp, _ = x.shape

    # cyclic shift
    if self._shift_size > 0:
      shifted_x = tf.roll(x, shift=[-self._shift_size, -self._shift_size], axis = [1, 2])
      attn_mask = mask_matrix      
    else:
      shifted_x = x
      attn_mask = None 
    
    # partition windows
    x_windows = window_partition(shifted_x, self._window_size)
    x_windows = tf.reshape(x_windows, [-1, self._window_size * self._window_size, C])

    attn_windows = self.attention_layer(x_windows, mask = attn_mask)

    attn_windows = tf.reshape(attn_windows, [-1, self._window_size, self._window_size, C])
    shifted_x = window_reverse(attn_windows, self._window_size, Hp, Wp)

    if self._shift_size > 0:
      x = tf.roll(shifted_x, shift=[self._shift_size, self._shift_size], axis = [1, 2])
    else:
      x = shifted_x

    if pad_r > 0 or pad_b > 0:
      x = x[:, :H, :W, :]

    # Feed Forward Network
    x = shortcut + self.drop_path(x)
    
    x = x + self.drop_path(self.ffn(self.norm1(x)))
    x = self.act(x)
    return x

class PatchMerge(tf.keras.layers.Layer):

  def __init__(self,
               norm_layer = 'layer_norm',
               kernel_initializer='VarianceScaling',
               kernel_regularizer=None,
               bias_initializer='zeros',
               bias_regularizer=None, 
               **kwargs):
    super().__init__(**kwargs)

    #default
    self._norm_layer = _get_norm_fn(norm_layer)

    # init and regularizer
    self._init_args = dict(
      use_bias = False,
      kernel_initializer = kernel_initializer,
      bias_initializer = bias_initializer,
      kernel_regularizer = kernel_regularizer,
      bias_regularizer = bias_regularizer,
    )
  
  def build(self, input_shape):
    self._dims = input_shape[-1]
    self.reduce = tf.keras.layers.Dense(self._dims * 2, **self._init_args)
    self.norm = self._norm_layer()
  
  def call(self, x):
    """Down sample by 2. """
    B, H, W, C = x.shape

    # x = tf.reshape(x, [-1, H, W, C])

    # padding 
    pad_input = (H % 2 == 1) or (W % 2 == 1)
    if pad_input:
      x = tf.pad(x, [[0,0], [0, H%2], [0, W%2], [0, 0]])

    x0 = x[:, 0::2, 0::2, :] # B H/2 W/2 C
    x1 = x[:, 1::2, 0::2, :] # B H/2 W/2 C
    x2 = x[:, 0::2, 1::2, :] # B H/2 W/2 C
    x3 = x[:, 1::2, 1::2, :] # B H/2 W/2 C

    x = tf.concat([x0, x1, x2, x3], axis = -1)
    x = tf.reshape(x, [-1, (H//2) * (W//2), C * 4])

    x = self.norm(x)
    x = self.reduce(x)
    
    x = tf.reshape(x, [-1, H//2, W//2, C * 2])
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
               norm_layer = 'batch_norm', 
               downsample = 'patch_and_merge',
               activation = 'gelu',
               kernel_initializer='VarianceScaling',
               kernel_regularizer=None,
               bias_initializer='zeros',
               bias_regularizer=None, 
               **kwargs):
    super().__init__(**kwargs)

    self._depth = depth
    self._num_heads = num_heads
    
    self._window_size = window_size
    self._shift_size = window_size // 2

    self._norm_layer = _get_norm_fn(norm_layer)
    self._downsample = downsample

    # init and regularizer
    self._swin_args = dict(
       window_size=window_size,
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
      shift_size = 0 if i % 2 == 0 else self._shift_size
      drop_path = self._drop_path[i] if index_drop_path else self._drop_path
      layer = SwinTransformerLayer(self._num_heads, 
                                   shift_size=shift_size, 
                                   drop_path=drop_path, 
                                   **self._swin_args)
      self.layers.append(layer)
    
    if self._downsample == "patch_and_merge":
      self.downsample = PatchMerge(**self._init_args)
    else:
      self.downsample = None
    
    # clarity on what this is and why is needed, is it 
    # just masking the locations to consider post shift so pixels rolled "out"
    # of the image get set to zero? we need to visualize this somehow.
    H, W = input_shape[1:-1]
    self.Hp = int(np.ceil(H / self._window_size) * self._window_size)
    self.Wp = int(np.ceil(W / self._window_size) * self._window_size)
    
    h_slices = (
      slice(0, -self._window_size), 
      slice(-self._window_size, -self._shift_size), 
      slice(-self._shift_size, None)
    )
    w_slices = (
      slice(0, -self._window_size), 
      slice(-self._window_size, -self._shift_size), 
      slice(-self._shift_size, None)
    )

    img_mask = np.zeros((1, self.Hp, self.Wp, 1))

    cnt = 0 
    for h in h_slices: 
      for w in w_slices:
        img_mask[:, h, w, :] = cnt 
        cnt += 1

    img_mask = tf.convert_to_tensor(img_mask)
    self.img_mask = img_mask
    # tf.Variable(
    #   initial_value=img_mask, 
    #   trainable=False, 
    #   name='{}_img_mask'.format(self.name), )
    return 

  def _build_mask(self, x_shape, dtype = 'float32'):
    B, H, W, C = x_shape

    # can we do a repeat pad here??
    img_mask = tf.cast(self.img_mask, dtype)
    mask_windows = window_partition(img_mask, self._window_size)
    mask_windows = tf.reshape(mask_windows, [-1, self._window_size*self._window_size])

    attn_mask = tf.expand_dims(mask_windows, axis = 1) - tf.expand_dims(mask_windows, axis = 2)
    attn_mask = tf.where(attn_mask == 0, tf.cast(0.0, dtype), attn_mask)
    attn_mask = tf.where(attn_mask != 0, tf.cast(-100.0, dtype), attn_mask)
    return attn_mask

  def call(self, x):
    attn_mask = self._build_mask(x.shape, dtype = x.dtype)

    for layer in self.layers:
      x = layer(x, mask_matrix = attn_mask)

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
    self._norm_layer = _get_norm_fn(norm_layer)

    # init and regularizer
    self._bias_regularizer = bias_regularizer
    self._init_args = dict(
      use_bn = False,
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

    # TF corner pads by default so no need to do it manually
    self.project = nn_blocks.ConvBN(
      filters =embed_dims, kernel_size = patch_size, 
      strides = patch_size, padding = 'same',
      **self._init_args)

    self.norm = None
    if self._norm_layer is not None:
      self.norm = self._norm_layer()

    if self._absolute_positional_embed:
      patch_resolution=(input_shape[0]//patch_size, input_shape[1]//patch_size)
      self.ape = self.add_weight(
        name = "absolute_positional_embed", 
        shape = [1, patch_resolution[0], patch_resolution[1], embed_dims],
        initializer = tf.keras.initializers.TruncatedNormal(
          mean=0.0, stddev=0.02), 
        regularizer = self._bias_regularizer, 
        trainable = True)

  
  def call(self, x):
    x = self.project(x)

    # if self.norm is not None:
    #   _, H, W, C = x.shape
    #   x = tf.reshape(x, [-1, H * W, C])
    #   x = self.norm(x)
    #   x = tf.reshape(x, [-1, H, W, C])

    if self._absolute_positional_embed:
      # B, W, H, C = embeddings.shape
      # interpolate positonal embeddings for different shapes, cannot do in TF
      absolute_positional_embed = self.ape
      x = x + absolute_positional_embed
    
    if self.norm is not None:
      x = self.norm(x)

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
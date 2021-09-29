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
from typing import Callable, List, Tuple
import tensorflow as tf
import numpy as np
from official.modeling import activations, tf_utils
from yolo.modeling.layers import nn_blocks
from official.vision.beta.modeling.layers import nn_layers

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
    self._kernel_initializer = kernel_initializer
    self._bias_initializer = bias_initializer
    self._kernel_regularizer = kernel_regularizer
    self._bias_regularizer = bias_regularizer

    # activation
    self._activation = activation
    self._leaky_alpha = leaky_alpha
    self._dropout = dropout

  def build(self, input_shape):
    hidden_features = self._hidden_features if self._hidden_features is not None else input_shape[-1]
    out_features = self._out_features if self._out_features is not None else input_shape[-1]

    self.fc1 = tf.keras.layers.Dense(hidden_features, 
                                     kernel_initializer = self._kernel_initializer, 
                                     kernel_regularizer = self._kernel_regularizer, 
                                     bias_initializer = self._bias_initializer, 
                                     bias_regularizer = self._bias_regularizer)

    if self._activation == 'leaky':
      self._activation_fn = tf.keras.layers.LeakyReLU(alpha=self._leaky_alpha)
    elif self._activation == 'mish':
      self._activation_fn = lambda x: x * tf.math.tanh(tf.math.softplus(x))
    else:
      self._activation_fn = tf_utils.get_activation(self._activation)
    
    self.drop = tf.keras.layers.Dropout(self._dropout)
    self.fc2 = tf.keras.layers.Dense(out_features, 
                                     kernel_initializer = self._kernel_initializer, 
                                     kernel_regularizer = self._kernel_regularizer, 
                                     bias_initializer = self._bias_initializer, 
                                     bias_regularizer = self._bias_regularizer)
    return 

  def call(self, x):
    x = self.fc1(x)
    x = self._activation_fn(x)
    x = self.drop(x)

    x = self.fc2(x)
    x = self.drop(x)
    return x

class WindowPartition(tf.keras.layers.Layer):

  def __init__(self, window_size, **kwargs):
    super().__init__(**kwargs)
    self._window_size = window_size
  
  def build(self, input_shape):
    self._input_shape = input_shape

  def call(self, x):
    _, H, W, C = self._input_shape
    window_size = self._window_size

    x = tf.reshape(x, [-1, H // window_size, window_size, 
                           W // window_size, window_size, C])
    x = tf.transpose(x, perm=(0, 1, 3, 2, 4, 5))
    x = tf.reshape(x, [-1, window_size, window_size, C])
    return x

class WindowReverse(tf.keras.layers.Layer):

  def __init__(self, height, width, **kwargs):
    super().__init__(**kwargs)
    self._height = height
    self._width = width 

  def build(self, input_shape):
    self._input_shape = input_shape

  def call(self, x):
    H = self._height
    W = self._width
    _, window_size, _, C = self._input_shape 

    x = tf.reshape(x, [-1, H // window_size, W // window_size, window_size, window_size, C])
    x = tf.transpose(x, perm=(0, 1, 3, 2, 4, 5))
    x = tf.reshape(x, [-1, H, W, C])
    return x

class WindowedMultiHeadAttention(tf.keras.layers.Layer):

  """Windowd multi head attention with a location based relative bias that is learned. 
  
  Functionally equivalent to multi head self attention but adds a gather for the 
  positional bias. 
  """

  def __init__(self, 
               window_size, 
               num_heads, 
               bias_qkv = True, 
               qk_scale = None, 
               attention_dropout = 0.0, 
               projection_dropout = 0.0, 
               kernel_initializer='VarianceScaling',
               kernel_regularizer=None,
               bias_initializer='zeros',
               bias_regularizer=None, 
               attention_activation = 'softmax', # typically jsut soft max, more for future developments of something better 
               **kwargs):
    super().__init__(**kwargs)

    if isinstance(window_size, int):
      window_size = (window_size, window_size)

    self._window_size = window_size # wH, wW
    self._num_heads = num_heads
    self._bias_qkv = bias_qkv
    self._qk_scale = qk_scale 

    # dropout
    self._attention_dropout = attention_dropout
    self._projection_dropout = projection_dropout

    # activation 
    self._attention_activation = attention_activation

    # init and regularizer
    self._kernel_initializer = kernel_initializer
    self._bias_initializer = bias_initializer
    self._kernel_regularizer = kernel_regularizer
    self._bias_regularizer = bias_regularizer
    return
  
  def build(self, input_shape):
    self._input_shape = input_shape
    input_dims = input_shape[-1]
    head_dims = input_dims/self._num_heads

    self.scale = self._qk_scale or head_dims ** -5
    window_size = self._window_size

    # biases to apply to each "position" learned 
    num_elements = (2 * window_size[0] - 1) * (2 * window_size[1] - 1)
    self._realtive_positional_bias_table = self.add_weight(
      name = '{}_realtive_bias_table'.format(self.name), 
      shape = [num_elements, self._num_heads], 
      initializer = self._bias_initializer, 
      regularizer = self._bias_regularizer, 
      trainable = True)

    # get the postions to associate the above bais table with    
    coords_h = np.arange(window_size[0])
    coords_w = np.arange(window_size[1])
    coords_matrix = np.meshgrid(coords_h, coords_w, indexing='ij')
    coords = np.stack(coords_matrix)
    coords_flatten = coords.reshape(2, -1)
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
    relative_coords = relative_coords.transpose([1, 2, 0])
    relative_coords[:, :, 0] += window_size[0] - 1
    relative_coords[:, :, 1] += window_size[1] - 1
    relative_coords[:, :, 0] *= 2 * window_size[1] - 1
    relative_positional_indexes = relative_coords.sum(-1)

    self._relative_positional_indexes = tf.Variable(
      initial_value=tf.convert_to_tensor(relative_positional_indexes), 
      trainable=False, 
      name='{}_realtive_indexes'.format(self.name))

    self.qkv = tf.keras.layers.Dense(input_dims * 3, 
                                use_bias = self._bias_qkv, 
                                kernel_initializer = self._kernel_initializer, 
                                kernel_regularizer = self._kernel_regularizer, 
                                bias_initializer = self._bias_initializer, 
                                bias_regularizer = self._bias_regularizer)
    self.attention_drop = tf.keras.layers.Dropout(self._attention_dropout)

    self.projection = tf.keras.layers.Dense(input_dims, 
                                kernel_initializer = self._kernel_initializer, 
                                kernel_regularizer = self._kernel_regularizer, 
                                bias_initializer = self._bias_initializer, 
                                bias_regularizer = self._bias_regularizer)
    self.projection_drop = tf.keras.layers.Dropout(self._projection_dropout)

    if self._attention_activation == 'leaky':
      self._activation_fn = tf.keras.layers.LeakyReLU(alpha=self._leaky_alpha)
    elif self._attention_activation == 'mish':
      self._activation_fn = lambda x: x * tf.math.tanh(tf.math.softplus(x))
    else:
      self._activation_fn = tf_utils.get_activation(self._attention_activation)
    return 
  
  def call(self, x, mask = None):
    # shape = tf.shape(x)
    _, N, C = self._input_shape
    window_size = self._window_size

    qkv = self.qkv(x)

    # qkv = tf.reshape(qkv, [B, N, 3, self._num_heads, C // self._num_heads])
    qkv = tf.reshape(qkv, [-1, N, 3, self._num_heads, C // self._num_heads])
    qkv = tf.transpose(qkv, perm=(2, 0, 3, 1, 4))
    q, k, v = qkv[0], qkv[1], qkv[2]
    
    # compute the matrix mul attention
    q = q * self.scale
    attention = tf.matmul(q, k, transpose_b = True)

    # compute the relative poisiton bias
    num_elems = window_size[0] * window_size[1]
    indexes = tf.reshape(self._relative_positional_indexes, [-1, ])
    positional_bias = tf.gather(self._realtive_positional_bias_table, indexes)
    positional_bias = tf.reshape(positional_bias, [num_elems, num_elems, -1])
    positional_bias = tf.transpose(positional_bias, perm=(2, 0, 1))
    attention = attention + tf.expand_dims(positional_bias, axis = 0)

    if mask is not None:
      mask_shape = tf.shape(mask)
      num_windows = mask_shape[0]
      mask = tf.cast(mask, attention.dtype)
      mask = tf.expand_dims(tf.expand_dims(mask, axis = 1), axis = 0)

      attention = tf.reshape(attention, [-1, num_windows, self._num_heads, N, N])
      attention = attention + mask
      attention = tf.reshape(attention, [-1, self._num_heads, N, N])
      attention = self._activation_fn(attention)
    else:
      attention = self._activation_fn(attention)

    attention = self.attention_drop(attention)

    x = tf.matmul(attention, v)
    x = tf.transpose(x, perm = (0, 2, 1, 3)) # move heads to be merged with features
    x = tf.reshape(x, [-1, N, C])

    x = self.projection(x)
    x = self.projection_drop(x)
    return x, attention
    
class SwinTransformerLayer(tf.keras.layers.Layer):

  def __init__(self, 
               num_heads, 
               window_size = 7, 
               shift_size = 0, 
               mlp_ratio = 4, 
               use_bias_qkv = True, 
               qk_scale = None, 
               drop = 0.0, 
               attention_drop = 0.0, 
               drop_path = 0.0, 
               activation = 'gelu',
               kernel_initializer='VarianceScaling',
               kernel_regularizer=None,
               bias_initializer='zeros',
               bias_regularizer=None, 
               **kwargs):
    super().__init__(**kwargs)

    self._num_heads = num_heads
    self._shift_size = shift_size
    
    self._window_size = window_size
    self._mlp_ratio = mlp_ratio
    self._use_bias_qkv = use_bias_qkv
    self._qk_scale = qk_scale
    
    self._drop = drop 
    self._attention_drop = attention_drop
    self._drop_path = drop_path

    self._activation = activation
    self._norm_fn = tf.keras.layers.LayerNormalization

    # init and regularizer
    self._kernel_initializer = kernel_initializer
    self._bias_initializer = bias_initializer
    self._kernel_regularizer = kernel_regularizer
    self._bias_regularizer = bias_regularizer
    
  def build(self, input_shape):
    self._input_resolution = input_shape[1:-1]
    self._dims = input_shape[-1]

    if min(self._input_resolution) <= self._window_size:
      # we cannot shift and the eindow size must be the input resolution 
      self._shift_size = 0
      self._window_size = min(self._input_resolution)
    
    if not (0 <= self._shift_size < self._window_size):
      raise ValueError("the shift must be contained between 0 and wndow_size "
                        "or the image will be shifted outside of the partition " 
                        "window. ")
    
    self.norm1 = self._norm_fn()

    # shift
    self.partition = WindowPartition(self._window_size)
    self.attention_layer = WindowedMultiHeadAttention(
        window_size=self._window_size, 
        num_heads=self._num_heads, 
        bias_qkv=self._use_bias_qkv, 
        qk_scale=self._qk_scale, 
        attention_dropout=self._attention_drop, 
        projection_dropout=self._drop)
    self.undo_partition = WindowReverse(
        self._input_resolution[0], self._input_resolution[1])
    # unshift

    if self._drop_path > 0.0:
      self.drop_path = nn_layers.StochasticDepth(self._drop_path)
    else:
      self.drop_path = nn_blocks.Identity()
    
    self.norm2 = self._norm_fn()
    self.mlp = MLP(hidden_features=int(self._dims * self._mlp_ratio), 
                   activation=self._activation, 
                   dropout=self._drop, 
                   kernel_initializer = self._kernel_initializer,
                   bias_initializer = self._bias_initializer,
                   kernel_regularizer = self._kernel_regularizer,
                   bias_regularizer = self._bias_regularizer)

    if self._shift_size > 0:
      # clarity on what this is and why is needed, is it 
      # just masking the locations to consider post shift so pixels rolled "out"
      # of the image get set to zero? we need to visualize this somehow.
      H, W = self._input_resolution
      

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

      img_mask = np.zeros((1, H, W, 1))

      cnt = 0 
      for h in h_slices: 
        for w in w_slices:
          img_mask[:, h, w, :] = cnt 
          cnt += 1

      img_mask = tf.convert_to_tensor(img_mask)
      mask_windows = WindowPartition(self._window_size)(img_mask)
      mask_windows = tf.reshape(mask_windows, 
                                  [-1, self._window_size * self._window_size])
      attention_mask = (tf.expand_dims(mask_windows, axis = 1) 
                                - tf.expand_dims(mask_windows, axis = 2))

      dtype = tf.keras.backend.floatx()
      attention_mask = tf.cast(attention_mask, dtype)
      attention_mask = tf.where(
          attention_mask != 0.0, float(-100.0), attention_mask)
      attention_mask = tf.where(
          attention_mask == 0.0, float(0.0), attention_mask)
      attention_mask = tf.Variable(
          initial_value=attention_mask, trainable=False, 
          name='{}_attn_mask'.format(self.name))
    else:
      attention_mask = None
    
    self.attention_mask = attention_mask
  
  def call(self, x, training = None):
    H, W = self._input_resolution
    C = self._dims

    x = tf.reshape(x, [-1, H * W, C])
    
    shortcut = x
    x = self.norm1(x)
    x = tf.reshape(x, [-1, H, W, C])

    if self._shift_size > 0:
      shifted_x = tf.roll(
          x, shift=[-self._shift_size, -self._shift_size], axis = [1, 2])
    else:
      shifted_x = x 
    
    x_windows = self.partition(shifted_x)
    x_windows = tf.reshape(
        x_windows, [-1, self._window_size * self._window_size, C])

    attention_windows, _ = self.attention_layer(
        x_windows, mask = self.attention_mask)

    attention_windows = tf.reshape(
        attention_windows, [-1, self._window_size, self._window_size, C])
    shifted_x = self.undo_partition(attention_windows)

    if self._shift_size > 0:
      x = tf.roll(
          shifted_x, shift=[self._shift_size, self._shift_size], axis = [1, 2])
    else:
      x = shifted_x

    # Feed Forward Network
    x = tf.reshape(x, [-1, H * W, C])
    x = shortcut + self.drop_path(x)
    x = x + self.drop_path(self.mlp(self.norm2(x)))

    x = tf.reshape(x, [-1, H, W, C])
    return x

class PatchMerge(tf.keras.layers.Layer):

  def __init__(self,
               kernel_initializer='VarianceScaling',
               kernel_regularizer=None,
               bias_initializer='zeros',
               bias_regularizer=None, 
               **kwargs):
    super().__init__(**kwargs)

    #default
    self._norm_fn = tf.keras.layers.LayerNormalization

    # init and regularizer
    self._kernel_initializer = kernel_initializer
    self._bias_initializer = bias_initializer
    self._kernel_regularizer = kernel_regularizer
    self._bias_regularizer = bias_regularizer
  
  def build(self, input_shape):
    self._input_resolution = input_shape[1:-1]
    self._dims = input_shape[-1]

    self.reduce = tf.keras.layers.Dense(self._dims * 2, 
                                  kernel_initializer = self._kernel_initializer, 
                                  kernel_regularizer = self._kernel_regularizer, 
                                  use_bias = False, 
                                  bias_initializer = self._bias_initializer, 
                                  bias_regularizer = self._bias_regularizer)
    self.norm = self._norm_fn()
  
  def call(self, x):
    """Down sample by 2. """

    H, W = self._input_resolution
    C = self._dims

    # x = tf.reshape(x, [-1, H, W, C])

    x0 = x[:, 0::2, 0::2, :]
    x1 = x[:, 1::2, 0::2, :]
    x2 = x[:, 0::2, 1::2, :]
    x3 = x[:, 1::2, 1::2, :]

    x = tf.concat([x0, x1, x2, x3], axis = -1)
    x = tf.reshape(x, [-1, ((H//2) * (W//2)), C * 4])

    x = self.norm(x)
    x = self.reduce(x)
    
    x = tf.reshape(x, [-1, H//2, W//2, C * 2])
    return x

class PatchEmbed(tf.keras.layers.Layer):

  def __init__(self,
               patch_size = 4, 
               embed_dimentions = 96, 
               use_layer_norm = False, 
               kernel_initializer='VarianceScaling',
               kernel_regularizer=None,
               bias_initializer='zeros',
               bias_regularizer=None, 
               activation = None,
               drop = 0.0, 
               absolute_positional_embed = False, 
               **kwargs):
    super().__init__(**kwargs)

    self._patch_size = patch_size
    self._embed_dimentions = embed_dimentions
    self._activation = activation
    self._drop_out = drop
    self._absolute_positional_embed = absolute_positional_embed

    if use_layer_norm:
      #default
      self._norm_fn = tf.keras.layers.LayerNormalization
    else:
      self._norm_fn = None

    # init and regularizer
    self._kernel_initializer = kernel_initializer
    self._bias_initializer = bias_initializer
    self._kernel_regularizer = kernel_regularizer
    self._bias_regularizer = bias_regularizer
  
  def build(self, input_shape):
    self._input_resolution = input_shape[1:-1]
    self._dims = input_shape[-1]

    self._patch_resolution = (self._input_resolution[0]//self._patch_size,
                              self._input_resolution[1]//self._patch_size)

    self.project = nn_blocks.ConvBN(
      filters = self._embed_dimentions, 
      kernel_size = self._patch_size, 
      strides = self._patch_size, 
      kernel_initializer = self._kernel_initializer, 
      kernel_regularizer = self._kernel_regularizer, 
      bias_initializer = self._bias_initializer, 
      bias_regularizer = self._bias_regularizer, 
      activation = self._activation,
      use_bn = False)
    if self._norm_fn is not None:
      self.norm = self._norm_fn()
    
    if self._drop_out > 0.0:
      self.drop = tf.keras.layers.Dropout(self._drop_out)

    if self._absolute_positional_embed:
      self.ape = self.add_weight(
        name = "absolute_positional_embed", 
        shape = [1, 
                 self._patch_resolution[0] * self._patch_resolution[1], 
                 self._embed_dimentions],
        initializer = self._kernel_initializer, 
        regularizer = self._bias_regularizer, 
        trainable = True)
  
  def call(self, x):
    """Down sample by 2. """
    x = self.project(x)

    H, W = self._patch_resolution
    C = self._embed_dimentions

    x = tf.reshape(x, [-1, H * W, C])
    if self._norm_fn is not None:
      x = self.norm(x)
    if self._absolute_positional_embed:
      x = x + self.ape
    if self._drop_out > 0.0:
      x = self.drop(x)
    x = tf.reshape(x, [-1, H, W, C])
    return x

class PatchExtractEmbed(tf.keras.layers.Layer):

  def __init__(self,
               patch_size = 4, 
               embed_dimentions = 96, 
               use_layer_norm = False, 
               kernel_initializer='VarianceScaling',
               kernel_regularizer=None,
               bias_initializer='zeros',
               bias_regularizer=None, 
               activation = None,
               drop = 0.0, 
               absolute_positional_embed = False, 
               **kwargs):
    super().__init__(**kwargs)

    self._patch_size = patch_size
    self._embed_dimentions = embed_dimentions
    self._activation = activation
    self._drop_out = drop
    self._absolute_positional_embed = absolute_positional_embed

    # init and regularizer
    self._kernel_initializer = kernel_initializer
    self._bias_initializer = bias_initializer
    self._kernel_regularizer = kernel_regularizer
    self._bias_regularizer = bias_regularizer

  def build(self, input_shape):
    self._input_resolution = input_shape[1:-1]
    self._dims = input_shape[-1]

    self._patch_resolution = (self._input_resolution[0]//self._patch_size,
                              self._input_resolution[1]//self._patch_size)

    self._num_patches = self._patch_resolution[0] * self._patch_resolution[1]  
    
    self.project = tf.keras.layers.Dense(self._embed_dimentions)
    self.pos_embed = tf.keras.layers.Embedding(input_dim = self._num_patches, 
                                               output_dim = self._embed_dimentions)

  def call(self, x):
    patches = tf.image.extract_patches(x, 
        sizes = (1, self._patch_size, self._patch_size, 1),
        strides = (1, self._patch_size, self._patch_size, 1), 
        rates=(1, 1, 1, 1), padding='VALID')

    patch_dim = patches.shape[-1]
    patch_num = patches.shape[1]
    patches = tf.reshape(patches, [-1, patch_num * patch_num, patch_dim])
    
    pos = tf.range(0, self._num_patches, delta = 1)
    embed = self.project(patches) + self.pos_embed(pos)

    H, W = self._patch_resolution
    C = self._embed_dimentions
    embed = tf.reshape(embed, [-1, H, W, C])
    return embed


class SwinTransformerBlock(tf.keras.layers.Layer):

  def __init__(self,
               depth, 
               num_heads, 
               window_size, 
               mlp_ratio = 4, 
               use_bias_qkv = True, 
               qk_scale = None, 
               drop = 0.0, 
               attention_drop = 0.0, 
               drop_path = 0.0, 
               downsample_type = 'patch_and_merge', 
               kernel_initializer='VarianceScaling',
               kernel_regularizer=None,
               bias_initializer='zeros',
               bias_regularizer=None, 
               activation = 'gelu',
               **kwargs):
    super().__init__(**kwargs)

    self._depth = depth
    self._num_heads = num_heads
    
    self._window_size = window_size
    self._mlp_ratio = mlp_ratio
    self._use_bias_qkv = use_bias_qkv
    self._qk_scale = qk_scale
    
    self._drop = drop 
    self._attention_drop = attention_drop
    self._drop_path = drop_path

    self._activation = activation

    self._downsample_type = downsample_type

    # init and regularizer
    self._kernel_initializer = kernel_initializer
    self._bias_initializer = bias_initializer
    self._kernel_regularizer = kernel_regularizer
    self._bias_regularizer = bias_regularizer
  
  def build(self, input_shape):
    
    self.layers = []

    for i in range(self._depth):
      layer = SwinTransformerLayer(
        self._num_heads, 
        window_size=self._window_size, 
        shift_size=(0 if i % 2 == 0 else self._window_size//2),  
        mlp_ratio=self._mlp_ratio, 
        use_bias_qkv=self._use_bias_qkv, 
        qk_scale=self._qk_scale, 
        drop=self._drop, 
        attention_drop=self._attention_drop, 
        drop_path=(self._drop_path[i] if isinstance(self._drop_path, List) else self._drop_path),
        activation=self._activation, 
        kernel_initializer = self._kernel_initializer,
        kernel_regularizer = self._kernel_regularizer, 
        bias_initializer = self._bias_initializer, 
        bias_regularizer = self._bias_regularizer)
      self.layers.append(layer)
    
    if self._downsample_type == "patch_and_merge":
      self.downsample = PatchMerge(
        kernel_initializer = self._kernel_initializer,
        kernel_regularizer = self._kernel_regularizer, 
        bias_initializer = self._bias_initializer, 
        bias_regularizer = self._bias_regularizer)
    elif self._downsample_type == "conv":
      self.downsample = nn_blocks.ConvBN(
                  filters = input_shape[-1] * 2, 
                  kernel_size = 2, 
                  strides = 2, 
                  kernel_initializer = self._kernel_initializer, 
                  kernel_regularizer = self._kernel_regularizer, 
                  bias_initializer = self._bias_initializer, 
                  bias_regularizer = self._bias_regularizer, 
                  activation = self._activation,
                  use_bn=self._use_bn, 
                  use_sync_bn=self._use_sync_bn)
      if not self._use_bn and not self._use_sync_bn:
        norm = tf.keras.layers.LayerNormalization()
        self.downsample = tf.keras.Sequential(layers = [self.downsample, norm])
    else:
      self.downsample = None
    return 

  def call(self, x):
    for layer in self.layers:
      x = layer(x)
    
    if self.downsample is not None:
      x = self.downsample(x)
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

  # input1 = built["5"]
  # layer = SwinTransformerBlock(2, 8, window_size=7)
  # output = layer(input1)
  # print(output.shape)
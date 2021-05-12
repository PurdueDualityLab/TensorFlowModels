# Lint as: python3
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Feature Pyramid Network and Path Aggregation variants used in YOLO"""

import tensorflow as tf

from yolo.modeling.layers import nn_blocks


@tf.keras.utils.register_keras_serializable(package='yolo')
class Identity_dup(tf.keras.layers.Layer):
  
  def __init__(self, **kwargs):
    super().__init__(**kwargs)

  def call(self, input):
    return None, input

@tf.keras.utils.register_keras_serializable(package='yolo')
class YoloFPN(tf.keras.layers.Layer):
  """YOLO Feature pyramid network."""

  def __init__(self,
               fpn_path_len=4,
               embed_sam = False, 
               convert_csp = False, 
               activation='leaky',
               use_sync_bn=False,
               norm_momentum=0.99,
               norm_epsilon=0.001,
               kernel_initializer='glorot_uniform',
               kernel_regularizer=None,
               bias_regularizer=None,
               subdivisions=8,
               **kwargs):
    """
    Yolo FPN initialization function. Yolo V4
    Args:
      fpn_path_len: `int`, number of layers ot use in each FPN path
        if you choose to use an FPN.
      use_sync_bn: if True, use synchronized batch normalization.
      norm_momentum: `float`, normalization omentum for the moving average.
      norm_epsilon: `float`, small float added to variance to avoid dividing by
        zero.
      activation: `str`, the activation function to use typically leaky or mish.
      kernel_initializer: kernel_initializer for convolutional layers.
      kernel_regularizer: tf.keras.regularizers.Regularizer object for Conv2D.
      bias_regularizer: tf.keras.regularizers.Regularizer object for Conv2d.
      **kwargs: keyword arguments to be passed.
    """
    super().__init__(**kwargs)
    self._fpn_path_len = fpn_path_len

    self._activation = activation
    self._use_sync_bn = use_sync_bn
    self._norm_momentum = norm_momentum
    self._norm_epsilon = norm_epsilon
    self._kernel_initializer = kernel_initializer
    self._kernel_regularizer = kernel_regularizer
    self._bias_regularizer = bias_regularizer
    self._subdivisions = subdivisions
    self._embed_sam = embed_sam
    self._convert_csp = convert_csp

    self._base_config = dict(
        activation=self._activation,
        use_sync_bn=self._use_sync_bn,
        subdivisions=self._subdivisions,
        kernel_regularizer=self._kernel_regularizer,
        kernel_initializer=self._kernel_initializer,
        bias_regularizer=self._bias_regularizer,
        norm_epsilon=self._norm_epsilon,
        norm_momentum=self._norm_momentum)

  def get_raw_depths(self, minimum_depth):
    depths = []
    for _ in range(self._min_level, self._max_level + 1):
      depths.append(minimum_depth)
      minimum_depth *= 2
    return list(reversed(depths))

  def build(self, inputs):
    """
    use config dictionary to generate all important attributes for head
    construction

    Args:
       inputs: dictionary of the shape of input args as a dictionary of lists
    """
    keys = [int(key) for key in inputs.keys()]
    self._min_level = min(keys)
    self._max_level = max(keys)
    self._min_depth = inputs[str(self._min_level)][-1]
    self._depths = self.get_raw_depths(self._min_depth)

    # directly connect to an input path and process it
    self.preprocessors = dict()
    # resample an input and merge it with the output of another path
    # inorder to aggregate backbone outputs
    self.resamples = dict()
    # set of convoltion layers and upsample layers that are used to
    # prepare the FPN processors for output

    for level, depth in zip(
        reversed(range(self._min_level, self._max_level + 1)), self._depths):
      if level == self._min_level:
        self.resamples[str(level)] = nn_blocks.PathAggregationBlock(
            filters=depth // 2, 
            inverted=True, 
            upsample=True,
            drop_final=True, 
            upsample_size=2, **self._base_config)
        self.preprocessors[str(level)] = Identity_dup()
        # nn_blocks.DarkRouteProcess(
        #     filters=depth,
        #     repetitions=self._fpn_path_len - int(level == self._min_level),
        #     block_invert=True,
        #     insert_spp=False,
        #     **self._base_config)
      elif level != self._max_level:
        self.resamples[str(level)] = nn_blocks.PathAggregationBlock(
            filters=depth // 2, 
            inverted=True, 
            upsample=True,
            drop_final=False, 
            upsample_size=2, **self._base_config)
        self.preprocessors[str(level)] = nn_blocks.DarkRouteProcess(
            filters=depth,
            repetitions=self._fpn_path_len - int(level == self._min_level),
            block_invert=True,
            insert_spp=False,
            **self._base_config)
      else:
        self.preprocessors[str(level)] = nn_blocks.DarkRouteProcess(
            filters=depth,
            repetitions=self._fpn_path_len + 1 ,
            insert_spp=True,
            block_invert=False, 
            **self._base_config)

  def call(self, inputs):
    outputs = dict()
    layer_in = inputs[str(self._max_level)]
    for level in reversed(range(self._min_level, self._max_level + 1)):
      _, x = self.preprocessors[str(level)](layer_in)
      outputs[str(level)] = x
      if level > self._min_level:
        x_next = inputs[str(level - 1)]
        _, layer_in = self.resamples[str(level - 1)]([x_next, x])
    return outputs


@tf.keras.utils.register_keras_serializable(package='yolo')
class YoloPAN(tf.keras.layers.Layer):
  """YOLO Path Aggregation Network"""

  def __init__(self,
               path_process_len=6,
               max_level_process_len=None,
               embed_spp=False,
               embed_sam = False, 
               convert_csp = False, 
               activation='leaky',
               use_sync_bn=False,
               norm_momentum=0.99,
               norm_epsilon=0.001,
               kernel_initializer='glorot_uniform',
               kernel_regularizer=None,
               bias_regularizer=None,
               subdivisions=8,
               fpn_input=True,
               **kwargs):
    """
      Yolo Path Aggregation Network initialization function. Yolo V3 and V4
      Args:
        path_process_len: `int`, number of layers ot use in each Decoder path
        max_level_process_len: `int`, number of layers ot use in the largest
          processing path, or the backbones largest output if it is different
        embed_spp: `bool`, use the SPP found in the YoloV3 and V4 model
        use_sync_bn: if True, use synchronized batch normalization.
        norm_momentum: `float`, normalization omentum for the moving average.
        norm_epsilon: `float`, small float added to variance to avoid dividing
          by zero.
        activation: `str`, the activation function to use typically leaky
          or mish
        kernel_initializer: kernel_initializer for convolutional layers.
        kernel_regularizer: tf.keras.regularizers.Regularizer object for Conv2D.
        bias_regularizer: tf.keras.regularizers.Regularizer object for Conv2d.
        fpn_input: `bool`, for whether the input into this fucntion is an FPN or
          a backbone.
        **kwargs: keyword arguments to be passed.
    """
    super().__init__(**kwargs)

    self._path_process_len = path_process_len
    self._embed_spp = embed_spp
    self._embed_sam = embed_sam

    self._activation = activation
    self._use_sync_bn = use_sync_bn
    self._norm_momentum = norm_momentum
    self._norm_epsilon = norm_epsilon
    self._kernel_initializer = kernel_initializer
    self._kernel_regularizer = kernel_regularizer
    self._bias_regularizer = bias_regularizer
    self._subdivisions = subdivisions
    self._fpn_input = fpn_input
    self._max_level_process_len = max_level_process_len
    self._convert_csp = convert_csp

    # if self._fpn_input:
    #   if max_level_process_len is None:
    #     self._max_level_process_len = 6 #2
    # else:
    if max_level_process_len is None:
      self._max_level_process_len = path_process_len

    self._base_config = dict(
        activation=self._activation,
        use_sync_bn=self._use_sync_bn,
        kernel_regularizer=self._kernel_regularizer,
        kernel_initializer=self._kernel_initializer,
        bias_regularizer=self._bias_regularizer,
        subdivisions=self._subdivisions,
        norm_epsilon=self._norm_epsilon,
        norm_momentum=self._norm_momentum)

  def build(self, inputs):
    """
    use config dictionary to generate all important attributes for head
    construction

    Args:
      inputs: dictionary of the shape of input args as a dictionary of lists
    """
    # define the key order
    keys = [int(key) for key in inputs.keys()]
    self._min_level = min(keys)
    self._max_level = max(keys)
    self._min_depth = inputs[str(self._min_level)][-1]
    self._depths = self.get_raw_depths(self._min_depth)

    # directly connect to an input path and process it
    self.preprocessors = dict()
    # resample an input and merge it with the output of another path
    # inorder to aggregate backbone outputs
    self.resamples = dict()

    # FPN will reverse the key process order for the backbone, so we need
    # adjust the order that objects are created and processed to adjust for
    # this. not using an FPN will directly connect the decoder to the backbone
    # therefore the object creation order needs to be done from the largest
    # to smallest level.
    if self._fpn_input:
      # process order {... 3, 4, 5}
      self._iterator = range(self._min_level, self._max_level + 1)
      self._check = lambda x: x < self._max_level
      self._key_shift = lambda x: x + 1
      self._input = self._min_level
      downsample = True
      upsample = False
    else:
      # process order {5, 4, 3, ...}
      self._iterator = list(
          reversed(range(self._min_level, self._max_level + 1)))
      self._check = lambda x: x > self._min_level
      self._key_shift = lambda x: x - 1
      self._input = self._max_level
      downsample = False
      upsample = True
    
    proc_filters = lambda x: x
    resample_filters = lambda x: x // 2
    for level, depth in zip(self._iterator, self._depths):
      if level == self._input:
        self.preprocessors[str(level)] = nn_blocks.DarkRouteProcess(
            filters=proc_filters(depth),
            repetitions=self._max_level_process_len,
            insert_spp=self._embed_spp,
            block_invert=False, 
            insert_sam=self._embed_sam,
            **self._base_config)
      else:
        self.resamples[str(level)] = nn_blocks.PathAggregationBlock(
            filters=resample_filters(depth),
            upsample=upsample,
            downsample=downsample,
            inverted=False, 
            **self._base_config)
        self.preprocessors[str(level)] = nn_blocks.DarkRouteProcess(
            filters=proc_filters(depth),
            repetitions=self._path_process_len,
            insert_spp=False,
            insert_sam=self._embed_sam,
            **self._base_config)

  def get_raw_depths(self, minimum_depth):
    depths = []
    for _ in range(self._min_level, self._max_level + 1):
      depths.append(minimum_depth)
      minimum_depth *= 2
    if self._fpn_input:
      return depths
    else:
      return list(reversed(depths))

  def call(self, inputs):
    outputs = dict()
    layer_in = inputs[str(self._input)]

    for level in self._iterator:
      x_route, x = self.preprocessors[str(level)](layer_in)
      outputs[str(level)] = x
      if self._check(level):
        x_next = inputs[str(self._key_shift(level))]
        _, layer_in = self.resamples[str(
            self._key_shift(level))]([x_route, x_next])
    return outputs


@tf.keras.utils.register_keras_serializable(package='yolo')
class YoloDecoder(tf.keras.Model):
  """Darknet Backbone Decoder"""

  def __init__(self,
               input_specs,
               embed_fpn=False,
               embed_sam=False, 
               convert_csp=False, 
               fpn_path_len=4,
               path_process_len=6,
               max_level_process_len=None,
               embed_spp=False,
               activation='leaky',
               use_sync_bn=False,
               norm_momentum=0.99,
               norm_epsilon=0.001,
               kernel_initializer='glorot_uniform',
               kernel_regularizer=None,
               bias_regularizer=None,
               subdivisions=8,
               **kwargs):
    """
    Yolo Decoder initialization function. A unified model that ties all decoder
    components into a conditionally build YOLO decder.

    Args:
      input_specs: `dict[str, tf.InputSpec]`: input specs of each of the inputs
        to the heads
      embed_fpn: `bool`, use the FPN found in the YoloV4 model
      fpn_path_len: `int`, number of layers ot use in each FPN path
        if you choose to use an FPN
      path_process_len: `int`, number of layers ot use in each Decoder path
      max_level_process_len: `int`, number of layers ot use in the largest
        processing path, or the backbones largest output if it is different
      embed_spp: `bool`, use the SPP found in the YoloV3 and V4 model
      use_sync_bn: if True, use synchronized batch normalization.
      norm_momentum: `float`, normalization omentum for the moving average.
      norm_epsilon: `float`, small float added to variance to avoid dividing by
        zero.
      activation: `str`, the activation function to use typically leaky or mish
      kernel_initializer: kernel_initializer for convolutional layers.
      kernel_regularizer: tf.keras.regularizers.Regularizer object for Conv2D.
      bias_regularizer: tf.keras.regularizers.Regularizer object for Conv2d.
      **kwargs: keyword arguments to be passed.
    """

    self._input_specs = input_specs
    self._embed_fpn = embed_fpn
    self._fpn_path_len = fpn_path_len
    self._path_process_len = path_process_len
    self._max_level_process_len = max_level_process_len
    self._embed_spp = embed_spp

    self._activation = activation
    self._use_sync_bn = use_sync_bn
    self._norm_momentum = norm_momentum
    self._norm_epsilon = norm_epsilon
    self._kernel_initializer = kernel_initializer
    self._kernel_regularizer = kernel_regularizer
    self._bias_regularizer = bias_regularizer
    self._subdivisions = subdivisions

    self._base_config = dict(
        embed_sam = embed_sam, 
        convert_csp = convert_csp, 
        activation=self._activation,
        use_sync_bn=self._use_sync_bn,
        norm_momentum=self._norm_momentum,
        norm_epsilon=self._norm_epsilon,
        kernel_initializer=self._kernel_initializer,
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer,
        subdivisions=self._subdivisions)

    self._decoder_config = dict(
        path_process_len=self._path_process_len,
        max_level_process_len=self._max_level_process_len,
        embed_spp=self._embed_spp,
        fpn_input=self._embed_fpn,
        **self._base_config)

    inputs = {
        key: tf.keras.layers.Input(shape=value[1:])
        for key, value in input_specs.items()
    }
    if self._embed_fpn:
      inter_outs = YoloFPN(
          fpn_path_len=self._fpn_path_len, **self._base_config)(
              inputs)
      outputs = YoloPAN(**self._decoder_config)(inter_outs)
    else:
      inter_outs = None
      outputs = YoloPAN(**self._decoder_config)(inputs)

    self._output_specs = {key: value.shape for key, value in outputs.items()}
    super().__init__(inputs=inputs, outputs=outputs, name='YoloDecoder')

  @property
  def embed_fpn(self):
    return self._embed_fpn

  @property
  def output_specs(self):
    return self._output_specs

  def get_config(self):
    config = dict(
        input_specs=self._input_specs,
        embed_fpn=self._embed_fpn,
        fpn_path_len=self._fpn_path_len,
        **self._decoder_config)
    return config

  @classmethod
  def from_config(cls, config, custom_objects=None):
    return cls(**config)

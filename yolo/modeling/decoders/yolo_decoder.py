import tensorflow as tf
from yolo.modeling.layers import nn_blocks


@tf.keras.utils.register_keras_serializable(package="yolo")
class YoloFPN(tf.keras.layers.Layer):

  def __init__(self,
               fpn_path_len=4,
               activation="leaky",
               use_sync_bn=False,
               norm_momentum=0.99,
               norm_epsilon=0.001,
               kernel_initializer="glorot_uniform",
               kernel_regularizer=None,
               bias_regularizer=None,
               subdivisions=8,
               **kwargs):

    super().__init__(**kwargs)
    self._fpn_path_len = fpn_path_len

    self._activation = "leaky" if activation is None else activation
    self._use_sync_bn = use_sync_bn
    self._norm_momentum = norm_momentum
    self._norm_epsilon = norm_epsilon
    self._kernel_initializer = kernel_initializer
    self._kernel_regularizer = kernel_regularizer
    self._bias_regularizer = bias_regularizer
    self._subdivisions = subdivisions

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
    """ use config dictionary to generate all important attributes for head construction """
    keys = [int(key) for key in inputs.keys()]
    self._min_level = min(keys)
    self._max_level = max(keys)
    self._min_depth = inputs[str(self._min_level)][-1]
    self._depths = self.get_raw_depths(self._min_depth)

    self.resamples = {}
    self.preprocessors = {}
    self.tails = {}
    for level, depth in zip(
        reversed(range(self._min_level, self._max_level + 1)), self._depths):

      if level != self._max_level:
        self.resamples[str(level)] = nn_blocks.RouteMerge(
            filters=depth // 2, **self._base_config)
        self.preprocessors[str(level)] = nn_blocks.DarkRouteProcess(
            filters=depth,
            repetitions=self._fpn_path_len,
            insert_spp=False,
            **self._base_config)
      else:
        self.preprocessors[str(level)] = nn_blocks.DarkRouteProcess(
            filters=depth,
            repetitions=self._fpn_path_len + 2,
            insert_spp=True,
            **self._base_config)
      if level == self._min_level:
        self.tails[str(level)] = nn_blocks.FPNTail(
            filters=depth, upsample=False, **self._base_config)
      else:
        self.tails[str(level)] = nn_blocks.FPNTail(
            filters=depth, upsample=True, **self._base_config)
    return

  def call(self, inputs):
    outputs = {}
    layer_in = inputs[str(self._max_level)]
    for level in reversed(range(self._min_level, self._max_level + 1)):
      _, x = self.preprocessors[str(level)](layer_in)
      if level > self._min_level:
        x_route, x = self.tails[str(level)](x)
        x_next = inputs[str(level - 1)]
        layer_in = self.resamples[str(level - 1)]([x_next, x])
      else:
        x_route = self.tails[str(level)](x)
      outputs[str(level)] = x_route
    return outputs


@tf.keras.utils.register_keras_serializable(package="yolo")
class YoloPAN(tf.keras.layers.Layer):

  def __init__(self,
               path_process_len=6,
               max_level_process_len=None,
               embed_spp=False,
               activation="leaky",
               use_sync_bn=False,
               norm_momentum=0.99,
               norm_epsilon=0.001,
               kernel_initializer="glorot_uniform",
               kernel_regularizer=None,
               bias_regularizer=None,
               subdivisions=8,
               fpn_input=True,
               **kwargs):
    super().__init__(**kwargs)

    self._path_process_len = path_process_len
    self._embed_spp = embed_spp

    self._activation = "leaky" if activation is None else activation
    self._use_sync_bn = use_sync_bn
    self._norm_momentum = norm_momentum
    self._norm_epsilon = norm_epsilon
    self._kernel_initializer = kernel_initializer
    self._kernel_regularizer = kernel_regularizer
    self._bias_regularizer = bias_regularizer
    self._subdivisions = subdivisions
    self._fpn_input = fpn_input

    if self._fpn_input:
      self._max_level_process_len = 1 if max_level_process_len is None else max_level_process_len
    else:
      self._max_level_process_len = path_process_len if max_level_process_len is None else max_level_process_len

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
    keys = [int(key) for key in inputs.keys()]
    self._min_level = min(keys)
    self._max_level = max(keys)
    self._min_depth = inputs[str(self._min_level)][-1]
    self._depths = self.get_raw_depths(self._min_depth)

    self.resamples = {}
    self.preprocessors = {}
    self.outputs = {}

    if self._fpn_input:
      self._iterator = range(self._min_level, self._max_level + 1)
      self._check = lambda x: x < self._max_level
      self._key_shift = lambda x: x + 1
      self._input = self._min_level
      proc_filters = lambda x: x * 2
      resample_filters = lambda x: x

      downsample = True
      upsample = False
    else:
      self._iterator = list(
          reversed(range(self._min_level, self._max_level + 1)))
      self._check = lambda x: x > self._min_level
      self._key_shift = lambda x: x - 1
      self._input = self._max_level
      proc_filters = lambda x: x
      resample_filters = lambda x: x // 2

      downsample = False
      upsample = True

    for level, depth in zip(self._iterator, self._depths):
      if level == self._input:
        self.preprocessors[str(level)] = nn_blocks.DarkRouteProcess(
            filters=proc_filters(depth),
            repetitions=self._max_level_process_len + 2 *
            (1 if self._embed_spp else 0),
            insert_spp=self._embed_spp,
            **self._base_config)
      else:
        self.resamples[str(level)] = nn_blocks.RouteMerge(
            filters=resample_filters(depth),
            upsample=upsample,
            downsample=downsample,
            **self._base_config)
        self.preprocessors[str(level)] = nn_blocks.DarkRouteProcess(
            filters=proc_filters(depth),
            repetitions=self._path_process_len,
            insert_spp=False,
            **self._base_config)
    return

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
        layer_in = self.resamples[str(
            self._key_shift(level))]([x_route, x_next])
    return outputs


@tf.keras.utils.register_keras_serializable(package="yolo")
class YoloDecoder(tf.keras.Model):

  def __init__(self,
               input_specs,
               embed_fpn=False,
               fpn_path_len=4,
               path_process_len=6,
               max_level_process_len=None,
               embed_spp=False,
               activation="leaky",
               use_sync_bn=False,
               norm_momentum=0.99,
               norm_epsilon=0.001,
               kernel_initializer="glorot_uniform",
               kernel_regularizer=None,
               bias_regularizer=None,
               subdivisions=8,
               **kwargs):
    super().__init__(**kwargs)
    self._embed_fpn = embed_fpn
    self._fpn_path_len = fpn_path_len
    self._path_process_len = path_process_len
    self._max_level_process_len = max_level_process_len
    self._embed_spp = embed_spp

    self._activation = "leaky" if activation is None else activation
    self._use_sync_bn = use_sync_bn
    self._norm_momentum = norm_momentum
    self._norm_epsilon = norm_epsilon
    self._kernel_initializer = kernel_initializer
    self._kernel_regularizer = kernel_regularizer
    self._bias_regularizer = bias_regularizer
    self._subdivisions = subdivisions

    self._base_config = dict(
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
    print(self._output_specs)
    super().__init__(inputs=inputs, outputs=outputs, name="YoloDecoder")

  @property
  def embed_fpn(self):
    return self._embed_fpn

  @property
  def output_specs(self):
    return self._output_specs

  def get_config(self):
    config = dict(
        embed_fpn=self._embed_fpn,
        fpn_path_len=self._fpn_path_len,
        **self._decoder_config)
    return config

  @classmethod
  def from_config(cls, config, custom_objects=None):
    return cls(**config)


# @tf.keras.utils.register_keras_serializable(package="yolo")
# class YoloRoutedDecoder(tf.keras.layers.Layer):

#   def __init__(self,
#                path_process_len=6,
#                max_level_process_len=None,
#                embed_spp=False,
#                activation="leaky",
#                use_sync_bn=False,
#                norm_momentum=0.99,
#                norm_epsilon=0.001,
#                kernel_initializer="glorot_uniform",
#                kernel_regularizer=None,
#                bias_regularizer=None,
#                subdivisions = 8,
#                fpn_input = True,
#                **kwargs):
#     super().__init__(**kwargs)
#     self._max_level_process_len = path_process_len if max_level_process_len is None else max_level_process_len
#     self._path_process_len = path_process_len
#     self._embed_spp = embed_spp

#     self._activation = "leaky" if activation is None else activation
#     self._use_sync_bn = use_sync_bn
#     self._norm_momentum = norm_momentum
#     self._norm_epsilon = norm_epsilon
#     self._kernel_initializer = kernel_initializer
#     self._kernel_regularizer = kernel_regularizer
#     self._bias_regularizer = bias_regularizer
#     self._subdivisions = subdivisions
#     self._fpn_input = fpn_input

#     self._base_config = dict(
#         activation=self._activation,
#         use_sync_bn=self._use_sync_bn,
#         kernel_regularizer=self._kernel_regularizer,
#         kernel_initializer=self._kernel_initializer,
#         bias_regularizer=self._bias_regularizer,
#         subdivisions = self._subdivisions,
#         norm_epsilon=self._norm_epsilon,
#         norm_momentum=self._norm_momentum)

#   def build(self, inputs):
#     keys = [int(key) for key in inputs.keys()]
#     self._min_level = min(keys)
#     self._max_level = max(keys)
#     self._min_depth = inputs[str(self._min_level)][-1]
#     self._depths = self.get_raw_depths(self._min_depth)

#     self.resamples = {}
#     self.preprocessors = {}
#     self.outputs = {}

#     for level, depth in zip(
#         reversed(range(self._min_level, self._max_level + 1)), self._depths):
#       if level == self._max_level:
#         self.preprocessors[str(level)] = nn_blocks.DarkRouteProcess(
#             filters=depth,
#             repetitions=self._max_level_process_len + 2 *
#             (1 if self._embed_spp else 0),
#             insert_spp=self._embed_spp,
#             **self._base_config)
#       else:
#         self.resamples[str(level)] = nn_blocks.RouteMerge(
#             filters=depth // 2, upsample=True, **self._base_config)
#         self.preprocessors[str(level)] = nn_blocks.DarkRouteProcess(
#             filters=depth,
#             repetitions=self._path_process_len,
#             insert_spp=False,
#             **self._base_config)

#     return

#   def get_raw_depths(self, minimum_depth):
#     depths = []
#     for _ in range(self._min_level, self._max_level + 1):
#       depths.append(minimum_depth)
#       minimum_depth *= 2
#     return list(reversed(depths))

#   def call(self, inputs):
#     outputs = dict()
#     layer_in = inputs[str(self._max_level)]
#     for level in reversed(range(self._min_level, self._max_level + 1)):
#       x_route, x = self.preprocessors[str(level)](layer_in)
#       outputs[str(level)] = x
#       if level > self._min_level:
#         x_next = inputs[str(level - 1)]
#         layer_in = self.resamples[str(level - 1)]([x_route, x_next])
#     return outputs

# @tf.keras.utils.register_keras_serializable(package="yolo")
# class YoloFPNDecoder(tf.keras.layers.Layer):

#   def __init__(self,
#                path_process_len=6,
#                max_level_process_len=None,
#                embed_spp=False,
#                activation="leaky",
#                use_sync_bn=False,
#                norm_momentum=0.99,
#                norm_epsilon=0.001,
#                kernel_initializer="glorot_uniform",
#                kernel_regularizer=None,
#                bias_regularizer=None,
#                subdivisions = 8,
#                **kwargs):
#     super().__init__(**kwargs)
#     self._path_process_len = path_process_len
#     self._max_level_process_len = 1 if max_level_process_len is None else max_level_process_len

#     self._embed_spp = embed_spp

#     self._activation = "leaky" if activation is None else activation
#     self._use_sync_bn = use_sync_bn
#     self._norm_momentum = norm_momentum
#     self._norm_epsilon = norm_epsilon
#     self._kernel_initializer = kernel_initializer
#     self._kernel_regularizer = kernel_regularizer
#     self._bias_regularizer = bias_regularizer
#     self._subdivisions = subdivisions

#     self._base_config = dict(
#         activation=self._activation,
#         use_sync_bn=self._use_sync_bn,
#         subdivisions=self._subdivisions,
#         kernel_regularizer=self._kernel_regularizer,
#         kernel_initializer=self._kernel_initializer,
#         bias_regularizer=self._bias_regularizer,
#         norm_epsilon=self._norm_epsilon,
#         norm_momentum=self._norm_momentum)

#   def get_raw_depths(self, minimum_depth):
#     depths = []
#     for _ in range(self._min_level, self._max_level + 1):
#       depths.append(minimum_depth)
#       minimum_depth *= 2
#     return depths

#   def build(self, inputs):
#     keys = [int(key) for key in inputs.keys()]
#     print(inputs)
#     self._min_level = min(keys)
#     self._max_level = max(keys)
#     self._min_depth = inputs[str(self._min_level)][-1]
#     self._depths = self.get_raw_depths(self._min_depth)
#     print(self._depths)

#     self.resamples = {}
#     self.preprocessors = {}
#     self.outputs = {}

#     for level, depth in zip(
#         range(self._min_level, self._max_level + 1), self._depths):
#       if level == self._min_level:
#         self.preprocessors[str(level)] = nn_blocks.DarkRouteProcess(
#             filters=depth * 2,
#             repetitions=self._max_level_process_len + 2 *
#             (1 if self._embed_spp else 0),
#             insert_spp=self._embed_spp,
#             **self._base_config)
#       else:
#         self.resamples[str(level)] = nn_blocks.RouteMerge(
#             filters=depth, downsample=True, **self._base_config)
#         self.preprocessors[str(level)] = nn_blocks.DarkRouteProcess(
#             filters=depth * 2,
#             repetitions=self._path_process_len,
#             insert_spp=False,
#             **self._base_config)

#     # print(inputs_)

#   def call(self, inputs):
#     outputs = dict()
#     layer_in = inputs[str(self._min_level)]
#     for level in range(self._min_level, self._max_level + 1):
#       x_route, x = self.preprocessors[str(level)](layer_in)
#       outputs[str(level)] = x
#       if level < self._max_level:
#         x_next = inputs[str(level + 1)]
#         layer_in = self.resamples[str(level + 1)]([x_route, x_next])
#     return outputs

# @tf.keras.utils.register_keras_serializable(package="yolo")
# class YoloDecoder(tf.keras.Model):

#   def __init__(self,
#                input_specs,
#                embed_fpn=False,
#                fpn_path_len=4,
#                path_process_len=6,
#                max_level_process_len=None,
#                embed_spp=False,
#                activation="leaky",
#                use_sync_bn=False,
#                norm_momentum=0.99,
#                norm_epsilon=0.001,
#                kernel_initializer="glorot_uniform",
#                kernel_regularizer=None,
#                bias_regularizer=None,
#                subdivisions = 8,
#                **kwargs):
#     # super().__init__(**kwargs)
#     self._embed_fpn = embed_fpn
#     self._fpn_path_len = fpn_path_len
#     self._path_process_len = path_process_len
#     self._max_level_process_len = max_level_process_len
#     self._embed_spp = embed_spp

#     self._activation = "leaky" if activation is None else activation
#     self._use_sync_bn = use_sync_bn
#     self._norm_momentum = norm_momentum
#     self._norm_epsilon = norm_epsilon
#     self._kernel_initializer = kernel_initializer
#     self._kernel_regularizer = kernel_regularizer
#     self._bias_regularizer = bias_regularizer
#     self._subdivisions = subdivisions

#     self._base_config = dict(
#         activation=self._activation,
#         use_sync_bn=self._use_sync_bn,
#         norm_momentum=self._norm_momentum,
#         norm_epsilon=self._norm_epsilon,
#         kernel_initializer=self._kernel_initializer,
#         kernel_regularizer=self._kernel_regularizer,
#         bias_regularizer=self._bias_regularizer,
#         subdivisions = self._subdivisions)

#     self._decoder_config = dict(
#         path_process_len=self._path_process_len,
#         max_level_process_len=self._max_level_process_len,
#         embed_spp=self._embed_spp,
#         **self._base_config)

#     inputs = {key: tf.keras.layers.Input(shape = value[1:]) for key, value in input_specs.items()}
#     if self._embed_fpn:
#       inter_outs = YoloFPN(fpn_path_len=self._fpn_path_len, **self._base_config)(inputs)
#       outputs = YoloFPNDecoder(**self._decoder_config)(inter_outs)
#     else:
#       outputs = YoloRoutedDecoder(**self._decoder_config)(inputs)

#     self._output_specs = {key: value.shape for key, value in outputs.items()}
#     super().__init__(inputs=inputs, outputs=outputs, name='YoloDecoder')

#   @property
#   def embed_fpn(self):
#     return self._embed_fpn

#   @property
#   def output_specs(self):
#     return self._output_specs

#   def get_config(self):
#     config = dict(
#         embed_fpn=self._embed_fpn,
#         fpn_path_len=self._fpn_path_len,
#         **self._decoder_config)
#     return config

#   @classmethod
#   def from_config(cls, config, custom_objects=None):
#     return cls(**config)

"""Contains common building blocks for yolo neural networks."""
from typing import Callable
import tensorflow as tf
from yolo.modeling.layers import subnormalization
from official.modeling import tf_utils
from official.vision.beta.ops import spatial_transform_ops

TPU_BASE = False


@tf.keras.utils.register_keras_serializable(package='yolo')
class Identity(tf.keras.layers.Layer):

  def __init__(self, **kwargs):
    super().__init__(**kwargs)

  def call(self, input):
    return input


@tf.keras.utils.register_keras_serializable(package='yolo')
class ConvBN(tf.keras.layers.Layer):
  """
    Modified Convolution layer to match that of the DarkNet Library.
    The Layer is a standards combination of Conv BatchNorm Activation,
    however, the use of bias in the conv is determined by the use of batch
    normalization.
    Cross Stage Partial networks (CSPNets) were proposed in:
    [1] Chien-Yao Wang, Hong-Yuan Mark Liao, I-Hau Yeh, Yueh-Hua Wu,
          Ping-Yang Chen, Jun-Wei Hsieh
        CSPNet: A New Backbone that can Enhance Learning Capability of CNN.
          arXiv:1911.11929
    Args:
      filters: integer for output depth, or the number of features to learn
      kernel_size: integer or tuple for the shape of the weight matrix or kernel
        to learn
      strides: integer of tuple how much to move the kernel after each kernel
        use
      padding: string 'valid' or 'same', if same, then pad the image, else do
        not
      dialtion_rate: tuple to indicate how much to modulate kernel weights and
        how many pixels in a feature map to skip
      kernel_initializer: string to indicate which function to use to initialize
        weights
      bias_initializer: string to indicate which function to use to initialize
        bias
      kernel_regularizer: string to indicate which function to use to
        regularizer weights
      bias_regularizer: string to indicate which function to use to regularizer
        bias
      use_bn: boolean for whether to use batch normalization
      use_sync_bn: boolean for whether sync batch normalization statistics
        of all batch norm layers to the models global statistics
        (across all input batches)
      norm_momentum: float for moment to use for batch normalization
      norm_epsilon: float for batch normalization epsilon
      activation: string or None for activation function to use in layer,
        if None activation is replaced by linear
      leaky_alpha: float to use as alpha if activation function is leaky
      **kwargs: Keyword Arguments
    """

  def __init__(self,
               filters=1,
               kernel_size=(1, 1),
               strides=(1, 1),
               padding='same',
               dilation_rate=(1, 1),
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               subdivisions=1,
               bias_regularizer=None,
               kernel_regularizer=None,
               use_bn=True,
               use_sync_bn=False,
               norm_momentum=0.99,
               norm_epsilon=0.001,
               activation='leaky',
               leaky_alpha=0.1,
               **kwargs):

    # convolution params
    self._filters = filters
    self._kernel_size = kernel_size
    self._strides = strides
    self._padding = padding
    self._dilation_rate = dilation_rate
    self._kernel_initializer = kernel_initializer
    self._bias_initializer = bias_initializer
    self._kernel_regularizer = kernel_regularizer

    self._bias_regularizer = bias_regularizer
    self._subdivisions = subdivisions

    # batch normalization params
    self._use_bn = use_bn
    self._use_sync_bn = use_sync_bn
    self._norm_momentum = norm_momentum
    self._norm_epsilon = norm_epsilon

    if tf.keras.backend.image_data_format() == 'channels_last':
      # format: (batch_size, height, width, channels)
      self._bn_axis = -1
    else:
      # format: (batch_size, channels, width, height)
      self._bn_axis = 1

    # activation params
    self._activation = activation
    self._leaky_alpha = leaky_alpha

    super().__init__(**kwargs)

  def build(self, input_shape):
    use_bias = not self._use_bn

    if not TPU_BASE:
      kernel_size = self._kernel_size if isinstance(
          self._kernel_size, int) else self._kernel_size[0]
      dilation_rate = self._dilation_rate if isinstance(
          self._dilation_rate, int) else self._dilation_rate[0]
      if self._padding == 'same' and kernel_size != 1:
        padding = (dilation_rate * (kernel_size - 1))
        left_shift = padding // 2
        self._paddings = tf.constant([[0, 0], [left_shift, left_shift],
                                      [left_shift, left_shift], [0, 0]])
        # self._zeropad = tf.keras.layers.ZeroPadding2D()
      else:
        self._paddings = tf.constant([[0, 0], [0, 0], [0, 0], [0, 0]])

      self.conv = tf.keras.layers.Conv2D(
          filters=self._filters,
          kernel_size=self._kernel_size,
          strides=self._strides,
          padding='valid',
          dilation_rate=self._dilation_rate,
          use_bias=use_bias,
          kernel_initializer=self._kernel_initializer,
          bias_initializer=self._bias_initializer,
          kernel_regularizer=self._kernel_regularizer,
          bias_regularizer=self._bias_regularizer)
    else:
      self.conv = tf.keras.layers.Conv2D(
          filters=self._filters,
          kernel_size=self._kernel_size,
          strides=self._strides,
          padding=self._padding,  # 'valid',
          dilation_rate=self._dilation_rate,
          use_bias=use_bias,
          kernel_initializer=self._kernel_initializer,
          bias_initializer=self._bias_initializer,
          kernel_regularizer=self._kernel_regularizer,
          bias_regularizer=self._bias_regularizer)

    if self._use_bn:
      if self._use_sync_bn:
        self.bn = tf.keras.layers.experimental.SyncBatchNormalization(
            momentum=self._norm_momentum,
            epsilon=self._norm_epsilon,
            axis=self._bn_axis)
      else:
        self.bn = tf.keras.layers.BatchNormalization(
            momentum=self._norm_momentum,
            epsilon=self._norm_epsilon,
            axis=self._bn_axis)

    if self._activation == 'leaky':
      self._activation_fn = tf.keras.layers.LeakyReLU(alpha=self._leaky_alpha)
    elif self._activation == 'mish':
      self._activation_fn = lambda x: x * tf.math.tanh(tf.math.softplus(x))
    else:
      self._activation_fn = tf_utils.get_activation(self._activation)

  def call(self, x):
    if not TPU_BASE:
      x = tf.pad(x, self._paddings, mode='CONSTANT', constant_values=0)
    x = self.conv(x)
    if self._use_bn:
      x = self.bn(x)
    x = self._activation_fn(x)
    return x

  def get_config(self):
    # used to store/share parameters to reconstruct the model
    layer_config = {
        'filters': self._filters,
        'kernel_size': self._kernel_size,
        'strides': self._strides,
        'padding': self._padding,
        'dilation_rate': self._dilation_rate,
        'kernel_initializer': self._kernel_initializer,
        'bias_initializer': self._bias_initializer,
        'bias_regularizer': self._bias_regularizer,
        'kernel_regularizer': self._kernel_regularizer,
        'use_bn': self._use_bn,
        'use_sync_bn': self._use_sync_bn,
        'norm_momentum': self._norm_momentum,
        'norm_epsilon': self._norm_epsilon,
        'activation': self._activation,
        'leaky_alpha': self._leaky_alpha
    }
    layer_config.update(super().get_config())
    return layer_config


@tf.keras.utils.register_keras_serializable(package='yolo')
class DarkResidual(tf.keras.layers.Layer):
  """
  DarkNet block with Residual connection for Yolo v3 Backbone
  Args:
    filters: integer for output depth, or the number of features to learn
    kernel_initializer: string to indicate which function to use to initialize
      weights
    bias_initializer: string to indicate which function to use to initialize
      bias
    kernel_regularizer: string to indicate which function to use to regularizer
      weights
    bias_regularizer: string to indicate which function to use to regularizer
      bias
    use_bn: boolean for whether to use batch normalization
    use_sync_bn: boolean for whether sync batch normalization statistics
      of all batch norm layers to the models global statistics
      (across all input batches)
    norm_momentum: float for moment to use for batch normalization
    norm_epsilon: float for batch normalization epsilon
    conv_activation: string or None for activation function to use in layer,
      if None activation is replaced by linear
    leaky_alpha: float to use as alpha if activation function is leaky
    sc_activation: string for activation function to use in layer
    downsample: boolean for if image input is larger than layer output, set
      downsample to True so the dimensions are forced to match
    **kwargs: Keyword Arguments
  """

  def __init__(self,
               filters=1,
               filter_scale=2,
               dilation_rate=1,
               subdivisions=8,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               use_bn=True,
               use_sync_bn=False,
               norm_momentum=0.99,
               norm_epsilon=0.001,
               activation='leaky',
               leaky_alpha=0.1,
               sc_activation='linear',
               downsample=False,
               **kwargs):

    # downsample
    self._downsample = downsample

    # ConvBN params
    self._filters = filters
    self._filter_scale = filter_scale
    self._kernel_initializer = kernel_initializer
    self._bias_initializer = bias_initializer
    self._bias_regularizer = bias_regularizer
    self._use_bn = use_bn
    self._use_sync_bn = use_sync_bn
    self._kernel_regularizer = kernel_regularizer
    self._subdivisions = subdivisions

    # normal params
    self._norm_momentum = norm_momentum
    self._norm_epsilon = norm_epsilon
    self._dilation_rate = dilation_rate if isinstance(dilation_rate,
                                                      int) else dilation_rate[0]

    # activation params
    self._conv_activation = activation
    self._leaky_alpha = leaky_alpha
    self._sc_activation = sc_activation

    super().__init__(**kwargs)

  def build(self, input_shape):
    _dark_conv_args = {
        'kernel_initializer': self._kernel_initializer,
        'bias_initializer': self._bias_initializer,
        'bias_regularizer': self._bias_regularizer,
        'use_bn': self._use_bn,
        'use_sync_bn': self._use_sync_bn,
        'norm_momentum': self._norm_momentum,
        'subdivisions': self._subdivisions,
        'norm_epsilon': self._norm_epsilon,
        'activation': self._conv_activation,
        'kernel_regularizer': self._kernel_regularizer,
        'leaky_alpha': self._leaky_alpha
    }
    if self._downsample:
      if self._dilation_rate > 1:
        dilation_rate = 1
        if self._dilation_rate // 2 > 0:
          dilation_rate = self._dilation_rate // 2
        down_stride = 1
      else:
        dilation_rate = 1
        down_stride = 2

      self._dconv = ConvBN(
          filters=self._filters,
          kernel_size=(3, 3),
          strides=down_stride,
          dilation_rate=dilation_rate,
          padding='same',
          **_dark_conv_args)

    self._conv1 = ConvBN(
        filters=self._filters // self._filter_scale,
        kernel_size=(1, 1),
        strides=(1, 1),
        #dilation_rate= self._dilation_rate,
        padding='same',
        **_dark_conv_args)

    self._conv2 = ConvBN(
        filters=self._filters,
        kernel_size=(3, 3),
        strides=(1, 1),
        dilation_rate=self._dilation_rate,
        padding='same',
        **_dark_conv_args)

    # if self._stochastic_depth_drop_rate:
    #   self._stochastic_depth = nn_layers.StochasticDepth(
    #       self._stochastic_depth_drop_rate)
    # else:
    #   self._stochastic_depth = None

    self._shortcut = tf.keras.layers.Add()
    if self._sc_activation == 'leaky':
      self._activation_fn = tf.keras.layers.LeakyReLU(alpha=self._leaky_alpha)
    elif self._sc_activation == 'mish':
      self._activation_fn = lambda x: x * tf.math.tanh(tf.math.softplus(x))
    else:
      self._activation_fn = tf_utils.get_activation(
          self._sc_activation
      )  # tf.keras.layers.Activation(self._sc_activation)
    super().build(input_shape)

  def call(self, inputs, training=None):
    if self._downsample:
      inputs = self._dconv(inputs)
    x = self._conv1(inputs)
    x = self._conv2(x)

    # if self._stochastic_depth:
    #   x = self._stochastic_depth(x, training=training)

    x = self._shortcut([x, inputs])
    return self._activation_fn(x)

  def get_config(self):
    # used to store/share parameters to reconstruct the model
    layer_config = {
        'filters': self._filters,
        'kernel_initializer': self._kernel_initializer,
        'bias_initializer': self._bias_initializer,
        'kernel_regularizer': self._kernel_regularizer,
        'dilation_rate': self._dilation_rate,
        'use_bn': self._use_bn,
        'use_sync_bn': self._use_sync_bn,
        'norm_momentum': self._norm_momentum,
        'norm_epsilon': self._norm_epsilon,
        'activation': self._conv_activation,
        'leaky_alpha': self._leaky_alpha,
        'sc_activation': self._sc_activation,
        'downsample': self._downsample,
        'subdivisions': self._subdivisions,
    }
    layer_config.update(super().get_config())
    return layer_config


@tf.keras.utils.register_keras_serializable(package='yolo')
class CSPTiny(tf.keras.layers.Layer):
  """
  A Small size convolution block proposed in the CSPNet. The layer uses
  shortcuts, routing(concatnation), and feature grouping in order to improve
  gradient variablity and allow for high efficency, low power residual learning
  for small networtf.keras.
  Cross Stage Partial networks (CSPNets) were proposed in:
  [1] Chien-Yao Wang, Hong-Yuan Mark Liao, I-Hau Yeh, Yueh-Hua Wu,
        Ping-Yang Chen, Jun-Wei Hsieh
      CSPNet: A New Backbone that can Enhance Learning Capability of CNN.
        arXiv:1911.11929
  Args:
    filters: integer for output depth, or the number of features to learn
    kernel_initializer: string to indicate which function to use to initialize
      weights
    bias_initializer: string to indicate which function to use to initialize
      bias
    use_bn: boolean for whether to use batch normalization
    kernel_regularizer: string to indicate which function to use to regularizer
      weights
    bias_regularizer: string to indicate which function to use to regularizer
      bias
    use_sync_bn: boolean for whether sync batch normalization statistics
      of all batch norm layers to the models global statistics
      (across all input batches)
    group_id: integer for which group of features to pass through the csp
      tiny stack.
    groups: integer for how many splits there should be in the convolution
      feature stack output
    norm_momentum: float for moment to use for batch normalization
    norm_epsilon: float for batch normalization epsilon
    conv_activation: string or None for activation function to use in layer,
      if None activation is replaced by linear
    leaky_alpha: float to use as alpha if activation function is leaky
    sc_activation: string for activation function to use in layer
    downsample: boolean for if image input is larger than layer output, set
      downsample to True so the dimensions are forced to match
    **kwargs: Keyword Arguments
  """

  def __init__(self,
               filters=1,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               bias_regularizer=None,
               kernel_regularizer=None,
               use_bn=True,
               dilation_rate=1,
               subdivisions=1,
               use_sync_bn=False,
               group_id=1,
               groups=2,
               norm_momentum=0.99,
               norm_epsilon=0.001,
               activation='leaky',
               downsample=True,
               leaky_alpha=0.1,
               **kwargs):

    # ConvBN params
    self._filters = filters
    self._kernel_initializer = kernel_initializer
    self._bias_initializer = bias_initializer
    self._bias_regularizer = bias_regularizer
    self._use_bn = use_bn
    self._dilation_rate = dilation_rate
    self._use_sync_bn = use_sync_bn
    self._kernel_regularizer = kernel_regularizer
    self._groups = groups
    self._group_id = group_id
    self._downsample = downsample
    self._subdivisions = subdivisions

    # normal params
    self._norm_momentum = norm_momentum
    self._norm_epsilon = norm_epsilon

    # activation params
    self._conv_activation = activation
    self._leaky_alpha = leaky_alpha

    super().__init__(**kwargs)

  def build(self, input_shape):
    _dark_conv_args = {
        'kernel_initializer': self._kernel_initializer,
        'bias_initializer': self._bias_initializer,
        'bias_regularizer': self._bias_regularizer,
        'use_bn': self._use_bn,
        'use_sync_bn': self._use_sync_bn,
        'subdivisions': self._subdivisions,
        'norm_momentum': self._norm_momentum,
        'norm_epsilon': self._norm_epsilon,
        'activation': self._conv_activation,
        'kernel_regularizer': self._kernel_regularizer,
        'leaky_alpha': self._leaky_alpha
    }
    self._convlayer1 = ConvBN(
        filters=self._filters,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='same',
        **_dark_conv_args)

    self._convlayer2 = ConvBN(
        filters=self._filters // 2,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='same',
        kernel_initializer=self._kernel_initializer,
        bias_initializer=self._bias_initializer,
        bias_regularizer=self._bias_regularizer,
        kernel_regularizer=self._kernel_regularizer,
        use_bn=self._use_bn,
        use_sync_bn=self._use_sync_bn,
        norm_momentum=self._norm_momentum,
        norm_epsilon=self._norm_epsilon,
        activation=self._conv_activation,
        leaky_alpha=self._leaky_alpha)

    self._convlayer3 = ConvBN(
        filters=self._filters // 2,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='same',
        **_dark_conv_args)

    self._convlayer4 = ConvBN(
        filters=self._filters,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding='same',
        **_dark_conv_args)

    if self._downsample:
      self._maxpool = tf.keras.layers.MaxPool2D(
          pool_size=2, strides=2, padding='same', data_format=None)

    super().build(input_shape)

  def call(self, inputs, training=None):
    x1 = self._convlayer1(inputs)
    x1_group = tf.split(x1, self._groups, axis=-1)[self._group_id]
    x2 = self._convlayer2(x1_group)  # grouping
    x3 = self._convlayer3(x2)
    x4 = tf.concat([x3, x2], axis=-1)  # csp partial using grouping
    x5 = self._convlayer4(x4)
    x = tf.concat([x1, x5], axis=-1)  # csp connect
    if self._downsample:
      x = self._maxpool(x)
    return x, x5


@tf.keras.utils.register_keras_serializable(package='yolo')
class CSPRoute(tf.keras.layers.Layer):
  """
  Down sampling layer to take the place of down sampleing done in Residual
  networks. This is the first of 2 layers needed to convert any Residual Network
  model to a CSPNet. At the start of a new level change, this CSPRoute layer
  creates a learned identity that will act as a cross stage connection,
  that is used to inform the inputs to the next stage. It is called cross stage
  partial because the number of filters required in every intermitent Residual
  layer is reduced by half. The sister layer will take the partial generated by
  this layer and concatnate it with the output of the final residual layer in
  the stack to create a fully feature level output. This concatnation merges the
  partial blocks of 2 levels as input to the next allowing the gradients of each
  level to be more unique, and reducing the number of parameters required by
  each level by 50% while keeping accuracy consistent.

  Cross Stage Partial networks (CSPNets) were proposed in:
  [1] Chien-Yao Wang, Hong-Yuan Mark Liao, I-Hau Yeh, Yueh-Hua Wu,
        Ping-Yang Chen, Jun-Wei Hsieh
      CSPNet: A New Backbone that can Enhance Learning Capability of CNN.
        arXiv:1911.11929
  Args:
    filters: integer for output depth, or the number of features to learn
    filter_scale: integer dicating (filters//2) or the number of filters in the
      partial feature stack
    downsample: down_sample the input
    activation: string for activation function to use in layer
    kernel_initializer: string to indicate which function to use to
      initialize weights
    bias_initializer: string to indicate which function to use to initialize
      bias
    kernel_regularizer: string to indicate which function to use to regularizer
      weights
    bias_regularizer: string to indicate which function to use to regularizer
      bias
    use_bn: boolean for whether to use batch normalization
    use_sync_bn: boolean for whether sync batch normalization statistics
      of all batch norm layers to the models global statistics
      (across all input batches)
    norm_momentum: float for moment to use for batch normalization
    norm_epsilon: float for batch normalization epsilon
    **kwargs: Keyword Arguments
  """

  def __init__(self,
               filters,
               filter_scale=2,
               subdivisions=1,
               activation='mish',
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               bias_regularizer=None,
               kernel_regularizer=None,
               dilation_rate=1,
               use_bn=True,
               use_sync_bn=False,
               norm_momentum=0.99,
               norm_epsilon=0.001,
               downsample=True,
               leaky_alpha=0.1,
               **kwargs):

    super().__init__(**kwargs)
    # layer params
    self._filters = filters
    self._filter_scale = filter_scale
    self._activation = activation

    # convoultion params
    self._kernel_initializer = kernel_initializer
    self._bias_initializer = bias_initializer
    self._kernel_regularizer = kernel_regularizer
    self._bias_regularizer = bias_regularizer
    self._dilation_rate = dilation_rate
    self._use_bn = use_bn
    self._use_sync_bn = use_sync_bn
    self._norm_momentum = norm_momentum
    self._norm_epsilon = norm_epsilon
    self._downsample = downsample
    self._subdivisions = subdivisions
    self._leaky_alpha = leaky_alpha

  def build(self, input_shape):
    _dark_conv_args = {
        'kernel_initializer': self._kernel_initializer,
        'bias_initializer': self._bias_initializer,
        'bias_regularizer': self._bias_regularizer,
        'use_bn': self._use_bn,
        'use_sync_bn': self._use_sync_bn,
        'norm_momentum': self._norm_momentum,
        'norm_epsilon': self._norm_epsilon,
        'activation': self._activation,
        'kernel_regularizer': self._kernel_regularizer,
        'subdivisions': self._subdivisions,
        'leaky_alpha': self._leaky_alpha,
    }
    if self._downsample:
      if self._dilation_rate > 1:
        dilation_rate = 1
        if self._dilation_rate // 2 > 0:
          dilation_rate = self._dilation_rate // 2
        down_stride = 1
      else:
        dilation_rate = 1
        down_stride = 2

      self._conv1 = ConvBN(
          filters=self._filters,
          kernel_size=(3, 3),
          strides=down_stride,
          dilation_rate=dilation_rate,
          **_dark_conv_args)

    self._conv2 = ConvBN(
        filters=self._filters // self._filter_scale,
        kernel_size=(1, 1),
        strides=(1, 1),
        **_dark_conv_args)

    self._conv3 = ConvBN(
        filters=self._filters // self._filter_scale,
        kernel_size=(1, 1),
        strides=(1, 1),
        **_dark_conv_args)

  def call(self, inputs, training=None):
    if self._downsample:
      inputs = self._conv1(inputs)
    y = self._conv2(inputs)
    x = self._conv3(inputs)
    return (x, y)


@tf.keras.utils.register_keras_serializable(package='yolo')
class CSPConnect(tf.keras.layers.Layer):
  """
  Sister Layer to the CSPRoute layer. Merges the partial feature stacks
  generated by the CSPDownsampling layer, and the finaly output of the
  residual stack. Suggested in the CSPNet paper.
  Cross Stage Partial networks (CSPNets) were proposed in:
  [1] Chien-Yao Wang, Hong-Yuan Mark Liao, I-Hau Yeh, Yueh-Hua Wu,
        Ping-Yang Chen, Jun-Wei Hsieh
      CSPNet: A New Backbone that can Enhance Learning Capability of CNN.
        arXiv:1911.11929
  Args:
    filters: integer for output depth, or the number of features to learn
    filter_scale: integer dicating (filters//2) or the number of filters in the
      partial feature stack
    activation: string for activation function to use in layer
    kernel_initializer: string to indicate which function to use to initialize
      weights
    bias_initializer: string to indicate which function to use to initialize
      bias
    subdivisions: `int` how many subdivision to usein training execution, not
      used in eval or inference
    kernel_regularizer: string to indicate which function to use to regularizer
      weights
    bias_regularizer: string to indicate which function to use to regularizer
      bias
    use_bn: boolean for whether to use batch normalization
    use_sync_bn: boolean for whether sync batch normalization statistics
      of all batch norm layers to the models global
      statistics (across all input batches)
    norm_momentum: float for moment to use for batch normalization
    norm_epsilon: float for batch normalization epsilon
    **kwargs: Keyword Arguments
  """

  def __init__(self,
               filters,
               filter_scale=2,
               subdivisions=1,
               drop_final=False,
               drop_first=False,
               activation='mish',
               kernel_size=(1, 1),
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               bias_regularizer=None,
               kernel_regularizer=None,
               dilation_rate=1,
               use_bn=True,
               use_sync_bn=False,
               norm_momentum=0.99,
               norm_epsilon=0.001,
               leaky_alpha=0.1,
               **kwargs):

    super().__init__(**kwargs)
    # layer params
    self._filters = filters
    self._filter_scale = filter_scale
    self._activation = activation

    # convoultion params
    self._kernel_size = kernel_size
    self._kernel_initializer = kernel_initializer
    self._bias_initializer = bias_initializer
    self._kernel_regularizer = kernel_regularizer
    self._bias_regularizer = bias_regularizer
    self._use_bn = use_bn
    self._use_sync_bn = use_sync_bn
    self._norm_momentum = norm_momentum
    self._norm_epsilon = norm_epsilon
    self._subdivisions = subdivisions
    self._drop_final = drop_final
    self._drop_first = drop_first
    self._leaky_alpha = leaky_alpha

  def build(self, input_shape):
    _dark_conv_args = {
        'kernel_initializer': self._kernel_initializer,
        'bias_initializer': self._bias_initializer,
        'bias_regularizer': self._bias_regularizer,
        'use_bn': self._use_bn,
        'use_sync_bn': self._use_sync_bn,
        'norm_momentum': self._norm_momentum,
        'norm_epsilon': self._norm_epsilon,
        'activation': self._activation,
        'kernel_regularizer': self._kernel_regularizer,
        'subdivisions': self._subdivisions,
        'leaky_alpha': self._leaky_alpha,
    }
    if not self._drop_first:
      self._conv1 = ConvBN(
          filters=self._filters // self._filter_scale,
          kernel_size=self._kernel_size,
          strides=(1, 1),
          **_dark_conv_args)
    self._concat = tf.keras.layers.Concatenate(axis=-1)

    if not self._drop_final:
      self._conv2 = ConvBN(
          filters=self._filters,
          kernel_size=(1, 1),
          strides=(1, 1),
          **_dark_conv_args)
    return

  def call(self, inputs, training=None):
    x_prev, x_csp = inputs
    if not self._drop_first:
      x_prev = self._conv1(x_prev)
    x = self._concat([x_prev, x_csp])

    # skipped if drop final is true
    if not self._drop_final:
      x = self._conv2(x)
    return x


class CSPStack(tf.keras.layers.Layer):
  """
  CSP full stack, combines the route and the connect in case you dont want to
  jsut quickly wrap an existing callable or list of layers to
  make it a cross stage partial. Added for ease of use. you should be able
  to wrap any layer stack with a CSP independent of wether it belongs
  to the Darknet family. if filter_scale = 2, then the blocks in the stack
  passed into the the CSP stack should also have filters = filters/filter_scale
  Cross Stage Partial networks (CSPNets) were proposed in:

  [1] Chien-Yao Wang, Hong-Yuan Mark Liao, I-Hau Yeh, Yueh-Hua Wu,
        Ping-Yang Chen, Jun-Wei Hsieh
      CSPNet: A New Backbone that can Enhance Learning Capability of CNN.
        arXiv:1911.11929
  Args:
    model_to_wrap: callable Model or a list of callable objects that will
      process the output of CSPRoute, and be input into CSPConnect.
      list will be called sequentially.
    subdivisions: `int` how many subdivision to usein training execution, not
      used in eval or inference
    downsample: down_sample the input
    filters: integer for output depth, or the number of features to learn
    filter_scale: integer dicating (filters//2) or the number of filters in the
      partial feature stack
    activation: string for activation function to use in layer
    kernel_initializer: string to indicate which function to use to initialize
      weights
    bias_initializer: string to indicate which function to use to initialize
      bias
    kernel_regularizer: string to indicate which function to use to regularizer
      weights
    bias_regularizer: string to indicate which function to use to regularizer
      bias
    use_bn: boolean for whether to use batch normalization
    use_sync_bn: boolean for whether sync batch normalization statistics
      of all batch norm layers to the models global statistics
      (across all input batches)
    norm_momentum: float for moment to use for batch normalization
    norm_epsilon: float for batch normalization epsilon
    **kwargs: Keyword Arguments
    """

  def __init__(self,
               filters,
               model_to_wrap=None,
               subdivisions=1,
               filter_scale=2,
               activation='mish',
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               bias_regularizer=None,
               kernel_regularizer=None,
               downsample=True,
               use_bn=True,
               use_sync_bn=False,
               norm_momentum=0.99,
               norm_epsilon=0.001,
               **kwargs):

    super().__init__(**kwargs)
    # layer params
    self._filters = filters
    self._filter_scale = filter_scale
    self._activation = activation
    self._downsample = downsample

    # convoultion params
    self._kernel_initializer = kernel_initializer
    self._bias_initializer = bias_initializer
    self._kernel_regularizer = kernel_regularizer
    self._bias_regularizer = bias_regularizer
    self._use_bn = use_bn
    self._use_sync_bn = use_sync_bn
    self._norm_momentum = norm_momentum
    self._norm_epsilon = norm_epsilon
    self._subdivisions = subdivisions

    if model_to_wrap is not None:
      if isinstance(model_to_wrap, Callable):
        self._model_to_wrap = [model_to_wrap]
      elif isinstance(model_to_wrap, List):
        self._model_to_wrap = model_to_wrap
      else:
        raise Exception(
            'the input to the CSPStack must be a list of layers that we can' +
            'iterate through, or \n a callable')
    else:
      self._model_to_wrap = []

  def build(self, input_shape):
    _dark_conv_args = {
        'filters': self._filters,
        'filter_scale': self._filter_scale,
        'activation': self._activation,
        'kernel_initializer': self._kernel_initializer,
        'bias_initializer': self._bias_initializer,
        'bias_regularizer': self._bias_regularizer,
        'use_bn': self._use_bn,
        'use_sync_bn': self._use_sync_bn,
        'norm_momentum': self._norm_momentum,
        'norm_epsilon': self._norm_epsilon,
        'kernel_regularizer': self._kernel_regularizer,
        'subdivisions': self._subdivisions,
    }
    self._route = CSPRoute(downsample=self._downsample, **_dark_conv_args)
    self._connect = CSPConnect(**_dark_conv_args)

  def call(self, inputs, training=None):
    x, x_route = self._route(inputs)
    for layer in self._model_to_wrap:
      x = layer(x)
    x = self._connect([x, x_route])
    return x


@tf.keras.utils.register_keras_serializable(package='yolo')
class PathAggregationBlock(tf.keras.layers.Layer):

  def __init__(self,
               filters=1,
               drop_final=True,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               bias_regularizer=None,
               subdivisions=1,
               kernel_regularizer=None,
               use_bn=True,
               use_sync_bn=False,
               inverted=False,
               norm_momentum=0.99,
               norm_epsilon=0.001,
               activation='leaky',
               leaky_alpha=0.1,
               downsample=False,
               upsample=False,
               upsample_size=2,
               **kwargs):
    """
    Args:
      filters: integer for output depth, or the number of features to learn
      kernel_initializer: string to indicate which function to use to initialize
        weights
      bias_initializer: string to indicate which function to use to initialize
        bias
      subdivisions: `int` how many subdivision to usein training execution, not
        used in eval or inference
      kernel_regularizer: string to indicate which function to use to
        regularizer weights
      bias_regularizer: string to indicate which function to use to regularizer
        bias
      use_bn: boolean for whether to use batch normalization
      use_sync_bn: boolean for whether sync batch normalization statistics
        of all batch norm layers to the models global statistics
        (across all input batches)
      norm_momentum: float for moment to use for batch normalization
      norm_epsilon: float for batch normalization epsilon
      activation: string or None for activation function to use in layer,
        if None activation is replaced by linear
      leaky_alpha: float to use as alpha if activation function is leaky
      downsample: `bool` for whehter to downwample and merge
      upsample: `bool` for whehter to upsample and merge
      upsample_size: `int` how much to upsample in order to match shapes
      **kwargs: Keyword Arguments
    """

    # darkconv params
    self._filters = filters
    self._kernel_initializer = kernel_initializer
    self._bias_initializer = bias_initializer
    self._bias_regularizer = bias_regularizer
    self._kernel_regularizer = kernel_regularizer
    self._use_bn = use_bn
    self._use_sync_bn = use_sync_bn

    # normal params
    self._norm_momentum = norm_momentum
    self._norm_epsilon = norm_epsilon

    # activation params
    self._conv_activation = activation
    self._leaky_alpha = leaky_alpha
    self._downsample = downsample
    self._upsample = upsample
    self._upsample_size = upsample_size
    self._subdivisions = subdivisions
    self._drop_final = drop_final

    #block params
    self._inverted = inverted

    super().__init__(**kwargs)

  def _build_regular(self, input_shape, kwargs):
    if self._downsample:
      self._conv = ConvBN(
          filters=self._filters,
          kernel_size=(3, 3),
          strides=(2, 2),
          padding='same',
          **kwargs)
    else:
      self._conv = ConvBN(
          filters=self._filters,
          kernel_size=(1, 1),
          strides=(1, 1),
          padding='same',
          **kwargs)

    if not self._drop_final:
      self._conv_concat = ConvBN(
          filters=self._filters,
          kernel_size=(1, 1),
          strides=(1, 1),
          padding='same',
          **kwargs)
    return

  def _build_reversed(self, input_shape, kwargs):
    if self._downsample:
      self._conv_prev = ConvBN(
          filters=self._filters,
          kernel_size=(3, 3),
          strides=(2, 2),
          padding='same',
          **kwargs)
    else:
      self._conv_prev = ConvBN(
          filters=self._filters,
          kernel_size=(1, 1),
          strides=(1, 1),
          padding='same',
          **kwargs)

    self._conv_route = ConvBN(
        filters=self._filters,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding='same',
        **kwargs)

    if not self._drop_final:
      self._conv_sync = ConvBN(
          filters=self._filters,
          kernel_size=(1, 1),
          strides=(1, 1),
          padding='same',
          **kwargs)
    return

  def build(self, input_shape):
    _dark_conv_args = {
        'kernel_initializer': self._kernel_initializer,
        'bias_initializer': self._bias_initializer,
        'bias_regularizer': self._bias_regularizer,
        'use_bn': self._use_bn,
        'use_sync_bn': self._use_sync_bn,
        'norm_momentum': self._norm_momentum,
        'norm_epsilon': self._norm_epsilon,
        'activation': self._conv_activation,
        'kernel_regularizer': self._kernel_regularizer,
        'leaky_alpha': self._leaky_alpha,
        'subdivisions': self._subdivisions,
    }

    if self._inverted:
      self._build_reversed(input_shape, _dark_conv_args)
    else:
      self._build_regular(input_shape, _dark_conv_args)

    self._concat = tf.keras.layers.Concatenate()
    super().build(input_shape)

  def _call_regular(self, inputs, training=None):
    inputToConvolve, inputToConcat = inputs
    x_prev = self._conv(inputToConvolve)
    if self._upsample:
      x_prev = spatial_transform_ops.nearest_upsampling(x_prev,
                                                        self._upsample_size)
    x = self._concat([x_prev, inputToConcat])

    # used in csp conversion
    if not self._drop_final:
      x = self._conv_concat(x)
    return x_prev, x

  def _call_reversed(self, inputs, training=None):
    x_route, x_prev = inputs
    x_prev = self._conv_prev(x_prev)
    if self._upsample:
      x_prev = spatial_transform_ops.nearest_upsampling(x_prev,
                                                        self._upsample_size)
    x_route = self._conv_route(x_route)
    x = self._concat([x_route, x_prev])
    if not self._drop_final:
      x = self._conv_sync(x)
    return x_prev, x

  def call(self, inputs, training=None):
    # done this way to prevent confusion in the auto graph
    if self._inverted:
      return self._call_reversed(inputs, training=training)
    else:
      return self._call_regular(inputs, training=training)


@tf.keras.utils.register_keras_serializable(package='yolo')
class SPP(tf.keras.layers.Layer):
  """
    a non-agregated SPP layer that uses Pooling to gain more performance
    """

  def __init__(self, sizes, **kwargs):
    self._sizes = list(reversed(sizes))
    if len(sizes) == 0:
      raise ValueError('More than one maxpool should be specified in SSP block')
    super().__init__(**kwargs)

  def build(self, input_shape):
    maxpools = []
    for size in self._sizes:
      maxpools.append(
          tf.keras.layers.MaxPool2D(
              pool_size=(size, size),
              strides=(1, 1),
              padding='same',
              data_format=None))
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


class SAM(tf.keras.layers.Layer):
  """
  [1] Sanghyun Woo, Jongchan Park, Joon-Young Lee, In So Kweon
  CBAM: Convolutional Block Attention Module. arXiv:1807.06521

  implementation of the Spatial Attention Model (SAM)
  """

  def __init__(self,
               use_pooling=False,
               filter_match=False,
               filters=1,
               kernel_size=(1, 1),
               strides=(1, 1),
               padding='same',
               dilation_rate=(1, 1),
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               subdivisions=1,
               bias_regularizer=None,
               kernel_regularizer=None,
               use_bn=True,
               use_sync_bn=True,
               norm_momentum=0.99,
               norm_epsilon=0.001,
               activation='sigmoid',
               output_activation=None,
               leaky_alpha=0.1,
               **kwargs):

    # use_pooling
    self._use_pooling = use_pooling
    self._filters = filters
    self._output_activation = output_activation
    self._leaky_alpha = leaky_alpha

    self._dark_conv_args = {
        'kernel_size': kernel_size,
        'strides': strides,
        'padding': padding,
        'dilation_rate': dilation_rate,
        'kernel_initializer': kernel_initializer,
        'bias_initializer': bias_initializer,
        'bias_regularizer': bias_regularizer,
        'use_bn': use_bn,
        'use_sync_bn': use_sync_bn,
        'subdivisions': subdivisions,
        'norm_momentum': norm_momentum,
        'norm_epsilon': norm_epsilon,
        'activation': activation,
        'kernel_regularizer': kernel_regularizer,
        'leaky_alpha': leaky_alpha
    }

    super().__init__(**kwargs)

  def build(self, input_shape):
    if self._filters == -1:
      self._filters = input_shape[-1]
    self._conv = ConvBN(filters=self._filters, **self._dark_conv_args)
    if self._output_activation == 'leaky':
      self._activation_fn = tf.keras.layers.LeakyReLU(alpha=self._leaky_alpha)
    elif self._output_activation == 'mish':
      self._activation_fn = lambda x: x * tf.math.tanh(tf.math.softplus(x))
    else:
      self._activation_fn = tf_utils.get_activation(self._output_activation)

  def call(self, inputs, training=None):
    if self._use_pooling:
      depth_max = tf.reduce_max(inputs, axis=-1, keep_dims=True)
      depth_avg = tf.reduce_mean(inputs, axis=-1, keep_dims=True)
      input_maps = tf.concat([depth_avg, depth_max], axis=-1)
    else:
      input_maps = inputs

    attention_mask = self._conv(input_maps)
    return self._activation_fn(inputs * attention_mask)


class CAM(tf.keras.layers.Layer):
  """
  [1] Sanghyun Woo, Jongchan Park, Joon-Young Lee, In So Kweon
  CBAM: Convolutional Block Attention Module. arXiv:1807.06521

  implementation of the Channel Attention Model (CAM)
  """

  def __init__(self,
               reduction_ratio=1.0,
               subdivisions=1,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               bias_regularizer=None,
               kernel_regularizer=None,
               use_bn=False,
               use_sync_bn=False,
               use_bias=False,
               norm_momentum=0.99,
               norm_epsilon=0.001,
               mlp_activation='linear',
               activation='sigmoid',
               leaky_alpha=0.1,
               **kwargs):

    self._reduction_ratio = reduction_ratio

    # use_pooling
    if use_sync_bn:
      self._bn = subnormalization.SubDivSyncBatchNormalization
    else:
      self._bn = subnormalization.SubDivBatchNormalization

    if not use_bn:
      self._bn = Identity
      self._bn_args = {}
    else:
      self._bn_args = {
          'subdivisions': subdivisions,
          'momentum': norm_momentum,
          'epsilon': norm_epsilon,
      }

    self._mlp_args = {
        'use_bias': use_bias,
        'kernel_initializer': kernel_initializer,
        'bias_initializer': bias_initializer,
        'bias_regularizer': bias_regularizer,
        'activation': mlp_activation,
        'kernel_regularizer': kernel_regularizer,
    }

    self._leaky_alpha = leaky_alpha
    self._activation = activation

    super().__init__(**kwargs)

  def build(self, input_shape):
    self._filters = input_shape[-1]

    self._mlp = tf.keras.Sequential([
        tf.keras.layers.Dense(self._filters, **self._mlp_args),
        self._bn(**self._bn_args),
        tf.keras.layers.Dense(
            int(self._filters * self._reduction_ratio), **self._mlp_args),
        self._bn(**self._bn_args),
        tf.keras.layers.Dense(self._filters, **self._mlp_args),
        self._bn(**self._bn_args),
    ])

    if self._activation == 'leaky':
      self._activation_fn = tf.keras.layers.LeakyReLU(alpha=self._leaky_alpha)
    elif self._activation == 'mish':
      self._activation_fn = lambda x: x * tf.math.tanh(tf.math.softplus(x))
    else:
      self._activation_fn = tf_utils.get_activation(self._activation)

    return

  def call(self, inputs, training=None):
    depth_max = self._mlp(tf.reduce_max(inputs, axis=(1, 2)))
    depth_avg = self._mlp(tf.reduce_mean(inputs, axis=(1, 2)))
    channel_mask = self._activation_fn(depth_avg + depth_max)

    channel_mask = tf.expand_dims(channel_mask, axis=1)
    attention_mask = tf.expand_dims(channel_mask, axis=1)

    return inputs * attention_mask


class CBAM(tf.keras.layers.Layer):
  """
  [1] Sanghyun Woo, Jongchan Park, Joon-Young Lee, In So Kweon
  CBAM: Convolutional Block Attention Module. arXiv:1807.06521

  implementation of the Convolution Block Attention Module (CBAM)
  """

  def __init__(self,
               use_pooling=False,
               filters=1,
               reduction_ratio=1.0,
               kernel_size=(1, 1),
               strides=(1, 1),
               padding='same',
               dilation_rate=(1, 1),
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               subdivisions=1,
               bias_regularizer=None,
               kernel_regularizer=None,
               use_bn=True,
               use_sync_bn=False,
               norm_momentum=0.99,
               norm_epsilon=0.001,
               mlp_activation=None,
               activation='sigmoid',
               leaky_alpha=0.1,
               **kwargs):

    # use_pooling

    self._sam_args = {
        'use_pooling': use_pooling,
        'filters': filters,
        'kernel_size': kernel_size,
        'strides': strides,
        'padding': padding,
        'dilation_rate': dilation_rate,
    }

    self._cam_args = {
        'reduction_ratio': reduction_ratio,
        'mlp_activation': mlp_activation
    }

    self._common_args = {
        'kernel_initializer': kernel_initializer,
        'bias_initializer': bias_initializer,
        'bias_regularizer': bias_regularizer,
        'use_bn': use_bn,
        'use_sync_bn': use_sync_bn,
        'subdivisions': subdivisions,
        'norm_momentum': norm_momentum,
        'norm_epsilon': norm_epsilon,
        'activation': activation,
        'kernel_regularizer': kernel_regularizer,
        'leaky_alpha': leaky_alpha
    }

    self._cam_args.update(self._common_args)
    self._sam_args.update(self._common_args)
    super().__init__(**kwargs)

  def build(self, input_shape):
    self._cam = CAM(**self._cam_args)
    self._sam = SAM(**self._sam_args)
    return

  def call(self, inputs, training=None):
    return self._sam(self._cam(inputs))


@tf.keras.utils.register_keras_serializable(package='yolo')
class DarkRouteProcess(tf.keras.layers.Layer):

  def __init__(
      self,
      filters=2,
      repetitions=2,
      insert_spp=False,
      insert_sam=False,
      insert_cbam=False,
      csp_stack=0,
      csp_scale=2,
      subdivisions=1,
      kernel_initializer='glorot_uniform',
      bias_initializer='zeros',
      bias_regularizer=None,
      use_sync_bn=False,
      kernel_regularizer=None,  # default find where is it is stated
      norm_momentum=0.99,
      norm_epsilon=0.001,
      block_invert=False,
      activation='leaky',
      leaky_alpha=0.1,
      spp_keys=None,
      **kwargs):
    """
    process darknet outputs and connect back bone to head more generalizably
    Abstracts repetition of DarkConv objects that is common in YOLO.

    It is used like the following:

    x = ConvBN(1024, (3, 3), (1, 1))(x)
    proc = DarkRouteProcess(filters = 1024,
                            repetitions = 3,
                            insert_spp = False)(x)

    Args:
      filters: the number of filters to be used in all subsequent layers
        filters should be the depth of the tensor input into this layer,
        as no downsampling can be done within this layer object
      repetitions: number of times to repeat the processign nodes
        for tiny: 1 repition, no spp allowed
        for spp: insert_spp = True, and allow for 3+ repetitions
        for regular: insert_spp = False, and allow for 3+ repetitions
      insert_spp: bool if true add the spatial pyramid pooling layer
      kernel_initializer: method to use to initializa kernel weights
      bias_initializer: method to use to initialize the bias of the conv
        layers
      subdivisions: `int` how many subdivision to usein training execution, not
        used in eval or inference
      norm_momentum: batch norm parameter see Tensorflow documentation
      norm_epsilon: batch norm parameter see Tensorflow documentation
      activation: activation function to use in processing
      leaky_alpha: if leaky acitivation function, the alpha to use in
        processing the relu input

    Returns:
      callable tensorflow layer

    Raises:
      None
    """

    super().__init__(**kwargs)
    # darkconv params
    self._filters = filters
    self._use_sync_bn = use_sync_bn
    self._kernel_initializer = kernel_initializer
    self._bias_initializer = bias_initializer
    self._bias_regularizer = bias_regularizer
    self._kernel_regularizer = kernel_regularizer
    self._subdivisions = subdivisions

    # normal params
    self._norm_momentum = norm_momentum
    self._norm_epsilon = norm_epsilon

    # activation params
    self._activation = activation
    self._leaky_alpha = leaky_alpha

    repetitions += (2 * int(insert_spp))
    if repetitions == 1:
      block_invert = True

    self._repetitions = repetitions
    self.layer_list, self.outputs = self._get_base_layers()

    if csp_stack > 0:
      self._csp_scale = csp_scale
      csp_stack += (2 * int(insert_spp))
      self._csp_filters = lambda x: x // csp_scale
      self._convert_csp(self.layer_list, self.outputs, csp_stack)
      block_invert = False

    self._csp_stack = csp_stack

    if block_invert:
      self._conv1_filters = lambda x: x
      self._conv2_filters = lambda x: x // 2
      self._conv1_kernel = (3, 3)
      self._conv2_kernel = (1, 1)
    else:
      self._conv1_filters = lambda x: x // 2
      self._conv2_filters = lambda x: x
      self._conv1_kernel = (1, 1)
      self._conv2_kernel = (3, 3)

    # insert SPP will always add to the total nuber of layer, never replace
    if insert_spp:
      self._spp_keys = spp_keys if spp_keys is not None else [5, 9, 13]
      self.layer_list = self._insert_spp(self.layer_list)

    if repetitions > 1:
      self.outputs[-2] = True

    if insert_sam:
      self.layer_list = self._insert_sam(self.layer_list, self.outputs)
      self._repetitions += 1
    self.outputs[-1] = True

  def _get_base_layers(self):
    layer_list = []
    outputs = []
    for i in range(self._repetitions):
      layers = ["conv1"] * ((i + 1) % 2) + ["conv2"] * (i % 2)
      layer_list.extend(layers)
      outputs = [False] + outputs
    return layer_list, outputs

  def _insert_spp(self, layer_list):
    if len(layer_list) <= 3:
      layer_list[1] = 'spp'
    else:
      layer_list[3] = 'spp'
    return layer_list

  def _convert_csp(self, layer_list, outputs, csp_stack_size):
    layer_list[0] = 'csp_route'
    layer_list.insert(csp_stack_size - 1, 'csp_connect')
    outputs.insert(csp_stack_size - 1, False)
    return layer_list, outputs

  def _insert_sam(self, layer_list, outputs):
    if len(layer_list) >= 2 and layer_list[-2] != 'spp':
      layer_list.insert(-2, 'sam')
      outputs.insert(-1, True)
    else:
      layer_list.insert(-1, 'sam')
      outputs.insert(-1, False)
    return layer_list

  def _conv1(self, filters, kwargs, csp=False):
    if csp:
      filters_ = self._csp_filters
    else:
      filters_ = self._conv1_filters

    x1 = ConvBN(
        filters=filters_(filters),
        kernel_size=self._conv1_kernel,
        strides=(1, 1),
        padding='same',
        use_bn=True,
        **kwargs)
    return x1

  def _conv2(self, filters, kwargs, csp=False):
    if csp:
      filters_ = self._csp_filters
    else:
      filters_ = self._conv2_filters

    x1 = ConvBN(
        filters=filters_(filters),
        kernel_size=self._conv2_kernel,
        strides=(1, 1),
        padding='same',
        use_bn=True,
        **kwargs)
    return x1

  def _csp_route(self, filters, kwargs):
    x1 = CSPRoute(
        filters=filters,
        filter_scale=self._csp_scale,
        downsample=False,
        **kwargs)
    return x1

  def _csp_connect(self, filters, kwargs):
    x1 = CSPConnect(filters=filters, drop_final=True, drop_first=True, **kwargs)
    return x1

  def _spp(self, filters, kwargs):
    x1 = SPP(self._spp_keys)
    return x1

  def _sam(self, filters, kwargs):
    x1 = SAM(filters=-1, use_pooling=False, use_bn=True, **kwargs)
    return x1

  def build(self, input_shape):
    _dark_conv_args = {
        'activation': self._activation,
    }

    _args = {
        'kernel_initializer': self._kernel_initializer,
        'bias_initializer': self._bias_initializer,
        'bias_regularizer': self._bias_regularizer,
        'use_sync_bn': self._use_sync_bn,
        'norm_momentum': self._norm_momentum,
        'norm_epsilon': self._norm_epsilon,
        'kernel_regularizer': self._kernel_regularizer,
        'leaky_alpha': self._leaky_alpha,
        'subdivisions': self._subdivisions,
    }

    _dark_conv_args.update(_args)
    csp = False
    self.layers = []
    for layer in self.layer_list:
      if layer == 'csp_route':
        self.layers.append(self._csp_route(self._filters, _dark_conv_args))
        csp = True
      elif layer == 'csp_connect':
        self.layers.append(self._csp_connect(self._filters, _dark_conv_args))
        csp = False
      elif layer == 'conv1':
        self.layers.append(self._conv1(self._filters, _dark_conv_args, csp=csp))
      elif layer == 'conv2':
        self.layers.append(self._conv2(self._filters, _dark_conv_args, csp=csp))
      elif layer == 'spp':
        self.layers.append(self._spp(self._filters, _dark_conv_args))
      elif layer == 'sam':
        self.layers.append(self._sam(-1, _args))

    self._lim = len(self.layers)
    super().build(input_shape)

  def _call_regular(self, inputs, training=None):
    # check efficiency
    x = inputs
    x_prev = x
    output_prev = True

    for i, (layer, output) in enumerate(zip(self.layers, self.outputs)):
      if output_prev:
        x_prev = x
      x = layer(x)
      output_prev = output
    return x_prev, x

  def _call_csp(self, inputs, training=None):
    # check efficiency
    x = inputs
    x_prev = x
    output_prev = True
    x_route = None

    for i, (layer, output) in enumerate(zip(self.layers, self.outputs)):
      if output_prev:
        x_prev = x
      if i == 0:
        x, x_route = layer(x)
      elif i == self._csp_stack - 1:
        x = layer([x, x_route])
      else:
        x = layer(x)
      output_prev = output
    return x_prev, x

  def call(self, inputs, training=None):
    if self._csp_stack > 0:
      return self._call_csp(inputs, training=training)
    else:
      return self._call_regular(inputs)

import tensorflow as tf
from official.modeling import tf_utils
from official.vision.beta.modeling.layers import nn_blocks as official_nn_blocks


@tf.keras.utils.register_keras_serializable(package='centernet')
class Identity(tf.keras.layers.Layer):

  def __init__(self, **kwargs):
    super().__init__(**kwargs)

  def call(self, input):
    return input


@tf.keras.utils.register_keras_serializable(package='centernet')
class ConvBN(tf.keras.layers.Layer):
  """
    Modified Convolution layer to match that of the DarkNet Library. The Layer is a standards combination of Conv BatchNorm Activation,
    however, the use of bias in the conv is determined by the use of batch normalization.
    Cross Stage Partial networks (CSPNets) were proposed in:
    [1] Chien-Yao Wang, Hong-Yuan Mark Liao, I-Hau Yeh, Yueh-Hua Wu, Ping-Yang Chen, Jun-Wei Hsieh
        CSPNet: A New Backbone that can Enhance Learning Capability of CNN. arXiv:1911.11929
    Args:
      filters: integer for output depth, or the number of features to learn
      kernel_size: integer or tuple for the shape of the weight matrix or kernel to learn
      strides: integer of tuple how much to move the kernel after each kernel use
      padding: string 'valid' or 'same', if same, then pad the image, else do not
      dialtion_rate: tuple to indicate how much to modulate kernel weights and
        how many pixels in a feature map to skip
      kernel_initializer: string to indicate which function to use to initialize weights
      bias_initializer: string to indicate which function to use to initialize bias
      kernel_regularizer: string to indicate which function to use to regularizer weights
      bias_regularizer: string to indicate which function to use to regularizer bias
      use_bn: boolean for whether to use batch normalization
      use_sync_bn: boolean for whether sync batch normalization statistics
        of all batch norm layers to the models global statistics (across all input batches)
      norm_moment: float for moment to use for batch normalization
      norm_epsilon: float for batch normalization epsilon
      activation: string or None for activation function to use in layer,
        if None activation is replaced by linear
      leaky_alpha: float to use as alpha if activation function is leaky
      **kwargs: Keyword Arguments
    """

  def __init__(
      self,
      filters=1,
      kernel_size=(1, 1),
      strides=(1, 1),
      padding='same',
      dilation_rate=(1, 1),
      kernel_initializer='glorot_uniform',
      bias_initializer='zeros',
      bias_regularizer=None,
      kernel_regularizer=None,  # Specify the weight decay as the default will not work.
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

    # batch normalization params
    self._use_bn = use_bn
    self._use_sync_bn = use_sync_bn
    self._norm_moment = norm_momentum
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
    kernel_size = self._kernel_size if isinstance(self._kernel_size,
                                                  int) else self._kernel_size[0]
    dilation_rate = self._dilation_rate if isinstance(
        self._dilation_rate, int) else self._dilation_rate[0]
    if self._padding == 'same' and kernel_size != 1:
      padding = (dilation_rate * (kernel_size - 1))
      left_shift = padding // 2
      self._zeropad = tf.keras.layers.ZeroPadding2D([[left_shift, left_shift],
                                                     [left_shift, left_shift]])
    else:
      self._zeropad = Identity()

    use_bias = not self._use_bn

    # self.conv = tf.keras.layers.Conv2D(
    #     filters=self._filters,
    #     kernel_size=self._kernel_size,
    #     strides=self._strides,
    #     padding= self._padding,# 'valid',
    #     dilation_rate=self._dilation_rate,
    #     use_bias=use_bias,
    #     kernel_initializer=self._kernel_initializer,
    #     bias_initializer=self._bias_initializer,
    #     kernel_regularizer=self._kernel_regularizer,
    #     bias_regularizer=self._bias_regularizer)

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

    if self._use_bn:
      if self._use_sync_bn:
        self.bn = tf.keras.layers.experimental.SyncBatchNormalization(
            momentum=self._norm_moment,
            epsilon=self._norm_epsilon,
            axis=self._bn_axis)
      else:
        self.bn = tf.keras.layers.BatchNormalization(
            momentum=self._norm_moment,
            epsilon=self._norm_epsilon,
            axis=self._bn_axis)
    else:
      self.bn = Identity()

    if self._activation == 'leaky':
      self._activation_fn = tf.keras.layers.LeakyReLU(alpha=self._leaky_alpha)
    elif self._activation == 'mish':
      self._activation_fn = lambda x: x * tf.math.tanh(tf.math.softplus(x))
    else:
      self._activation_fn = tf_utils.get_activation(
          self._activation)  # tf.keras.layers.Activation(self._activation)

  def call(self, x):
    x = self._zeropad(x)
    x = self.conv(x)
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
        'norm_moment': self._norm_moment,
        'norm_epsilon': self._norm_epsilon,
        'activation': self._activation,
        'leaky_alpha': self._leaky_alpha
    }
    layer_config.update(super().get_config())
    return layer_config


@tf.keras.utils.register_keras_serializable(package='centernet')
class HourglassBlock(tf.keras.layers.Layer):
  """
  Hourglass module
  """

  def __init__(self,
               channel_dims_per_stage,
               blocks_per_stage,
               strides=1,
               **kwargs):
    """
    Args:
      channel_dims_per_stage: list of filter sizes for Residual blocks
      blocks_per_stage: list of residual block repetitions per down/upsample
      strides: integer, stride parameter to the Residual block
    """
    self._order = len(channel_dims_per_stage) - 1
    self._channel_dims_per_stage = channel_dims_per_stage
    self._blocks_per_stage = blocks_per_stage
    self._strides = strides

    assert len(channel_dims_per_stage) == len(blocks_per_stage), 'filter ' \
        'size and residual block repetition lists must have the same length'

    self._filters = channel_dims_per_stage[0]
    self._reps = blocks_per_stage[0]

    super().__init__()

  def build(self, input_shape):
    if self._order == 1:
      # base case, residual block repetitions in most inner part of hourglass
      blocks = [
          official_nn_blocks.ResidualBlock(
              filters=self._filters, strides=self._strides, use_projection=True)
          for _ in range(self._reps)
      ]
      self.blocks = tf.keras.Sequential(blocks)

    else:
      # outer hourglass structures
      main_block = [
          official_nn_blocks.ResidualBlock(
              filters=self._filters, strides=self._strides, use_projection=True)
          for _ in range(self._reps)
      ]
      self.main_block = tf.keras.Sequential(main_block, name='Main_Block')

      side_block = [
          official_nn_blocks.ResidualBlock(
              filters=self._filters, strides=self._strides, use_projection=True)
          for _ in range(self._reps)
      ]
      self.side_block = tf.keras.Sequential(side_block, name='Side_Block')

      self.pool = tf.keras.layers.MaxPool2D(pool_size=2)

      # recursively define inner hourglasses
      self.inner_hg = type(self)(
          channel_dims_per_stage=self._channel_dims_per_stage[1:],
          blocks_per_stage=self._blocks_per_stage[1:],
          strides=self._strides)

      # outer hourglass structures
      end_block = [
          official_nn_blocks.ResidualBlock(
              filters=self._filters, strides=self._strides, use_projection=True)
          for _ in range(self._reps)
      ]
      self.end_block = tf.keras.Sequential(end_block, name='End_Block')

      self.upsample_layer = tf.keras.layers.UpSampling2D(
          size=2, interpolation='nearest')

    super().build(input_shape)

  def call(self, x):
    if self._order == 1:
      return self.blocks(x)
    else:
      x_pre_pooled = self.main_block(x)
      x_side = self.side_block(x_pre_pooled)
      x_pooled = self.pool(x_pre_pooled)
      inner_output = self.inner_hg(x_pooled)
      hg_output = self.end_block(inner_output)
      return self.upsample_layer(hg_output) + x_side

  def get_config(self):
    layer_config = {
        'channel_dims_per_stage': self._channel_dims_per_stage,
        'blocks_per_stage': self._blocks_per_stage,
        'strides': self._strides
    }
    layer_config.update(super().get_config())
    return layer_config

class CenterNetDecoderConv(tf.keras.layers.Layer):
  """
  Convolution block for the CenterNet head. This is used to generate
  both the confidence heatmaps and other regressed predictions such as 
  center offsets, object size, etc.
  """
  def __init__(self,
               output_filters: int,
               bias_init : float,
               name: str,
               **kwargs):
    """
    Args:
      output_filters: int, channel depth of layer output
      bias_init: float, value to initialize the bias vector for the final
        convolution layer
      name: string, layer name
    """
    self._output_filters = output_filters
    self._bias_init = bias_init
    super().__init__(name=name, **kwargs)
  
  def build(self, input_shape):
    n_channels = input_shape[-1]

    self.conv1 = tf.keras.layers.Conv2D(filters=n_channels,
      kernel_size=(3, 3), padding='same')

    self.relu = tf.keras.layers.ReLU()

    # Initialize bias to the last Conv2D Layer
    self.conv2 = tf.keras.layers.Conv2D(filters=self._output_filters,
      kernel_size=(1, 1), 
      bias_initializer=tf.constant_initializer(self._bias_init))

  def call(self, x):
    x = self.conv1(x)
    x = self.relu(x)
    x = self.conv2(x)
    return x
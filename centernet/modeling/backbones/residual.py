# Copy and paste from official.vision.beta.modeling.layers.nn_blocks
# because had trouble pip installing some dependencies that are not 
# used for the residual block
import tensorflow as tf
from official.modeling import tf_utils


class ResidualBlock(tf.keras.layers.Layer):
  """A residual block."""

  def __init__(self,
               filters,
               strides,
               use_projection=False,
               se_ratio=None,
               stochastic_depth_drop_rate=None,
               kernel_initializer='VarianceScaling',
               kernel_regularizer=None,
               bias_regularizer=None,
               activation='relu',
               use_sync_bn=False,
               norm_momentum=0.99,
               norm_epsilon=0.001,
               **kwargs):
    """A residual block with BN after convolutions.

    Args:
      filters: `int` number of filters for the first two convolutions. Note that
        the third and final convolution will use 4 times as many filters.
      strides: `int` block stride. If greater than 1, this block will ultimately
        downsample the input.
      use_projection: `bool` for whether this block should use a projection
        shortcut (versus the default identity shortcut). This is usually `True`
        for the first block of a block group, which may change the number of
        filters and the resolution.
      se_ratio: `float` or None. Ratio of the Squeeze-and-Excitation layer.
      stochastic_depth_drop_rate: `float` or None. if not None, drop rate for
        the stochastic depth layer.
      kernel_initializer: kernel_initializer for convolutional layers.
      kernel_regularizer: tf.keras.regularizers.Regularizer object for Conv2D.
                          Default to None.
      bias_regularizer: tf.keras.regularizers.Regularizer object for Conv2d.
                        Default to None.
      activation: `str` name of the activation function.
      use_sync_bn: if True, use synchronized batch normalization.
      norm_momentum: `float` normalization omentum for the moving average.
      norm_epsilon: `float` small float added to variance to avoid dividing by
        zero.
      **kwargs: keyword arguments to be passed.
    """
    super(ResidualBlock, self).__init__(**kwargs)

    self._filters = filters
    self._strides = strides
    self._use_projection = use_projection
    self._se_ratio = se_ratio
    self._use_sync_bn = use_sync_bn
    self._activation = activation
    self._stochastic_depth_drop_rate = stochastic_depth_drop_rate
    self._kernel_initializer = kernel_initializer
    self._norm_momentum = norm_momentum
    self._norm_epsilon = norm_epsilon
    self._kernel_regularizer = kernel_regularizer
    self._bias_regularizer = bias_regularizer

    if use_sync_bn:
      self._norm = tf.keras.layers.experimental.SyncBatchNormalization
    else:
      self._norm = tf.keras.layers.BatchNormalization
    if tf.keras.backend.image_data_format() == 'channels_last':
      self._bn_axis = -1
    else:
      self._bn_axis = 1
    self._activation_fn = tf_utils.get_activation(activation)

  def build(self, input_shape):
    if self._use_projection:
      self._shortcut = tf.keras.layers.Conv2D(
          filters=self._filters,
          kernel_size=1,
          strides=self._strides,
          use_bias=False,
          kernel_initializer=self._kernel_initializer,
          kernel_regularizer=self._kernel_regularizer,
          bias_regularizer=self._bias_regularizer)
      self._norm0 = self._norm(
          axis=self._bn_axis,
          momentum=self._norm_momentum,
          epsilon=self._norm_epsilon)

    self._conv1 = tf.keras.layers.Conv2D(
        filters=self._filters,
        kernel_size=3,
        strides=self._strides,
        padding='same',
        use_bias=False,
        kernel_initializer=self._kernel_initializer,
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer)
    self._norm1 = self._norm(
        axis=self._bn_axis,
        momentum=self._norm_momentum,
        epsilon=self._norm_epsilon)

    self._conv2 = tf.keras.layers.Conv2D(
        filters=self._filters,
        kernel_size=3,
        strides=1,
        padding='same',
        use_bias=False,
        kernel_initializer=self._kernel_initializer,
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer)
    self._norm2 = self._norm(
        axis=self._bn_axis,
        momentum=self._norm_momentum,
        epsilon=self._norm_epsilon)

    if self._se_ratio and self._se_ratio > 0 and self._se_ratio <= 1:
      self._squeeze_excitation = nn_layers.SqueezeExcitation(
          in_filters=self._filters,
          out_filters=self._filters,
          se_ratio=self._se_ratio,
          kernel_initializer=self._kernel_initializer,
          kernel_regularizer=self._kernel_regularizer,
          bias_regularizer=self._bias_regularizer)
    else:
      self._squeeze_excitation = None

    if self._stochastic_depth_drop_rate:
      self._stochastic_depth = nn_layers.StochasticDepth(
          self._stochastic_depth_drop_rate)
    else:
      self._stochastic_depth = None

    super(ResidualBlock, self).build(input_shape)

  def get_config(self):
    config = {
        'filters': self._filters,
        'strides': self._strides,
        'use_projection': self._use_projection,
        'se_ratio': self._se_ratio,
        'stochastic_depth_drop_rate': self._stochastic_depth_drop_rate,
        'kernel_initializer': self._kernel_initializer,
        'kernel_regularizer': self._kernel_regularizer,
        'bias_regularizer': self._bias_regularizer,
        'activation': self._activation,
        'use_sync_bn': self._use_sync_bn,
        'norm_momentum': self._norm_momentum,
        'norm_epsilon': self._norm_epsilon
    }
    base_config = super(ResidualBlock, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def call(self, inputs, training=None):
    shortcut = inputs
    if self._use_projection:
      shortcut = self._shortcut(shortcut)
      shortcut = self._norm0(shortcut)

    x = self._conv1(inputs)
    x = self._norm1(x)
    x = self._activation_fn(x)

    x = self._conv2(x)
    x = self._norm2(x)

    if self._squeeze_excitation:
      x = self._squeeze_excitation(x)

    if self._stochastic_depth:
      x = self._stochastic_depth(x, training=training)

    return self._activation_fn(x + shortcut)
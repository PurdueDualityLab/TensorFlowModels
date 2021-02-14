import tensorflow as tf
from official.modeling import tf_utils
from official.vision.beta.modeling.layers import nn_blocks as official_nn_blocks


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

class CenterNetHeadConv(tf.keras.layers.Layer):
  """
  Convolution block for the CenterNet head. This is used to generate
  both the confidence heatmaps and other regressed predictions such as 
  center offsets, object size, etc.
  """
  def __init__(self,
               output_filters: int,
               name: str,
               **kwargs):
    """
    Args:
      output_filters: int, channel depth of layer output
      name: string, layer name
    """
    self._output_filters = output_filters
    super().__init__(name=name, **kwargs)
  
  def build(self, input_shape):
    n_channels = input_shape[-1]

    self.conv1 = tf.keras.layers.Conv2D(filters=n_channels,
      kernel_size=(3, 3), padding='same')

    self.relu = tf.keras.layers.ReLU()

    self.conv2 = tf.keras.layers.Conv2D(filters=self._output_filters,
    kernel_size=(1, 1), padding='valid')

  def call(self, x):
    x = self.conv1(x)
    x = self.relu(x)
    x = self.conv2(x)
    return x
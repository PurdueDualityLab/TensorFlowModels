import tensorflow as tf
from official.modeling import tf_utils
from official.vision.beta.modeling.layers import nn_blocks as official_nn_blocks

class HourglassBlock(tf.keras.layers.Layer):
  """
  Hourglass module
  """

  def __init__(self, filter_sizes, rep_sizes, strides=1, **kwargs):
    """
    Args:
      filter_sizes: list of filter sizes for Residual blocks
      rep_sizes: list of residual block repetitions per down/upsample
      strides: integer, stride parameter to the Residual block
    """
    self._order = len(filter_sizes) - 1
    self._filter_sizes = filter_sizes
    self._rep_sizes = rep_sizes
    self._strides = strides

    assert len(filter_sizes) == len(rep_sizes), 'filter size and ' \
        'residual block repetition lists must have the same length'

    self._filters = filter_sizes[0]
    self._reps = rep_sizes[0]

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
      if (self._order == 1):
        self.inner_hg = HourglassBlockBase(
          filters=self._filter_sizes[-1],
          reps=self._rep_sizes[-1],
          strides=self._strides)
      else:
        self.inner_hg = type(self)(
          filter_sizes=self._filter_sizes[1:],
          rep_sizes=self._rep_sizes[1:],
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

# Copied from YOLO (yolo/modeling/layers/nn_blocks)
class Identity(tf.keras.layers.Layer):

  def __init__(self, **kwargs):
    super().__init__(**kwargs)

  def call(self, input):
    return input

import tensorflow as tf
from centernet.modeling.layers import nn_blocks

# from official.vision.beta.modeling.layers import official_nn_blocks
from centernet.modeling.backbones import residual as official_nn_blocks


class CenterNetBackbone(tf.keras.Model):
  """
  CenterNet Hourglass backbone
  """

  def __init__(self,
               filter_sizes,
               rep_sizes,
               n_stacks=2,
               pre_layers=None,
               **kwargs):
    """
    Args:
        order: integer, number of downsampling (and subsequent upsampling)
               steps per hourglass module
        filter_sizes: list of filter sizes for Residual blocks
        rep_sizes: list of residual block repetitions per down/upsample
        n_stacks: integer, number of hourglass modules in backbone
        pre_layers: tf.keras layer to process input before stacked hourglasses
    """
    self._n_stacks = n_stacks
    self._pre_layers = pre_layers
    self._filter_sizes = filter_sizes
    self._rep_sizes = rep_sizes

    super().__init__(**kwargs)

  def build(self, input_shape):
    if self._pre_layers is None:
      self._pre_layers = tf.keras.Sequential([
          tf.keras.layers.Conv2D(
              filters=128, kernel_size=7, strides=2, padding='same'),
          tf.keras.layers.BatchNormalization(),
          tf.keras.layers.ReLU(),
          official_nn_blocks.ResidualBlock(
              filters=256, use_projection=True,
              strides=2)  # shape errors happening
      ])

    # Create hourglass stacks
    self.hgs = tf.keras.Sequential([
        nn_blocks.HourglassBlock(
            filter_sizes=self._filter_sizes, rep_sizes=self._rep_sizes)
        for _ in range(self._n_stacks)
    ])

    super().build(input_shape)

  def call(self, x):
    x = self._pre_layers(x)

    # TODO: add intermediate layers
    return self.hgs(x)

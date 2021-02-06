import tensorflow as tf
from centernet.modeling.layers import nn_blocks

from official.vision.beta.modeling.layers import nn_blocks as official_nn_blocks

class CenterNetBackbone(tf.keras.Model):
  """
  CenterNet Hourglass backbone
  """

  def __init__(self,
               filter_sizes,
               rep_sizes,
               n_stacks=2,
               downsample=True,
               **kwargs):
    """
    Args:
        filter_sizes: list of filter sizes for Residual blocks
        rep_sizes: list of residual block repetitions per down/upsample
        n_stacks: integer, number of hourglass modules in backbone
        pre_layers: tf.keras layer to process input before stacked hourglasses
    """
    self._n_stacks = n_stacks
    self._downsample = downsample
    self._filter_sizes = filter_sizes
    self._rep_sizes = rep_sizes
    
    super().__init__(**kwargs)

  def build(self, input_shape):
    # Create prelayers if downsampling input
    if self._downsample:
      self._pre_layers = tf.keras.Sequential([
        nn_blocks.ConvBN(
            filters=128, kernel_size=(7, 7), strides=(2, 2), 
            padding='same', use_bn=True, activation='reLU'),
        official_nn_blocks.ResidualBlock(
            filters=256, use_projection=True,
            strides=2)
      ], name="Prelayers")

    # Create hourglass stacks
    self.hgs = [
        nn_blocks.HourglassBlock(
            filter_sizes=self._filter_sizes, rep_sizes=self._rep_sizes)
        for _ in range(self._n_stacks)
    ]

    # Create some intermediate and postlayers to generate the heatmaps (document and make cleaner later)
    inp_filters = self._filter_sizes[0]
    
    # cnvs 
    self.post_hg_convs = [nn_blocks.ConvBN(
                    filters=inp_filters, kernel_size=(3, 3), strides=(1, 1), 
                    padding='same', use_bn=True, activation='reLU')
                  for _ in range(self._n_stacks)
    ]
    #cnvs_
    self.inter_hg_convs1 = [nn_blocks.ConvBN(
                    filters=256, kernel_size=(1, 1), strides=(1, 1), 
                    padding='same', use_bn=True, activation='linear')
                  for _ in range(self._n_stacks - 1)
    ]
    #inters_
    self.inter_hg_convs2 = [nn_blocks.ConvBN(
                    filters=inp_filters, kernel_size=(1, 1), strides=(1, 1), 
                    padding='same', use_bn=True, activation='linear')
                  for _ in range(self._n_stacks - 1)
    ]
    # inters
    self.res = [official_nn_blocks.ResidualBlock(
                    filters=inp_filters, use_projection=True,
                    strides=2)
                  for _ in range(self._n_stacks - 1)
    ]

    self.relu = tf.keras.layers.ReLU()

    super().build(input_shape)

  def call(self, x):
    x_inter = x
    if self._downsample:
      x_inter = self._pre_layers(x_inter)

    all_heatmaps = []

    for i in range(self._n_stacks):
      hg = self.hgs[i]
      post_conv = self.post_hg_convs[i]

      x_hg = hg(x_inter)
      x_hg = post_conv(x_hg)

      all_heatmaps.append(x_hg)

      if i < self._n_stacks - 1:
        inter_hg_conv1 = self.inter_hg_convs1[i]
        inter_hg_conv2 = self.inter_hg_convs2[i]
        res            = self.res[i]

        x_inter = inter_hg_conv1(x_inter) + inter_hg_conv2(x_hg)
        x_inter = self.relu(x_inter)
        x_inter = res(x_inter)

    return x_hg
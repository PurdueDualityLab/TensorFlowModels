from typing import List

import tensorflow as tf

from centernet.configs import backbones as cfg
from centernet.modeling.layers import nn_blocks
# from official.vision.beta.modeling.layers import \
#     nn_blocks as official_nn_blocks
from utils import register

BATCH_NORM_MOMENTUM = 0.1
BATCH_NORM_EPSILON = 1e-5

@tf.keras.utils.register_keras_serializable(package='centernet')
class Hourglass(tf.keras.Model):
  """
  CenterNet Hourglass backbone
  """

  def __init__(
      self,
      input_channel_dims: int,
      channel_dims_per_stage: List[int],
      blocks_per_stage: List[int],
      num_hourglasses: int,
      initial_downsample: bool = True,
      input_specs=tf.keras.layers.InputSpec(shape=[None, None, None, 3]),
      **kwargs):
    """
    Args:
        channel_dims_per_stage: list of filter sizes for Residual blocks
        blocks_per_stage: list of residual block repetitions per down/upsample
        num_hourglasses: integer, number of hourglass modules in backbone
        pre_layers: tf.keras layer to process input before stacked hourglasses
    """
    # yapf: disable
    input = tf.keras.layers.Input(shape=input_specs.shape[1:], name='input')

    inp_filters = channel_dims_per_stage[0]

    # Create downsampling layers
    if initial_downsample:
      prelayer_kernel_size = 7
      prelayer_strides = 2
    else:
      prelayer_kernel_size = 3
      prelayer_strides = 1

    x_downsampled = nn_blocks.ConvBN(filters=input_channel_dims,
                                     kernel_size=prelayer_kernel_size,
                                     strides=prelayer_strides,
                                     padding='valid',
                                     activation='relu',
                                     use_sync_bn=True,
                                     norm_momentum=BATCH_NORM_MOMENTUM,
                                     norm_epsilon=BATCH_NORM_EPSILON)(input)

    x_downsampled = nn_blocks.CenterNetResidualBlock(
      filters=inp_filters, 
      use_projection=True, 
      strides=prelayer_strides,
      use_sync_bn=True,
      norm_momentum=BATCH_NORM_MOMENTUM, 
      norm_epsilon=BATCH_NORM_EPSILON)(x_downsampled)

    # Used for storing each hourglass heatmap output
    all_heatmaps = []

    for i in range(num_hourglasses):
      # Create hourglass stacks
      x_hg = nn_blocks.HourglassBlock(
          channel_dims_per_stage=channel_dims_per_stage,
          blocks_per_stage=blocks_per_stage,
      )(x_downsampled)

      # cnvs
      x_hg = nn_blocks.ConvBN(
          filters=inp_filters,
          kernel_size=(3, 3),
          strides=(1, 1),
          padding='valid',
          activation='relu',
          use_sync_bn=True,
          norm_momentum=BATCH_NORM_MOMENTUM,
          norm_epsilon=BATCH_NORM_EPSILON
      )(x_hg)

      all_heatmaps.append(x_hg)

      # between hourglasses, we insert intermediate layers
      if i < num_hourglasses - 1:
        # cnvs_
        inter_hg_conv1 = nn_blocks.ConvBN(
            filters=inp_filters,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding='same',
            activation='identity',
            use_sync_bn=True,
            norm_momentum=BATCH_NORM_MOMENTUM,
            norm_epsilon=BATCH_NORM_EPSILON
        )(x_downsampled)

        # inters_
        inter_hg_conv2 = nn_blocks.ConvBN(
            filters=inp_filters,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding='same',
            activation='identity',
            use_sync_bn=True,
            norm_momentum=BATCH_NORM_MOMENTUM,
            norm_epsilon=BATCH_NORM_EPSILON
        )(x_hg)

        x_downsampled = tf.keras.layers.Add()([inter_hg_conv1, inter_hg_conv2])
        x_downsampled = tf.keras.layers.ReLU()(x_downsampled)

        # inters
        x_downsampled = nn_blocks.CenterNetResidualBlock(
            filters=inp_filters, use_projection=False, strides=1,
            norm_momentum=BATCH_NORM_MOMENTUM, norm_epsilon=BATCH_NORM_EPSILON
        )(x_downsampled)
    # yapf: enable

    super().__init__(inputs=input, outputs=all_heatmaps, **kwargs)

    self._input_channel_dims = input_channel_dims
    self._channel_dims_per_stage = channel_dims_per_stage
    self._blocks_per_stage = blocks_per_stage
    self._num_hourglasses = num_hourglasses
    self._initial_downsample = initial_downsample
    self._output_specs = [hm.get_shape() for hm in all_heatmaps]

  def get_config(self):
    layer_config = {
        'input_channel_dims': self._input_channel_dims,
        'channel_dims_per_stage': self._channel_dims_per_stage,
        'blocks_per_stage': self._blocks_per_stage,
        'num_hourglasses': self._num_hourglasses,
        'initial_downsample': self._initial_downsample
    }
    layer_config.update(super().get_config())
    return layer_config

  @property
  def output_specs(self):
    return self._output_specs

# @factory.register_backbone_builder('hourglass')
@register.backbone('hourglass', cfg.Hourglass)
def build_hourglass(
    input_specs: tf.keras.layers.InputSpec,
    model_config,
    l2_regularizer: tf.keras.regularizers.Regularizer = None) -> tf.keras.Model:
  """Builds Hourglass backbone from a config."""
  backbone_type = model_config.backbone.type
  backbone_cfg = model_config.backbone.get()
  assert backbone_type == 'hourglass', (f'Inconsistent backbone type '
                                        f'{backbone_type}')

  return Hourglass(
      input_channel_dims=backbone_cfg.input_channel_dims,
      channel_dims_per_stage=backbone_cfg.channel_dims_per_stage,
      blocks_per_stage=backbone_cfg.blocks_per_stage,
      num_hourglasses=backbone_cfg.num_hourglasses,
      initial_downsample=backbone_cfg.initial_downsample,
      input_specs=input_specs)

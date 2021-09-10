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
"""Contains definitions of Mobilenet Networks."""

from typing import Text, Optional, Dict, Any, Tuple

# Import libraries
import dataclasses
import tensorflow as tf
from official.modeling import hyperparams
from official.modeling import tf_utils
from official.vision.beta.modeling.backbones import factory
from official.vision.beta.modeling.layers import nn_blocks
from official.vision.beta.modeling.layers import nn_layers

layers = tf.keras.layers
regularizers = tf.keras.regularizers


#  pylint: disable=pointless-string-statement


class Conv2DBNBlock(tf.keras.layers.Layer):
  """A convolution block with batch normalization."""

  def __init__(
      self,
      filters: int,
      kernel_size: int = 3,
      strides: int = 1,
      use_bias: bool = False,
      activation: Text = 'relu6',
      kernel_initializer: Text = 'VarianceScaling',
      kernel_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
      bias_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
      use_normalization: bool = True,
      use_sync_bn: bool = False,
      norm_momentum: float = 0.99,
      norm_epsilon: float = 0.001,
      **kwargs):
    """A convolution block with batch normalization.

    Args:
      filters: `int` number of filters for the first two convolutions. Note that
        the third and final convolution will use 4 times as many filters.
      kernel_size: `int` an integer specifying the height and width of the
        2D convolution window.
      strides: `int` block stride. If greater than 1, this block will ultimately
        downsample the input.
      use_bias: if True, use biase in the convolution layer.
      activation: `str` name of the activation function.
      kernel_initializer: kernel_initializer for convolutional layers.
      kernel_regularizer: tf.keras.regularizers.Regularizer object for Conv2D.
                          Default to None.
      bias_regularizer: tf.keras.regularizers.Regularizer object for Conv2d.
                        Default to None.
      use_normalization: if True, use batch normalization.
      use_sync_bn: if True, use synchronized batch normalization.
      norm_momentum: `float` normalization momentum for the moving average.
      norm_epsilon: `float` small float added to variance to avoid dividing by
        zero.
      **kwargs: keyword arguments to be passed.
    """
    super(Conv2DBNBlock, self).__init__(**kwargs)
    self._filters = filters
    self._kernel_size = kernel_size
    self._strides = strides
    self._activation = activation
    self._use_bias = use_bias
    self._kernel_initializer = kernel_initializer
    self._kernel_regularizer = kernel_regularizer
    self._bias_regularizer = bias_regularizer
    self._use_normalization = use_normalization
    self._use_sync_bn = use_sync_bn
    self._norm_momentum = norm_momentum
    self._norm_epsilon = norm_epsilon

    if use_sync_bn:
      self._norm = tf.keras.layers.experimental.SyncBatchNormalization
    else:
      self._norm = tf.keras.layers.BatchNormalization
    if tf.keras.backend.image_data_format() == 'channels_last':
      self._bn_axis = -1
    else:
      self._bn_axis = 1
    self._activation_fn = tf_utils.get_activation(activation)

  def get_config(self):
    config = {
        'filters': self._filters,
        'strides': self._strides,
        'kernel_size': self._kernel_size,
        'use_bias': self._use_bias,
        'kernel_initializer': self._kernel_initializer,
        'kernel_regularizer': self._kernel_regularizer,
        'bias_regularizer': self._bias_regularizer,
        'activation': self._activation,
        'use_sync_bn': self._use_sync_bn,
        'use_normalization': self._use_normalization,
        'norm_momentum': self._norm_momentum,
        'norm_epsilon': self._norm_epsilon
    }
    base_config = super(Conv2DBNBlock, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def build(self, input_shape):
    self._conv0 = tf.keras.layers.Conv2D(
        filters=self._filters,
        kernel_size=self._kernel_size,
        strides=self._strides,
        padding='same',
        use_bias=self._use_bias,
        kernel_initializer=self._kernel_initializer,
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer)
    if self._use_normalization:
      self._norm0 = self._norm(
          axis=self._bn_axis,
          momentum=self._norm_momentum,
          epsilon=self._norm_epsilon)

    super(Conv2DBNBlock, self).build(input_shape)

  def call(self, inputs, training=None):
    x = self._conv0(inputs)
    if self._use_normalization:
      x = self._norm0(x)
    return self._activation_fn(x)

"""
Architecture: https://arxiv.org/abs/1704.04861.

"MobileNets: Efficient Convolutional Neural Networks for Mobile Vision
Applications" Andrew G. Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko,
Weijun Wang, Tobias Weyand, Marco Andreetto, Hartwig Adam
"""
MNV1_BLOCK_SPECS = {
    'spec_name': 'MobileNetV1',
    'block_spec_schema': ['block_fn', 'kernel_size', 'strides', 'filters'],
    'block_specs': [
        ('convbn', 3, 2, 32),
        ('depsepconv', 3, 1, 64),
        ('depsepconv', 3, 2, 128),
        ('depsepconv', 3, 1, 128),
        ('depsepconv', 3, 2, 256),
        ('depsepconv', 3, 1, 256),
        ('depsepconv', 3, 2, 512),
        ('depsepconv', 3, 1, 512),
        ('depsepconv', 3, 1, 512),
        ('depsepconv', 3, 1, 512),
        ('depsepconv', 3, 1, 512),
        ('depsepconv', 3, 1, 512),
        ('depsepconv', 3, 2, 1024),
        ('depsepconv', 3, 1, 1024),
    ]
}

"""
Architecture: https://arxiv.org/abs/1801.04381

"MobileNetV2: Inverted Residuals and Linear Bottlenecks"
Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen
"""
MNV2_BLOCK_SPECS = {
    'spec_name': 'MobileNetV2',
    'block_spec_schema': ['block_fn', 'kernel_size', 'strides', 'filters',
                          'expand_ratio'],
    'block_specs': [
        ('convbn', 3, 2, 32, None),
        ('invertedbottleneck', 3, 1, 16, 1.),
        ('invertedbottleneck', 3, 2, 24, 6.),
        ('invertedbottleneck', 3, 1, 24, 6.),
        ('invertedbottleneck', 3, 2, 32, 6.),
        ('invertedbottleneck', 3, 1, 32, 6.),
        ('invertedbottleneck', 3, 1, 32, 6.),
        ('invertedbottleneck', 3, 2, 64, 6.),
        ('invertedbottleneck', 3, 1, 64, 6.),
        ('invertedbottleneck', 3, 1, 64, 6.),
        ('invertedbottleneck', 3, 1, 64, 6.),
        ('invertedbottleneck', 3, 1, 96, 6.),
        ('invertedbottleneck', 3, 1, 96, 6.),
        ('invertedbottleneck', 3, 1, 96, 6.),
        ('invertedbottleneck', 3, 2, 160, 6.),
        ('invertedbottleneck', 3, 1, 160, 6.),
        ('invertedbottleneck', 3, 1, 160, 6.),
        ('invertedbottleneck', 3, 1, 320, 6.),
        ('convbn', 1, 1, 1280, None),
    ]
}

"""
Architecture: https://arxiv.org/abs/1905.02244

"Searching for MobileNetV3"
Andrew Howard, Mark Sandler, Grace Chu, Liang-Chieh Chen, Bo Chen, Mingxing Tan,
Weijun Wang, Yukun Zhu, Ruoming Pang, Vijay Vasudevan, Quoc V. Le, Hartwig Adam
"""
MNV3Large_BLOCK_SPECS = {
    'spec_name': 'MobileNetV3Large',
    'block_spec_schema': ['block_fn', 'kernel_size', 'strides', 'filters',
                          'activation', 'se_ratio', 'expand_ratio',
                          'use_normalization', 'use_bias'],
    'block_specs': [
        ('convbn', 3, 2, 16, 'hard_swish', None, None, True, False),
        ('invertedbottleneck', 3, 1, 16, 'relu', None, 1., None, False),
        ('invertedbottleneck', 3, 2, 24, 'relu', None, 4., None, False),
        ('invertedbottleneck', 3, 1, 24, 'relu', None, 3., None, False),
        ('invertedbottleneck', 5, 2, 40, 'relu', 0.25, 3., None, False),
        ('invertedbottleneck', 5, 1, 40, 'relu', 0.25, 3., None, False),
        ('invertedbottleneck', 5, 1, 40, 'relu', 0.25, 3., None, False),
        ('invertedbottleneck', 3, 2, 80, 'hard_swish', None, 6., None, False),
        ('invertedbottleneck', 3, 1, 80, 'hard_swish', None, 2.5, None, False),
        ('invertedbottleneck', 3, 1, 80, 'hard_swish', None, 2.3, None, False),
        ('invertedbottleneck', 3, 1, 80, 'hard_swish', None, 2.3, None, False),
        ('invertedbottleneck', 3, 1, 112, 'hard_swish', 0.25, 6., None, False),
        ('invertedbottleneck', 3, 1, 112, 'hard_swish', 0.25, 6., None, False),
        ('invertedbottleneck', 5, 2, 160, 'hard_swish', 0.25, 6., None, False),
        ('invertedbottleneck', 5, 1, 160, 'hard_swish', 0.25, 6., None, False),
        ('invertedbottleneck', 5, 1, 160, 'hard_swish', 0.25, 6., None, False),
        ('convbn', 1, 1, 960, 'hard_swish', None, None, True, False),
        ('gpooling', None, None, None, None, None, None, None, None),
        ('convbn', 1, 1, 1280, 'hard_swish', None, None, False, True),
    ]
}

MNV3Small_BLOCK_SPECS = {
    'spec_name': 'MobileNetV3Small',
    'block_spec_schema': ['block_fn', 'kernel_size', 'strides', 'filters',
                          'activation', 'se_ratio', 'expand_ratio',
                          'use_normalization', 'use_bias'],
    'block_specs': [
        ('convbn', 3, 2, 16, 'hard_swish', None, None, True, False),
        ('invertedbottleneck', 3, 2, 16, 'relu', 0.25, 1, None, False),
        ('invertedbottleneck', 3, 2, 24, 'relu', None, 72. / 16, None, False),
        ('invertedbottleneck', 3, 1, 24, 'relu', None, 88. / 24, None, False),
        ('invertedbottleneck', 5, 2, 40, 'hard_swish', 0.25, 4., None, False),
        ('invertedbottleneck', 5, 1, 40, 'hard_swish', 0.25, 6., None, False),
        ('invertedbottleneck', 5, 1, 40, 'hard_swish', 0.25, 6., None, False),
        ('invertedbottleneck', 5, 1, 48, 'hard_swish', 0.25, 3., None, False),
        ('invertedbottleneck', 5, 1, 48, 'hard_swish', 0.25, 3., None, False),
        ('invertedbottleneck', 5, 2, 96, 'hard_swish', 0.25, 6., None, False),
        ('invertedbottleneck', 5, 1, 96, 'hard_swish', 0.25, 6., None, False),
        ('invertedbottleneck', 5, 1, 96, 'hard_swish', 0.25, 6., None, False),
        ('convbn', 1, 1, 576, 'hard_swish', None, None, True, False),
        ('gpooling', None, None, None, None, None, None, None, None),
        ('convbn', 1, 1, 1024, 'hard_swish', None, None, False, True),
    ]
}

"""
The EdgeTPU version is taken from
github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet_v3.py
"""
MNV3EdgeTPU_BLOCK_SPECS = {
    'spec_name': 'MobileNetV3EdgeTPU',
    'block_spec_schema': ['block_fn', 'kernel_size', 'strides', 'filters',
                          'activation', 'se_ratio', 'expand_ratio',
                          'use_residual', 'use_depthwise'],
    'block_specs': [
        ('convbn', 3, 2, 32, 'relu', None, None, None, None),
        ('invertedbottleneck', 3, 1, 16, 'relu', None, 1., True, False),
        ('invertedbottleneck', 3, 2, 32, 'relu', None, 8., True, False),
        ('invertedbottleneck', 3, 1, 32, 'relu', None, 4., True, False),
        ('invertedbottleneck', 3, 1, 32, 'relu', None, 4., True, False),
        ('invertedbottleneck', 3, 1, 32, 'relu', None, 4., True, False),
        ('invertedbottleneck', 3, 2, 48, 'relu', None, 8., True, False),
        ('invertedbottleneck', 3, 1, 48, 'relu', None, 4., True, False),
        ('invertedbottleneck', 3, 1, 48, 'relu', None, 4., True, False),
        ('invertedbottleneck', 3, 1, 48, 'relu', None, 4., True, False),
        ('invertedbottleneck', 3, 2, 96, 'relu', None, 8., True, True),
        ('invertedbottleneck', 3, 1, 96, 'relu', None, 4., True, True),
        ('invertedbottleneck', 3, 1, 96, 'relu', None, 4., True, True),
        ('invertedbottleneck', 3, 1, 96, 'relu', None, 4., True, True),
        ('invertedbottleneck', 3, 1, 96, 'relu', None, 8., False, True),
        ('invertedbottleneck', 3, 1, 96, 'relu', None, 4., True, True),
        ('invertedbottleneck', 3, 1, 96, 'relu', None, 4., True, True),
        ('invertedbottleneck', 3, 1, 96, 'relu', None, 4., True, True),
        ('invertedbottleneck', 5, 2, 160, 'relu', None, 8., True, True),
        ('invertedbottleneck', 5, 1, 160, 'relu', None, 4., True, True),
        ('invertedbottleneck', 5, 1, 160, 'relu', None, 4., True, True),
        ('invertedbottleneck', 5, 1, 160, 'relu', None, 4., True, True),
        ('invertedbottleneck', 3, 1, 192, 'relu', None, 8., True, True),
        ('convbn', 1, 1, 1280, 'relu', None, None, None, None),
    ]
}

"""
Architecture: https://arxiv.org/pdf/2008.08178.pdf

"Discovering Multi-Hardware Mobile Models via Architecture Search"
Grace Chu, Okan Arikan, Gabriel Bender, Weijun Wang,
Achille Brighton, Pieter-Jan Kindermans, Hanxiao Liu,
Berkin Akin, Suyog Gupta, and Andrew Howard
"""
MNMultiMAX_BLOCK_SPECS = {
    'spec_name': 'MobileNetMultiMAX',
    'block_spec_schema': ['block_fn', 'kernel_size', 'strides', 'filters',
                          'activation', 'expand_ratio',
                          'use_normalization', 'use_bias'],
    'block_specs': [
        ('convbn', 3, 2, 32, 'relu', None, True, False),
        ('invertedbottleneck', 3, 2, 32, 'relu', 3., None, False),
        ('invertedbottleneck', 5, 2, 64, 'relu', 6., None, False),
        ('invertedbottleneck', 3, 1, 64, 'relu', 2., None, False),
        ('invertedbottleneck', 3, 1, 64, 'relu', 2., None, False),
        ('invertedbottleneck', 5, 2, 128, 'relu', 6., None, False),
        ('invertedbottleneck', 3, 1, 128, 'relu', 4., None, False),
        ('invertedbottleneck', 3, 1, 128, 'relu', 3., None, False),
        ('invertedbottleneck', 3, 1, 128, 'relu', 3., None, False),
        ('invertedbottleneck', 3, 1, 128, 'relu', 6., None, False),
        ('invertedbottleneck', 3, 1, 128, 'relu', 3., None, False),
        ('invertedbottleneck', 3, 2, 160, 'relu', 6., None, False),
        ('invertedbottleneck', 5, 1, 160, 'relu', 4., None, False),
        ('invertedbottleneck', 3, 1, 160, 'relu', 5., None, False),
        ('invertedbottleneck', 5, 1, 160, 'relu', 4., None, False),
        ('convbn', 1, 1, 960, 'relu', None, True, False),
        ('gpooling', None, None, None, None, None, None, None),
        ('convbn', 1, 1, 1280, 'relu', None, False, True),
    ]
}

MNMultiAVG_BLOCK_SPECS = {
    'spec_name': 'MobileNetMultiAVG',
    'block_spec_schema': ['block_fn', 'kernel_size', 'strides', 'filters',
                          'activation', 'expand_ratio',
                          'use_normalization', 'use_bias'],
    'block_specs': [
        ('convbn', 3, 2, 32, 'relu', None, True, False),
        ('invertedbottleneck', 3, 2, 32, 'relu', 3., None, False),
        ('invertedbottleneck', 3, 1, 32, 'relu', 2., None, False),
        ('invertedbottleneck', 5, 2, 64, 'relu', 5., None, False),
        ('invertedbottleneck', 3, 1, 64, 'relu', 3., None, False),
        ('invertedbottleneck', 3, 1, 64, 'relu', 2., None, False),
        ('invertedbottleneck', 3, 1, 64, 'relu', 3., None, False),
        ('invertedbottleneck', 5, 2, 128, 'relu', 6., None, False),
        ('invertedbottleneck', 3, 1, 128, 'relu', 3., None, False),
        ('invertedbottleneck', 3, 1, 128, 'relu', 3., None, False),
        ('invertedbottleneck', 3, 1, 128, 'relu', 3., None, False),
        ('invertedbottleneck', 3, 1, 160, 'relu', 6., None, False),
        ('invertedbottleneck', 3, 1, 160, 'relu', 4., None, False),
        ('invertedbottleneck', 3, 2, 192, 'relu', 6., None, False),
        ('invertedbottleneck', 5, 1, 192, 'relu', 4., None, False),
        ('invertedbottleneck', 5, 1, 192, 'relu', 4., None, False),
        ('invertedbottleneck', 5, 1, 192, 'relu', 4., None, False),
        ('convbn', 1, 1, 960, 'relu', None, True, False),
        ('gpooling', None, None, None, None, None, None, None),
        ('convbn', 1, 1, 1280, 'relu', None, False, True),
    ]
}

SUPPORTED_SPECS_MAP = {
    'MobileNetV1': MNV1_BLOCK_SPECS,
    'MobileNetV2': MNV2_BLOCK_SPECS,
    'MobileNetV3Large': MNV3Large_BLOCK_SPECS,
    'MobileNetV3Small': MNV3Small_BLOCK_SPECS,
    'MobileNetV3EdgeTPU': MNV3EdgeTPU_BLOCK_SPECS,
    'MobileNetMultiMAX': MNMultiMAX_BLOCK_SPECS,
    'MobileNetMultiAVG': MNMultiAVG_BLOCK_SPECS,
}


@dataclasses.dataclass
class BlockSpec(hyperparams.Config):
  """A container class that specifies the block configuration for MobileNet."""

  block_fn: Text = 'convbn'
  kernel_size: int = 3
  strides: int = 1
  filters: int = 32
  use_bias: bool = False
  use_normalization: bool = True
  activation: Text = 'relu6'
  # used for block type InvertedResConv
  expand_ratio: Optional[float] = 6.
  # used for block type InvertedResConv with SE
  se_ratio: Optional[float] = None
  use_depthwise: bool = True
  use_residual: bool = True


def block_spec_decoder(specs: Dict[Any, Any],
                       filter_size_scale: float,
                       # set to 1 for mobilenetv1
                       divisible_by: int = 8,
                       finegrain_classification_mode: bool = True):
  """Decode specs for a block.

  Args:
    specs: `dict` specification of block specs of a mobilenet version.
    filter_size_scale: `float` multiplier for the filter size
      for all convolution ops. The value must be greater than zero. Typical
      usage will be to set this value in (0, 1) to reduce the number of
      parameters or computation cost of the model.
    divisible_by: `int` ensures all inner dimensions are divisible by
      this number.
    finegrain_classification_mode: if True, the model
      will keep the last layer large even for small multipliers. Following
      https://arxiv.org/abs/1801.04381

  Returns:
    List[BlockSpec]` defines structure of the base network.
  """

  spec_name = specs['spec_name']
  block_spec_schema = specs['block_spec_schema']
  block_specs = specs['block_specs']

  if not block_specs:
    raise ValueError(
        'The block spec cannot be empty for {} !'.format(spec_name))

  if len(block_specs[0]) != len(block_spec_schema):
    raise ValueError('The block spec values {} do not match with '
                     'the schema {}'.format(block_specs[0], block_spec_schema))

  decoded_specs = []

  for s in block_specs:
    kw_s = dict(zip(block_spec_schema, s))
    decoded_specs.append(BlockSpec(**kw_s))

  # This adjustment applies to V2 and V3
  if (spec_name != 'MobileNetV1'
      and finegrain_classification_mode
      and filter_size_scale < 1.0):
    decoded_specs[-1].filters /= filter_size_scale

  for ds in decoded_specs:
    if ds.filters:
      ds.filters = nn_layers.round_filters(filters=ds.filters,
                                           multiplier=filter_size_scale,
                                           divisor=divisible_by,
                                           min_depth=8)

  return decoded_specs


@tf.keras.utils.register_keras_serializable(package='Vision')
class MobileNet(tf.keras.Model):
  """Class to build MobileNet family model."""

  def __init__(self,
               model_id: Text = 'MobileNetV2',
               filter_size_scale: float = 1.0,
               input_specs: layers.InputSpec = layers.InputSpec(
                   shape=[None, None, None, 3]),
               # The followings are for hyper-parameter tuning
               norm_momentum: float = 0.99,
               norm_epsilon: float = 0.001,
               kernel_initializer: Text = 'VarianceScaling',
               kernel_regularizer: Optional[regularizers.Regularizer] = None,
               bias_regularizer: Optional[regularizers.Regularizer] = None,
               # The followings should be kept the same most of the times
               output_stride: int = None,
               min_depth: int = 8,
               # divisible is not used in MobileNetV1
               divisible_by: int = 8,
               stochastic_depth_drop_rate: float = 0.0,
               regularize_depthwise: bool = False,
               use_sync_bn: bool = False,
               # finegrain is not used in MobileNetV1
               finegrain_classification_mode: bool = True,
               **kwargs):
    """MobileNet initializer.

    Args:
      model_id: `str` version of MobileNet. The supported values are
       'MobileNetV1', 'MobileNetV2', 'MobileNetV3Large', 'MobileNetV3Small',
        and 'MobileNetV3EdgeTPU'.
      filter_size_scale: `float` multiplier for the filters (number of channels)
        for all convolution ops. The value must be greater than zero. Typical
        usage will be to set this value in (0, 1) to reduce the number of
        parameters or computation cost of the model.
      input_specs: `tf.keras.layers.InputSpec` specs of the input tensor.
      norm_momentum: `float` normalization omentum for the moving average.
      norm_epsilon: `float` small float added to variance to avoid dividing by
        zero.
      kernel_initializer: `str` kernel_initializer for convolutional layers.
      kernel_regularizer: tf.keras.regularizers.Regularizer object for Conv2D.
        Default to None.
      bias_regularizer: tf.keras.regularizers.Regularizer object for Conv2d.
        Default to None.
      output_stride: `int` specifies the requested ratio of input to output
        spatial resolution. If not None, then we invoke atrous convolution
        if necessary to prevent the network from reducing the spatial resolution
        of activation maps. Allowed values are 8 (accurate fully convolutional
        mode), 16 (fast fully convolutional mode), 32 (classification mode).
      min_depth: `int` minimum depth (number of channels) for all conv ops.
        Enforced when filter_size_scale < 1, and not an active constraint when
        filter_size_scale >= 1.
      divisible_by: `int` ensures all inner dimensions are divisible by
        this number.
      stochastic_depth_drop_rate: `float` drop rate for drop connect layer.
      regularize_depthwise: if Ture, apply regularization on depthwise.
      use_sync_bn: if True, use synchronized batch normalization.
      finegrain_classification_mode: if True, the model
        will keep the last layer large even for small multipliers. Following
        https://arxiv.org/abs/1801.04381
      **kwargs: keyword arguments to be passed.
    """
    if model_id not in SUPPORTED_SPECS_MAP:
      raise ValueError('The MobileNet version {} '
                       'is not supported'.format(model_id))

    if filter_size_scale <= 0:
      raise ValueError('filter_size_scale is not greater than zero.')

    if output_stride is not None:
      if model_id == 'MobileNetV1':
        if output_stride not in [8, 16, 32]:
          raise ValueError('Only allowed output_stride values are 8, 16, 32.')
      else:
        if output_stride == 0 or (output_stride > 1 and output_stride % 2):
          raise ValueError('Output stride must be None, 1 or a multiple of 2.')

    self._model_id = model_id
    self._input_specs = input_specs
    self._filter_size_scale = filter_size_scale
    self._min_depth = min_depth
    self._output_stride = output_stride
    self._divisible_by = divisible_by
    self._stochastic_depth_drop_rate = stochastic_depth_drop_rate
    self._regularize_depthwise = regularize_depthwise
    self._kernel_initializer = kernel_initializer
    self._kernel_regularizer = kernel_regularizer
    self._bias_regularizer = bias_regularizer
    self._use_sync_bn = use_sync_bn
    self._norm_momentum = norm_momentum
    self._norm_epsilon = norm_epsilon
    self._finegrain_classification_mode = finegrain_classification_mode

    inputs = tf.keras.Input(shape=input_specs.shape[1:])

    block_specs = SUPPORTED_SPECS_MAP.get(model_id)
    self._decoded_specs = block_spec_decoder(
        specs=block_specs,
        filter_size_scale=self._filter_size_scale,
        divisible_by=self._get_divisible_by(),
        finegrain_classification_mode=self._finegrain_classification_mode)

    x, endpoints = self._mobilenet_base(inputs=inputs)

    endpoints[max(endpoints.keys()) + 1] = x
    self._output_specs = {l: endpoints[l].get_shape() for l in endpoints}

    super(MobileNet, self).__init__(
        inputs=inputs, outputs=endpoints, **kwargs)

  def _get_divisible_by(self):
    if self._model_id == 'MobileNetV1':
      return 1
    else:
      return self._divisible_by

  def _mobilenet_base(self,
                      inputs: tf.Tensor
                      ) -> Tuple[tf.Tensor, Dict[int, tf.Tensor]]:
    """Build the base MobileNet architecture.

    Args:
      inputs: Input tensor of shape [batch_size, height, width, channels].

    Returns:
      A tuple of output Tensor and dictionary that collects endpoints.
    """

    input_shape = inputs.get_shape().as_list()
    if len(input_shape) != 4:
      raise ValueError('Expected rank 4 input, was: %d' % len(input_shape))

    # The current_stride variable keeps track of the output stride of the
    # activations, i.e., the running product of convolution strides up to the
    # current network layer. This allows us to invoke atrous convolution
    # whenever applying the next convolution would result in the activations
    # having output stride larger than the target output_stride.
    current_stride = 1

    # The atrous convolution rate parameter.
    rate = 1

    net = inputs
    endpoints = {}
    endpoint_level = 1
    for i, block_def in enumerate(self._decoded_specs):
      block_name = 'block_group_{}_{}'.format(block_def.block_fn, i)
      # A small catch for gpooling block with None strides
      if not block_def.strides:
        block_def.strides = 1
      if self._output_stride is not None \
          and current_stride == self._output_stride:
        # If we have reached the target output_stride, then we need to employ
        # atrous convolution with stride=1 and multiply the atrous rate by the
        # current unit's stride for use in subsequent layers.
        layer_stride = 1
        layer_rate = rate
        rate *= block_def.strides
      else:
        layer_stride = block_def.strides
        layer_rate = 1
        current_stride *= block_def.strides

      if block_def.block_fn == 'convbn':

        net = Conv2DBNBlock(
            filters=block_def.filters,
            kernel_size=block_def.kernel_size,
            strides=block_def.strides,
            activation=block_def.activation,
            use_bias=block_def.use_bias,
            use_normalization=block_def.use_normalization,
            kernel_initializer=self._kernel_initializer,
            kernel_regularizer=self._kernel_regularizer,
            bias_regularizer=self._bias_regularizer,
            use_sync_bn=self._use_sync_bn,
            norm_momentum=self._norm_momentum,
            norm_epsilon=self._norm_epsilon
        )(net)

      elif block_def.block_fn == 'depsepconv':
        net = nn_blocks.DepthwiseSeparableConvBlock(
            filters=block_def.filters,
            kernel_size=block_def.kernel_size,
            strides=block_def.strides,
            activation=block_def.activation,
            dilation_rate=layer_rate,
            regularize_depthwise=self._regularize_depthwise,
            kernel_initializer=self._kernel_initializer,
            kernel_regularizer=self._kernel_regularizer,
            use_sync_bn=self._use_sync_bn,
            norm_momentum=self._norm_momentum,
            norm_epsilon=self._norm_epsilon,
        )(net)

      elif block_def.block_fn == 'invertedbottleneck':
        use_rate = rate
        if layer_rate > 1 and block_def.kernel_size != 1:
          # We will apply atrous rate in the following cases:
          # 1) When kernel_size is not in params, the operation then uses
          #   default kernel size 3x3.
          # 2) When kernel_size is in params, and if the kernel_size is not
          #   equal to (1, 1) (there is no need to apply atrous convolution to
          #   any 1x1 convolution).
          use_rate = layer_rate
        in_filters = net.shape.as_list()[-1]
        net = nn_blocks.InvertedBottleneckBlock(
            in_filters=in_filters,
            out_filters=block_def.filters,
            kernel_size=block_def.kernel_size,
            strides=layer_stride,
            expand_ratio=block_def.expand_ratio,
            se_ratio=block_def.se_ratio,
            expand_se_in_filters=True,
            se_gating_activation='hard_sigmoid',
            activation=block_def.activation,
            use_depthwise=block_def.use_depthwise,
            use_residual=block_def.use_residual,
            dilation_rate=use_rate,
            regularize_depthwise=self._regularize_depthwise,
            kernel_initializer=self._kernel_initializer,
            kernel_regularizer=self._kernel_regularizer,
            bias_regularizer=self._bias_regularizer,
            use_sync_bn=self._use_sync_bn,
            norm_momentum=self._norm_momentum,
            norm_epsilon=self._norm_epsilon,
            stochastic_depth_drop_rate=self._stochastic_depth_drop_rate,
            divisible_by=self._get_divisible_by()
        )(net)

      elif block_def.block_fn == 'gpooling':
        net = layers.GlobalAveragePooling2D()(net)
        net = layers.Reshape((1, 1, net.shape[1]))(net)

      else:
        raise ValueError('Unknown block type {} for layer {}'.format(
            block_def.block_fn, i))

      endpoints[endpoint_level] = net
      endpoint_level += 1
      net = tf.identity(net, name=block_name)
    return net, endpoints

  def get_config(self):
    config_dict = {
        'model_id': self._model_id,
        'filter_size_scale': self._filter_size_scale,
        'min_depth': self._min_depth,
        'output_stride': self._output_stride,
        'divisible_by': self._divisible_by,
        'stochastic_depth_drop_rate': self._stochastic_depth_drop_rate,
        'regularize_depthwise': self._regularize_depthwise,
        'kernel_initializer': self._kernel_initializer,
        'kernel_regularizer': self._kernel_regularizer,
        'bias_regularizer': self._bias_regularizer,
        'use_sync_bn': self._use_sync_bn,
        'norm_momentum': self._norm_momentum,
        'norm_epsilon': self._norm_epsilon,
        'finegrain_classification_mode': self._finegrain_classification_mode,
    }
    return config_dict

  @classmethod
  def from_config(cls, config, custom_objects=None):
    return cls(**config)

  @property
  def output_specs(self):
    """A dict of {level: TensorShape} pairs for the model output."""
    return self._output_specs


@factory.register_backbone_builder('mobilenet')
def build_mobilenet(
    input_specs: tf.keras.layers.InputSpec,
    model_config,
    l2_regularizer: tf.keras.regularizers.Regularizer = None) -> tf.keras.Model:
  """Builds MobileNet 3d backbone from a config."""
  backbone_type = model_config.backbone.type
  backbone_cfg = model_config.backbone.get()
  norm_activation_config = model_config.norm_activation
  assert backbone_type == 'mobilenet', (f'Inconsistent backbone type '
                                        f'{backbone_type}')

  return MobileNet(
      model_id=backbone_cfg.model_id,
      filter_size_scale=backbone_cfg.filter_size_scale,
      input_specs=input_specs,
      stochastic_depth_drop_rate=backbone_cfg.stochastic_depth_drop_rate,
      use_sync_bn=norm_activation_config.use_sync_bn,
      norm_momentum=norm_activation_config.norm_momentum,
      norm_epsilon=norm_activation_config.norm_epsilon,
      kernel_regularizer=l2_regularizer)

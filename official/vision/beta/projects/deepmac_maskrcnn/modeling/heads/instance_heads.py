# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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

"""Instance prediction heads."""

# Import libraries

from absl import logging
import tensorflow as tf

from official.modeling import tf_utils
from official.vision.beta.projects.deepmac_maskrcnn.modeling.heads import hourglass_network


@tf.keras.utils.register_keras_serializable(package='Vision')
class DeepMaskHead(tf.keras.layers.Layer):
  """Creates a mask head."""

  def __init__(self,
               num_classes,
               upsample_factor=2,
               num_convs=4,
               num_filters=256,
               use_separable_conv=False,
               activation='relu',
               use_sync_bn=False,
               norm_momentum=0.99,
               norm_epsilon=0.001,
               kernel_regularizer=None,
               bias_regularizer=None,
               class_agnostic=False,
               convnet_variant='default',
               **kwargs):
    """Initializes a mask head.

    Args:
      num_classes: An `int` of the number of classes.
      upsample_factor: An `int` that indicates the upsample factor to generate
        the final predicted masks. It should be >= 1.
      num_convs: An `int` number that represents the number of the intermediate
        convolution layers before the mask prediction layers.
      num_filters: An `int` number that represents the number of filters of the
        intermediate convolution layers.
      use_separable_conv: A `bool` that indicates whether the separable
        convolution layers is used.
      activation: A `str` that indicates which activation is used, e.g. 'relu',
        'swish', etc.
      use_sync_bn: A `bool` that indicates whether to use synchronized batch
        normalization across different replicas.
      norm_momentum: A `float` of normalization momentum for the moving average.
      norm_epsilon: A `float` added to variance to avoid dividing by zero.
      kernel_regularizer: A `tf.keras.regularizers.Regularizer` object for
        Conv2D. Default is None.
      bias_regularizer: A `tf.keras.regularizers.Regularizer` object for Conv2D.
      class_agnostic: A `bool`. If set, we use a single channel mask head that
        is shared between all classes.
      convnet_variant: A `str` denoting the architecture of network used in the
        head. Supported options are 'default', 'hourglass20', 'hourglass52'
        and 'hourglass100'.
      **kwargs: Additional keyword arguments to be passed.
    """
    super(DeepMaskHead, self).__init__(**kwargs)
    self._config_dict = {
        'num_classes': num_classes,
        'upsample_factor': upsample_factor,
        'num_convs': num_convs,
        'num_filters': num_filters,
        'use_separable_conv': use_separable_conv,
        'activation': activation,
        'use_sync_bn': use_sync_bn,
        'norm_momentum': norm_momentum,
        'norm_epsilon': norm_epsilon,
        'kernel_regularizer': kernel_regularizer,
        'bias_regularizer': bias_regularizer,
        'class_agnostic': class_agnostic,
        'convnet_variant': convnet_variant,
    }

    if tf.keras.backend.image_data_format() == 'channels_last':
      self._bn_axis = -1
    else:
      self._bn_axis = 1
    self._activation = tf_utils.get_activation(activation)

  def _get_conv_op_and_kwargs(self):
    conv_op = (tf.keras.layers.SeparableConv2D
               if self._config_dict['use_separable_conv']
               else tf.keras.layers.Conv2D)
    conv_kwargs = {
        'filters': self._config_dict['num_filters'],
        'kernel_size': 3,
        'padding': 'same',
    }
    if self._config_dict['use_separable_conv']:
      conv_kwargs.update({
          'depthwise_initializer': tf.keras.initializers.VarianceScaling(
              scale=2, mode='fan_out', distribution='untruncated_normal'),
          'pointwise_initializer': tf.keras.initializers.VarianceScaling(
              scale=2, mode='fan_out', distribution='untruncated_normal'),
          'bias_initializer': tf.zeros_initializer(),
          'depthwise_regularizer': self._config_dict['kernel_regularizer'],
          'pointwise_regularizer': self._config_dict['kernel_regularizer'],
          'bias_regularizer': self._config_dict['bias_regularizer'],
      })
    else:
      conv_kwargs.update({
          'kernel_initializer': tf.keras.initializers.VarianceScaling(
              scale=2, mode='fan_out', distribution='untruncated_normal'),
          'bias_initializer': tf.zeros_initializer(),
          'kernel_regularizer': self._config_dict['kernel_regularizer'],
          'bias_regularizer': self._config_dict['bias_regularizer'],
      })

    return conv_op, conv_kwargs

  def _get_bn_op_and_kwargs(self):

    bn_op = (tf.keras.layers.experimental.SyncBatchNormalization
             if self._config_dict['use_sync_bn']
             else tf.keras.layers.BatchNormalization)
    bn_kwargs = {
        'axis': self._bn_axis,
        'momentum': self._config_dict['norm_momentum'],
        'epsilon': self._config_dict['norm_epsilon'],
    }

    return bn_op, bn_kwargs

  def build(self, input_shape):
    """Creates the variables of the head."""

    conv_op, conv_kwargs = self._get_conv_op_and_kwargs()

    self._build_convnet_variant()

    self._deconv = tf.keras.layers.Conv2DTranspose(
        filters=self._config_dict['num_filters'],
        kernel_size=self._config_dict['upsample_factor'],
        strides=self._config_dict['upsample_factor'],
        padding='valid',
        kernel_initializer=tf.keras.initializers.VarianceScaling(
            scale=2, mode='fan_out', distribution='untruncated_normal'),
        bias_initializer=tf.zeros_initializer(),
        kernel_regularizer=self._config_dict['kernel_regularizer'],
        bias_regularizer=self._config_dict['bias_regularizer'],
        name='mask-upsampling')

    bn_op, bn_kwargs = self._get_bn_op_and_kwargs()
    self._deconv_bn = bn_op(name='mask-deconv-bn', **bn_kwargs)

    if self._config_dict['class_agnostic']:
      num_filters = 1
    else:
      num_filters = self._config_dict['num_classes']

    conv_kwargs = {
        'filters': num_filters,
        'kernel_size': 1,
        'padding': 'valid',
    }
    if self._config_dict['use_separable_conv']:
      conv_kwargs.update({
          'depthwise_initializer': tf.keras.initializers.VarianceScaling(
              scale=2, mode='fan_out', distribution='untruncated_normal'),
          'pointwise_initializer': tf.keras.initializers.VarianceScaling(
              scale=2, mode='fan_out', distribution='untruncated_normal'),
          'bias_initializer': tf.zeros_initializer(),
          'depthwise_regularizer': self._config_dict['kernel_regularizer'],
          'pointwise_regularizer': self._config_dict['kernel_regularizer'],
          'bias_regularizer': self._config_dict['bias_regularizer'],
      })
    else:
      conv_kwargs.update({
          'kernel_initializer': tf.keras.initializers.VarianceScaling(
              scale=2, mode='fan_out', distribution='untruncated_normal'),
          'bias_initializer': tf.zeros_initializer(),
          'kernel_regularizer': self._config_dict['kernel_regularizer'],
          'bias_regularizer': self._config_dict['bias_regularizer'],
      })
    self._mask_regressor = conv_op(name='mask-logits', **conv_kwargs)

    super(DeepMaskHead, self).build(input_shape)

  def call(self, inputs, training=None):
    """Forward pass of mask branch for the Mask-RCNN model.

    Args:
      inputs: A `list` of two tensors where
        inputs[0]: A `tf.Tensor` of shape [batch_size, num_instances,
          roi_height, roi_width, roi_channels], representing the ROI features.
        inputs[1]: A `tf.Tensor` of shape [batch_size, num_instances],
          representing the classes of the ROIs.
      training: A `bool` indicating whether it is in `training` mode.

    Returns:
      mask_outputs: A `tf.Tensor` of shape
        [batch_size, num_instances, roi_height * upsample_factor,
         roi_width * upsample_factor], representing the mask predictions.
    """
    roi_features, roi_classes = inputs
    batch_size, num_rois, height, width, filters = (
        roi_features.get_shape().as_list())
    if batch_size is None:
      batch_size = tf.shape(roi_features)[0]

    x = tf.reshape(roi_features, [-1, height, width, filters])

    x = self._call_convnet_variant(x)

    x = self._deconv(x)
    x = self._deconv_bn(x)
    x = self._activation(x)

    logits = self._mask_regressor(x)

    mask_height = height * self._config_dict['upsample_factor']
    mask_width = width * self._config_dict['upsample_factor']

    if self._config_dict['class_agnostic']:
      logits = tf.reshape(logits, [-1, num_rois, mask_height, mask_width, 1])
    else:
      logits = tf.reshape(
          logits,
          [-1, num_rois, mask_height, mask_width,
           self._config_dict['num_classes']])

    batch_indices = tf.tile(
        tf.expand_dims(tf.range(batch_size), axis=1), [1, num_rois])
    mask_indices = tf.tile(
        tf.expand_dims(tf.range(num_rois), axis=0), [batch_size, 1])

    if self._config_dict['class_agnostic']:
      class_gather_indices = tf.zeros_like(roi_classes, dtype=tf.int32)
    else:
      class_gather_indices = tf.cast(roi_classes, dtype=tf.int32)

    gather_indices = tf.stack(
        [batch_indices, mask_indices, class_gather_indices],
        axis=2)
    mask_outputs = tf.gather_nd(
        tf.transpose(logits, [0, 1, 4, 2, 3]), gather_indices)
    return mask_outputs

  def _build_convnet_variant(self):

    variant = self._config_dict['convnet_variant']
    if variant == 'default':
      conv_op, conv_kwargs = self._get_conv_op_and_kwargs()
      bn_op, bn_kwargs = self._get_bn_op_and_kwargs()
      self._convs = []
      self._conv_norms = []
      for i in range(self._config_dict['num_convs']):
        conv_name = 'mask-conv_{}'.format(i)
        self._convs.append(conv_op(name=conv_name, **conv_kwargs))
        bn_name = 'mask-conv-bn_{}'.format(i)
        self._conv_norms.append(bn_op(name=bn_name, **bn_kwargs))

    elif variant == 'hourglass20':
      logging.info('Using hourglass 20 network.')
      self._hourglass = hourglass_network.hourglass_20(
          self._config_dict['num_filters'], initial_downsample=False)

    elif variant == 'hourglass52':
      logging.info('Using hourglass 52 network.')
      self._hourglass = hourglass_network.hourglass_52(
          self._config_dict['num_filters'], initial_downsample=False)

    elif variant == 'hourglass100':
      logging.info('Using hourglass 100 network.')
      self._hourglass = hourglass_network.hourglass_100(
          self._config_dict['num_filters'], initial_downsample=False)

    else:
      raise ValueError('Unknown ConvNet variant - {}'.format(variant))

  def _call_convnet_variant(self, x):

    variant = self._config_dict['convnet_variant']
    if variant == 'default':
      for conv, bn in zip(self._convs, self._conv_norms):
        x = conv(x)
        x = bn(x)
        x = self._activation(x)
      return x
    elif variant == 'hourglass20':
      return self._hourglass(x)[-1]
    elif variant == 'hourglass52':
      return self._hourglass(x)[-1]
    elif variant == 'hourglass100':
      return self._hourglass(x)[-1]
    else:
      raise ValueError('Unknown ConvNet variant - {}'.format(variant))

  def get_config(self):
    return self._config_dict

  @classmethod
  def from_config(cls, config):
    return cls(**config)

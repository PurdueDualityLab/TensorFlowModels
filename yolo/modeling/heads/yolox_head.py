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

# Lint as: python3
"""Yolox heads."""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from yolo.modeling.layers import nn_blocks
from yolo.ops import box_ops

class YOLOXHead(tf.keras.layers.Layer):
  """YOLOX Prediction Head."""

  def __init__(
      self,
      num_classes,
      width=1.0,
      strides=[8, 16, 32],
      in_channels=[256, 512, 1024],
      act='silu',
      depthwise=False,
      **kwargs
  ):


    """
    Args:
        act (str): activation type of conv. Defalut value: "silu".
        depthwise (bool): whether apply depthwise conv in conv branch. Defalut value: False.
    """

    super().__init__(**kwargs)

    self._n_anchors = 1
    self._num_classes = num_classes
    self._decode_in_inference = True

    self._cls_conv = []
    self._reg_conv = []
    
    self._cls_pred = []
    self._reg_preds = []
    self._obj_preds = []
    self._stems = []

    self._prior_prob = 1e-2
    self.bias=-tf.math.log((1-self.prior_prob)/self.prior_prob)

    Conv = nn_blocks.DWConv if depthwise else nn_blocks.ConvBN 
    for i in range(len(in_channels)):
      self._stems.append(
        nn_blocks.ConvBN(
          filters=int(256 * width),
          kernel_size=1,
          strides=(1, 1),
          padding='same', # TODO
          use_bn = True,
          activation=act,
        ),
      )

      self._cls_conv.append (
        Sequential(
          [
          nn_blocks.ConvBN(
            filters=int(256 * width),
            kernel_size=3,
            strides=(1, 1),
            use_bn = True,
            activation=act,
          ),
           nn_blocks.ConvBN(
            filters=int(256 * width),
            kernel_size=3,
            strides=(1, 1),
            use_bn = True,
            activation=act,
          ),
          ]
        )
      )
    

      self._reg_conv.append(
        Sequential(
          [
            Conv(
              filters=int(256 * width),
              kernel_size=3,
              strides=(1, 1),
              use_bn = True,
              activation=act,
            ),
            Conv(
              filters=int(256 * width),
              kernel_size=3,
              strides=(1, 1),
              use_bn = True,
              activation=act,
            ),
          ]
        )
      )
        


      self._cls_pred.append(
        tf.keras.layers.Conv2D(
          filters=self._n_anchors * self._num_classes,
          kernel_size=1,
          strides=(1, 1),
          padding='valid',
          bias_initializer=tf.keras.initializers.constant(self.bias)
        ),
      )
      self._reg_preds.append(
        tf.keras.layers.Conv2D(
          filters=4,
          kernel_size=1,
          strides=(1, 1),
          padding='valid',
        ),
      )
      self._obj_preds.append(
        tf.keras.layers.Conv2D(
          filters=self._n_anchors * 1,
          kernel_size=1,
          strides=(1, 1),
          padding='valid',
          bias_initializer=tf.keras.initializers.constant(self.bias)
        ),
      )
      
      # self.use_l1 = False
      # self.l1_loss = tf.keras.losses.MAE # TODO need reduce_mean after the loss
      # self.bcewithlog_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
      # self.iou_loss = box_ops.compute_iou
      # self.strides = strides
      # self.grids = [tf.zeros(1)] * len(in_channels)

    def call(self, xin, labels=None, imgs=None):
      outputs = []
      origin_preds = []
      x_shifts = []
      y_shifts = []
      expanded_strides = []

      for k, (cls_conv, reg_conv, stride_this_level, x) in enumerate(
          zip(self.cls_convs, self.reg_convs, self.strides, xin)
      ):
          x = self.stems[k](x)
          cls_x = x
          reg_x = x

# cls - feature and output
          cls_feat = cls_conv(cls_x)
          cls_output = self.cls_preds[k](cls_feat)

# reg & obj - feature and output
          reg_feat = reg_conv(reg_x)
          reg_output = self.reg_preds[k](reg_feat)
          obj_output = self.obj_preds[k](reg_feat)
          output=Concatenate(-1)([reg_output,obj_output,cls_output])
          outputs.append(output)
      return outputs
          # if self.training:
          #     output = tf.concat([reg_output, obj_output, cls_output], 1)
          #     output, grid = self.get_output_and_grid(
          #         output, k, stride_this_level, xin[0].type()
          #     )
          #     x_shifts.append(grid[:, :, 0])
          #     y_shifts.append(grid[:, :, 1])
          #     expanded_strides.append(
          #         tf.zeros(1, grid.shape[1])
          #         .fill_(stride_this_level)
          #         .type_as(xin[0])
          #     )
          #     if self.use_l1:
          #         batch_size = reg_output.shape[0]
          #         hsize, wsize = reg_output.shape[-2:]
          #         reg_output = reg_output.view(
          #             batch_size, self.n_anchors, 4, hsize, wsize
          #         )
          #         reg_output = reg_output.permute(0, 1, 3, 4, 2).reshape( 
          #         # reg_output: batch_size * n_anchors * 4 * height * width
          #         # permute: batch_size * n_anchors * height * width * 4
          #         # reshape: [batch_size, N_anchors height width, 4]
          #         # This is also called "origin_pred"
          #             batch_size, -1, 4
          #         )
          #         origin_preds.append(reg_output.clone())

          # else:   # not training
          #     output = tf.concat(
          #         [reg_output, obj_output.sigmoid(), cls_output.sigmoid()], 1
          #     )

          # outputs.append(output)
          # return outputs
    
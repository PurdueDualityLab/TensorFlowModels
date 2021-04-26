from yolo.utils.run_utils import prep_gpu
# try:
#
# except BaseException:
#   print("GPUs ready")

from absl import app
from absl import flags
import gin
import sys

from official.core import train_utils
# pylint: disable=unused-import
from yolo.common import registry_imports
# pylint: enable=unused-import
from official.common import flags as tfm_flags

from typing import Tuple, List
from official.core import train_utils
from official.modeling import performance
from official.core import task_factory
import os

from yolo.demos import video_detect_gpu as vgu
from yolo.demos import video_detect_cpu as vcu


import tensorflow as tf


from yolo.ops import box_ops
import tensorflow.keras as ks

def build(sample):
  return 

def loss(y_pred, y_true):
  pred_box, pred_conf, pred_class = tf.split(y_pred, [4, 1, -1], axis=-1)
  true_box, true_conf, true_class = tf.split(y_true, [4, 1, -1], axis=-1)

  iou, liou = box_ops.compute_ciou(true_box, pred_box, darknet=True)
  box_loss = 1 - liou

  class_loss = ks.losses.binary_crossentropy(
        tf.expand_dims(true_class, axis=-1),
        tf.expand_dims(pred_class, axis=-1),
        label_smoothing=0.0,
        from_logits=False)

  conf_loss = ks.losses.binary_crossentropy(
        tf.expand_dims(true_conf, axis=-1),
        tf.expand_dims(pred_conf, axis=-1),
        label_smoothing=0.0,
        from_logits=False)
  
  loss = box_loss + class_loss + conf_loss
  return loss

import matplotlib.pyplot as plt
def main(_):
  prep_gpu()

  points = 3
  linspace = tf.linspace(0, 1, points)

  space = tf.reshape(linspace, [-1, 1, 1, 1, 1, 1, 1])
  space = tf.tile(space, [1, points, 1, 1, 1, 1, 1])
  space = tf.tile(space, [1, 1, points, 1, 1, 1, 1])
  space = tf.tile(space, [1, 1,  1, points, 1, 1, 1])
  space = tf.tile(space, [1, 1,  1, 1, points, 1, 1])
  space = tf.tile(space, [1, 1,  1, 1, 1, points, 1])

  # x_pred = []
  # x_true = []

  # for val1 in linspace:
  #   for val2 in linspace:
  #     for val3 in linspace:
  #       for val4 in linspace:
  #         for val5 in linspace:
  #           for val6 in linspace:
  #             y_pred = [val1, val2, val3, val4, val5, val6]
  #             y_true = [0.5, 0.5, 1, 1, 1, 1]
              
  #             x_pred.append(y_pred)
  #             x_true.append(y_true)

  # x_true = tf.cast(tf.convert_to_tensor(x_true), tf.float32)
  # x_pred = tf.cast(tf.convert_to_tensor(x_pred), x_true.dtype)

  # loss_val = loss(x_pred, x_true).numpy()
  # plt.imshow(loss_val)
  # # plt.show()
  # print(x_pred, loss_val)



if __name__ == '__main__':
  app.run(main)


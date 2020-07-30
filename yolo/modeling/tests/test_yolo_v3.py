import tensorflow as tf
import tensorflow.keras as ks
import numpy as np
from absl.testing import parameterized

import os
import unittest

from yolo.modeling import yolo_v3
from yolo.utils.file_manager import download


class Yolov3Test(tf.test.TestCase, parameterized.TestCase):
    pass

if __name__ == '__main__':
    # init = tf.random_normal_initializer()
    # x = tf.Variable(initial_value=init(shape=(1, 416, 416, 3), dtype=tf.float32))
    with tf.device("/GPU:0"):
        model = yolo_v3.Yolov3(dn2tf_backbone = True, classes=80, dn2tf_head = True, input_shape= (None, 416, 416, 3), config_file=download("yolov3.cfg"), weights_file=download('yolov3.weights'))
        model.build(input_shape = (1, 416, 416, 3))
        model.summary()

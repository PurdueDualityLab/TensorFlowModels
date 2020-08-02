import tensorflow as tf
import tensorflow.keras as ks
import numpy as np
from absl.testing import parameterized

import os
import unittest

from yolo.modeling import yolo_v3
from yolo.utils.file_manager import download


class Yolov3Test(tf.test.TestCase, parameterized.TestCase):
    @parameterized.named_parameters(("yolov3", 'Yolov3', 'yolov3'),
                                    ("yolov3_spp", 'Yolov3_spp', 'yolov3-spp'),
                                    ("yolov3_tiny", 'Yolov3_tiny', 'yolov3-tiny'))
    def test_yolov3(self, model_name, filename):
        for device in ['/CPU:0', '/GPU:0']:
            with tf.device(device):
                models = yolo_v3.__dict__
                model = models[model_name](dn2tf_backbone = True, classes=80, dn2tf_head = True, input_shape= (None, 416, 416, 3), config_file=download(filename + ".cfg"), weights_file=download(filename + '.weights'))
                model.build(input_shape = (1, 416, 416, 3))
                model.summary()

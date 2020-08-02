import tensorflow as tf
import tensorflow.keras as ks
import numpy as np
from absl.testing import parameterized

import os
import unittest

from yolo.modeling import yolo_v3


class Yolov3Test(tf.test.TestCase, parameterized.TestCase):
    @parameterized.named_parameters(("yolov3", 'Yolov3'),
                                    #("yolov3_spp", 'Yolov3_spp'),
                                    #("yolov3_tiny", 'Yolov3_tiny')
                                    )
    def test_yolov3(self, model_name):
        for device in ['/CPU:0', '/GPU:0']:
            with tf.device(device):
                models = yolo_v3.__dict__
                model = models[model_name](classes=80)
                model.build(input_shape = (1, 416, 416, 3))
                model.load_weights_from_dn(dn2tf_backbone=True, dn2tf_head=True)
                model.summary()

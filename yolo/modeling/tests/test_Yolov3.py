import tensorflow as tf
import tensorflow.keras as ks
import numpy as np
from absl.testing import parameterized

import os
import unittest

from yolo import Yolov3

try:
    from tensorflow.python.framework.errors_impl import FailedPreconditionError
except ImportError:
    FailedPreconditionError = None


class Yolov3Test(tf.test.TestCase, parameterized.TestCase):
    @parameterized.named_parameters(
        ("yolov3", 'regular', "yolov3_416.weights"),
        ("yolov3_spp", 'spp', "yolov3-spp.weights"),
        ("yolov3_tiny", 'tiny', "yolov3-tiny.weights"))
    def test_yolov3(self, model_name, weights_file):
        for device in ['/CPU:0', '/GPU:0']:
            with tf.device(device):
                model = Yolov3(classes=80, model=model_name)
                model.load_weights_from_dn(dn2tf_backbone=True,
                                           dn2tf_head=True)

                if model_name == 'tiny':
                    input_shape = [1, 416, 416, 3]
                else:
                    input_shape = [1, 608, 608, 3]
                x = tf.ones(shape=input_shape, dtype=tf.float32)
                model.predict(x)

                model.summary()

                with self.assertRaises(FailedPreconditionError
                                       or NotADirectoryError):
                    model.save(os.devnull)
        return


if __name__ == "__main__":
    tf.test.main()

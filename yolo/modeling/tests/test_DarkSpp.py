import tensorflow as tf
import tensorflow.keras as ks
import numpy as np
from absl.testing import parameterized

from yolo.modeling.building_blocks import DarkSpp


class DarkSppTest(tf.test.TestCase, parameterized.TestCase):
    @parameterized.named_parameters(
        ("RouteProcessSpp", 224, 224, 3, [5, 9, 13]),
        ("test1", 300, 300, 10, [2, 3, 4, 5]), ("test2", 256, 256, 5, [10]))
    def test_pass_through(self, width, height, channels, sizes):
        x = ks.Input(shape=(width, height, channels))
        test_layer = DarkSpp(sizes=sizes)
        outx = test_layer(x)
        self.assertAllEqual(outx.shape.as_list(),
                            [None, width, height, channels * (len(sizes) + 1)])
        return

    @parameterized.named_parameters(
        ("RouteProcessSpp", 224, 224, 3, [5, 9, 13]),
        ("test1", 300, 300, 10, [2, 3, 4, 5]), ("test2", 256, 256, 5, [10]))
    def test_gradient_pass_though(self, width, height, channels, sizes):
        loss = ks.losses.MeanSquaredError()
        optimizer = ks.optimizers.SGD()
        test_layer = DarkSpp(sizes=sizes)

        init = tf.random_normal_initializer()
        x = tf.Variable(initial_value=init(shape=(1, width, height, channels),
                                           dtype=tf.float32))
        y = tf.Variable(initial_value=init(shape=(1, width, height,
                                                  channels * (len(sizes) + 1)),
                                           dtype=tf.float32))

        with tf.GradientTape() as tape:
            x_hat = test_layer(x)
            grad_loss = loss(x_hat, y)
        grad = tape.gradient(grad_loss, test_layer.trainable_variables)
        optimizer.apply_gradients(zip(grad, test_layer.trainable_variables))

        self.assertNotIn(None, grad)
        return


if __name__ == "__main__":
    tf.test.main()

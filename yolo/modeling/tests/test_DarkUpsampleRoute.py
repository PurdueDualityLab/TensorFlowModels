import tensorflow as tf
import tensorflow.keras as ks
import numpy as np
from absl.testing import parameterized

from yolo.modeling.building_blocks import DarkUpsampleRoute


class DarkUpsampleRouteTest(tf.test.TestCase, parameterized.TestCase):
    @parameterized.named_parameters(("test1", 224, 224, 64, (3, 3)),
                                    ("test2", 223, 223, 32, (2, 2)),
                                    ("test3", 255, 255, 16, (4, 4)))
    def test_pass_through(self, width, height, filters, upsampling_size):
        x_conv = ks.Input(shape=(width, height, filters))
        x_route = ks.Input(shape=(width * upsampling_size[0],
                                  height * upsampling_size[1], filters))
        test_layer = DarkUpsampleRoute(filters=filters,
                                       upsampling_size=upsampling_size)
        outx = test_layer([x_conv, x_route])
        self.assertAllEqual(outx.shape.as_list(), [
            None, width * upsampling_size[0], height * upsampling_size[1],
            filters * 2
        ])

    @parameterized.named_parameters(("test1", 224, 224, 64, (3, 3)),
                                    ("test2", 223, 223, 32, (2, 2)),
                                    ("test3", 255, 255, 16, (4, 4)))
    def test_gradient_pass_though(self, width, height, filters,
                                  upsampling_size):
        loss = ks.losses.MeanSquaredError()
        optimizer = ks.optimizers.SGD()
        test_layer = DarkUpsampleRoute(filters=filters,
                                       upsampling_size=upsampling_size)

        init = tf.random_normal_initializer()
        x_conv = tf.Variable(initial_value=init(
            shape=(1, width, height, filters), dtype=tf.float32))
        x_route = tf.Variable(
            initial_value=init(shape=(1, width * upsampling_size[0],
                                      height * upsampling_size[1], filters),
                               dtype=tf.float32))
        y = tf.Variable(initial_value=init(shape=(1,
                                                  width * upsampling_size[0],
                                                  height * upsampling_size[1],
                                                  filters * 2),
                                           dtype=tf.float32))

        with tf.GradientTape() as tape:
            x_hat = test_layer([x_conv, x_route])
            grad_loss = loss(x_hat, y)
        grad = tape.gradient(grad_loss, test_layer.trainable_variables)
        optimizer.apply_gradients(zip(grad, test_layer.trainable_variables))

        self.assertNotIn(None, grad)
        return


if __name__ == "__main__":
    tf.test.main()

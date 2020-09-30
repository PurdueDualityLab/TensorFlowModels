import tensorflow as tf
import tensorflow.keras as ks
import numpy as np
from absl.testing import parameterized

from yolo.modeling.building_blocks import DarkRouteProcess


class DarkRouteProcessTest(tf.test.TestCase, parameterized.TestCase):
    @parameterized.named_parameters(
        ("test1", 224, 224, 64, 7, False), ("test2", 223, 223, 32, 3, False),
        ("tiny", 223, 223, 16, 1, False), ("spp", 224, 224, 64, 7, False))
    def test_pass_through(self, width, height, filters, repetitions, spp):
        x = ks.Input(shape=(width, height, filters))
        test_layer = DarkRouteProcess(filters=filters,
                                      repetitions=repetitions,
                                      insert_spp=spp)
        outx = test_layer(x)
        self.assertEqual(len(outx), 2, msg="len(outx) != 2")
        self.assertAllEqual(outx[1].shape.as_list(),
                            [None, width, height, filters])
        self.assertAllEqual(
            filters % 2,
            0,
            msg=
            "Output of a DarkRouteProcess layer has an odd number of filters")
        self.assertAllEqual(outx[0].shape.as_list(),
                            [None, width, height, filters // 2])

    @parameterized.named_parameters(
        ("test1", 224, 224, 64, 7, False), ("test2", 223, 223, 32, 3, False),
        ("tiny", 223, 223, 16, 1, False), ("spp", 224, 224, 64, 7, False))
    def test_gradient_pass_though(self, width, height, filters, repetitions,
                                  spp):
        loss = ks.losses.MeanSquaredError()
        optimizer = ks.optimizers.SGD()
        test_layer = DarkRouteProcess(filters=filters,
                                      repetitions=repetitions,
                                      insert_spp=spp)

        init = tf.random_normal_initializer()
        x = tf.Variable(initial_value=init(shape=(1, width, height, filters),
                                           dtype=tf.float32))
        y_0 = tf.Variable(initial_value=init(
            shape=(1, width, height, filters // 2), dtype=tf.float32))
        y_1 = tf.Variable(initial_value=init(shape=(1, width, height, filters),
                                             dtype=tf.float32))

        with tf.GradientTape() as tape:
            x_hat_0, x_hat_1 = test_layer(x)
            grad_loss_0 = loss(x_hat_0, y_0)
            grad_loss_1 = loss(x_hat_1, y_1)
        grad = tape.gradient([grad_loss_0, grad_loss_1],
                             test_layer.trainable_variables)
        optimizer.apply_gradients(zip(grad, test_layer.trainable_variables))

        self.assertNotIn(None, grad)
        return


if __name__ == "__main__":
    tf.test.main()

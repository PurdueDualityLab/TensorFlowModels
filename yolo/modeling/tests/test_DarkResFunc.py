import tensorflow as tf
import tensorflow.keras as ks
import numpy as np
from absl.testing import parameterized

from yolo.modeling.building_blocks import DarkResFunc as layer


class DarkResidualTest(tf.test.TestCase, parameterized.TestCase):
    @parameterized.named_parameters(("same", 224, 224, 64, False),
                                    ("downsample", 223, 223, 32, True),
                                    ("oddball", 223, 223, 32, False))
    def test_pass_through(self, width, height, filters, downsample):
        mod = 1
        if downsample:
            mod = 2
        init = tf.random_normal_initializer()
        inp = tf.Variable(initial_value=init(shape=(1, width, height, filters),
                                             dtype=tf.float32))

        x = ks.Input(shape=(width, height, filters))
        test_layer = layer(filters=filters, downsample=downsample)(x)
        model = ks.Model(inputs=x, outputs=test_layer)
        model.build(input_shape=(None, width, height, filters))

        outx = model(inp)
        print(inp.shape, outx.shape.as_list())
        self.assertAllEqual(
            outx.shape.as_list(),
            [1, np.ceil(width / mod),
             np.ceil(height / mod), filters])
        return

    @parameterized.named_parameters(("same", 64, 224, 224, False),
                                    ("downsample", 32, 223, 223, True),
                                    ("oddball", 32, 223, 223, False))
    def test_gradient_pass_though(self, filters, width, height, downsample):
        loss = ks.losses.MeanSquaredError()
        optimizer = ks.optimizers.SGD()
        p = ks.Input(shape=(width, height, filters))
        test_layer = layer(filters=filters, downsample=downsample)(p)
        model = ks.Model(inputs=p, outputs=test_layer)

        if downsample:
            mod = 2
        else:
            mod = 1

        init = tf.random_normal_initializer()
        x = tf.Variable(initial_value=init(shape=(1, width, height, filters),
                                           dtype=tf.float32))
        y = tf.Variable(initial_value=init(shape=(1, int(np.ceil(width / mod)),
                                                  int(np.ceil(height / mod)),
                                                  filters),
                                           dtype=tf.float32))

        with tf.GradientTape() as tape:
            x_hat = model(x)
            grad_loss = loss(x_hat, y)
        grad = tape.gradient(grad_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grad, model.trainable_variables))

        self.assertNotIn(None, grad)
        return


if __name__ == "__main__":
    tf.test.main()

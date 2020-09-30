import tensorflow as tf
import tensorflow.keras as ks
import numpy as np
from absl.testing import parameterized

from yolo.modeling.backbones import csp_backbone_builder as builder


class CSP_BackBoneTest(tf.test.TestCase, parameterized.TestCase):
    @parameterized.named_parameters(
        ("simple", "darknet53", (1, 224, 224, 3)),
        ("odd_shape_reg", "darknet53", (1, 224, 128, 3)),
    )
    # ("darknet_tiny", "darknet_tiny",(1, 224, 224, 3)),
    # ("odd_shape_tiny", "darknet_tiny",(1, 224, 128, 3)),
    # ("odd_shape_reg", "darknet53", (1, 224, 128, 3)),
    # ("odd_shape_rev_tiny","darknet_tiny", (1, 128, 224, 3)),
    # ("odd_shape_rev_reg","darknet53", (1, 128, 224, 3)),)
    def test_pass_through(self, model, input_shape):
        if model == "darknet53":
            check = {
                '256': [
                    input_shape[0], input_shape[1] // 8, input_shape[2] // 8,
                    256
                ],
                '512': [
                    input_shape[0], input_shape[1] // 16, input_shape[2] // 16,
                    512
                ],
                '1024': [
                    input_shape[0], input_shape[1] // 32, input_shape[2] // 32,
                    1024
                ],
            }
        elif model == "darknet_tiny":
            check = {
                '256': [
                    input_shape[0], input_shape[1] // 16, input_shape[2] // 16,
                    256
                ],
                '1024': [
                    input_shape[0], input_shape[1] // 32, input_shape[2] // 32,
                    1024
                ],
            }
        else:
            check = None
        init = tf.random_normal_initializer()
        x = tf.Variable(
            initial_value=init(shape=input_shape, dtype=tf.float32))
        y = builder.CSP_Backbone_Builder(model)(x)

        y_shape = {key: value.shape.as_list() for key, value in y.items()}
        self.assertAllEqual(check, y_shape)
        print(y_shape, check)
        return

    @parameterized.named_parameters(("simple", "darknet53", (1, 224, 224, 3)))
    def test_gradient_pass_though(self, model, input_shape):
        if model == "darknet53":
            check = {
                '256': [
                    input_shape[0], input_shape[1] // 8, input_shape[2] // 8,
                    256
                ],
                '512': [
                    input_shape[0], input_shape[1] // 16, input_shape[2] // 16,
                    512
                ],
                '1024': [
                    input_shape[0], input_shape[1] // 32, input_shape[2] // 32,
                    1024
                ],
            }
        elif model == "darknet_tiny":
            check = {
                '256': [
                    input_shape[0], input_shape[1] // 16, input_shape[2] // 16,
                    256
                ],
                '1024': [
                    input_shape[0], input_shape[1] // 32, input_shape[2] // 32,
                    1024
                ],
            }
        else:
            check = None
            return

        loss = ks.losses.MeanSquaredError()
        optimizer = ks.optimizers.SGD()
        test_layer = builder.CSP_Backbone_Builder(model)

        init = tf.random_normal_initializer()
        x = tf.Variable(
            initial_value=init(shape=input_shape, dtype=tf.float32))
        y = {
            key: tf.Variable(initial_value=init(shape=value, dtype=tf.float32))
            for key, value in check.items()
        }

        with tf.GradientTape() as tape:
            x_hat = test_layer(x)
            losses = 0
            for key in y:
                grad_loss = loss(x_hat[key], y[key])
                losses += grad_loss

        grad = tape.gradient(losses, test_layer.trainable_variables)
        optimizer.apply_gradients(zip(grad, test_layer.trainable_variables))
        self.assertNotIn(None, grad)
        return


if __name__ == "__main__":
    tf.test.main()

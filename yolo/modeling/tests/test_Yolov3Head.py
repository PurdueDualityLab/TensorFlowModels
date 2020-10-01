import tensorflow as tf
import tensorflow.keras as ks
import numpy as np
from absl.testing import parameterized

import functools
import os
import unittest

from yolo.modeling.backbones import backbone_builder as builder
from yolo.modeling.model_heads._Yolov3Head import Yolov3Head


class Yolov3HeadTest(tf.test.TestCase, parameterized.TestCase):
    @parameterized.named_parameters(
        ("simple", "darknet53", (1, 224, 224, 3)),
        ("darknet_tiny", "darknet_tiny", (1, 224, 224, 3)),
        ("odd_shape_tiny", "darknet_tiny", (1, 224, 128, 3)),
        ("odd_shape_reg", "darknet53", (1, 224, 128, 3)),
        ("odd_shape_rev_tiny", "darknet_tiny", (1, 128, 224, 3)),
        ("odd_shape_rev_reg", "darknet53", (1, 128, 224, 3)),
    )
    def test_pass_through(self, model, input_shape):
        if model == "darknet53":
            check = {
                '1024': [
                    input_shape[0], input_shape[1] // 32, input_shape[2] // 32,
                    255
                ],
                '512': [
                    input_shape[0], input_shape[1] // 16, input_shape[2] // 16,
                    255
                ],
                '256': [
                    input_shape[0], input_shape[1] // 8, input_shape[2] // 8,
                    255
                ],
            }
            name = "spp"
        elif model == "darknet_tiny":
            check = {
                '1024': [
                    input_shape[0], input_shape[1] // 32, input_shape[2] // 32,
                    340
                ],
                '256': [
                    input_shape[0], input_shape[1] // 16, input_shape[2] // 16,
                    340
                ],
            }
            name = "tiny"
        else:
            check = None
            return
        init = tf.random_normal_initializer()
        x = tf.Variable(
            initial_value=init(shape=input_shape, dtype=tf.float32))
        y = builder.Backbone_Builder(model)(x)
        pred = Yolov3Head(name)(y)

        y_shape = {key: value.shape.as_list() for key, value in pred.items()}
        self.assertAllEqual(check, y_shape)
        print(y_shape, check)
        return

    @parameterized.named_parameters(
        ("simple", "darknet53", (1, 224, 224, 3)),
        ("darknet_tiny", "darknet_tiny", (1, 224, 224, 3)),
        ("odd_shape_tiny", "darknet_tiny", (1, 224, 128, 3)),
        ("odd_shape_reg", "darknet53", (1, 224, 128, 3)),
        ("odd_shape_rev_tiny", "darknet_tiny", (1, 128, 224, 3)),
        ("odd_shape_rev_reg", "darknet53", (1, 128, 224, 3)),
    )
    def test_gradient_pass_though(self, model, input_shape):
        if model == "darknet53":
            check = {
                '1024': [
                    input_shape[0], input_shape[1] // 32, input_shape[2] // 32,
                    255
                ],
                '512': [
                    input_shape[0], input_shape[1] // 16, input_shape[2] // 16,
                    255
                ],
                '256': [
                    input_shape[0], input_shape[1] // 8, input_shape[2] // 8,
                    255
                ],
            }
            name = "spp"
        elif model == "darknet_tiny":
            check = {
                '1024': [
                    input_shape[0], input_shape[1] // 32, input_shape[2] // 32,
                    340
                ],
                '256': [
                    input_shape[0], input_shape[1] // 16, input_shape[2] // 16,
                    340
                ],
            }
            name = "tiny"
        else:
            check = None
            return

        loss = ks.losses.MeanSquaredError()
        optimizer = ks.optimizers.SGD()
        test_layer = builder.Backbone_Builder(model)
        pred = Yolov3Head(name)

        init = tf.random_normal_initializer()
        x = tf.Variable(
            initial_value=init(shape=input_shape, dtype=tf.float32))
        y = {
            key: tf.Variable(initial_value=init(shape=value, dtype=tf.float32))
            for key, value in check.items()
        }

        with tf.GradientTape() as tape:
            x_cent = test_layer(x)
            x_hat = pred(x_cent)

            losses = 0
            for key in y:
                grad_loss = loss(x_hat[key], y[key])
                losses += grad_loss

        grad = tape.gradient(losses, pred.trainable_variables)
        optimizer.apply_gradients(zip(grad, pred.trainable_variables))

        self.assertNotIn(None, grad)
        return


if __name__ == "__main__":
    tf.test.main()

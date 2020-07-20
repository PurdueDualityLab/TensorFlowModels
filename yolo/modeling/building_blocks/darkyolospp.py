"""Contains common building blocks for yolo neural networks."""
import tensorflow as tf
import tensorflow.keras as ks


@ks.utils.register_keras_serializable(package='yolo')
class darkyolospp(ks.layers.Layer):
    def __init__(self, stride, sizes, **kwargs):
        self.stride = stride
        self.sizes = sizes
        super().__init__(**kwargs)
        return

    def build(self, input_shape):
        maxpools = []
        for size in sizes:
            maxpools.append(tf.keras.layers.MaxPool2D(
                pool_size=self.size, strides=self.stride, padding='valid', data_format=None))
        self.maxpools = maxpools
        super().build(input_shape)
        return

    def call(self, inputs):
        outputs = []
        for maxpool in self.maxpools:
            outputs.append(maxpool(inputs))
        concat_output = ks.layers.concatenate(outputs)
        return concat_output

    def get_config(self):
        layer_config = {
            "stride": self.stride,
            "sizes": self.sizes
        }
        layer_config.update(super().get_config())
        return layer_config

"""Contains common building blocks for yolo neural networks."""
import tensorflow as tf
import tensorflow.keras as ks


@ks.utils.register_keras_serializable(package='yolo')
class DarkSpp(ks.layers.Layer):
    def __init__(self, sizes, **kwargs):
        self._sizes = list(reversed(sizes))
        # print(self._sizes)
        if len(sizes) == 0:
            raise ValueError(
                "More than one maxpool should be specified in SSP block")
        super().__init__(**kwargs)
        return

    def build(self, input_shape):
        maxpools = []
        for size in self._sizes:
            maxpools.append(
                tf.keras.layers.MaxPool2D(pool_size=(size, size),
                                          strides=(1, 1),
                                          padding='same',
                                          data_format=None))
        self._maxpools = maxpools
        super().build(input_shape)
        return

    def call(self, inputs):
        outputs = []
        for maxpool in self._maxpools:
            outputs.append(maxpool(inputs))
        outputs.append(inputs)
        concat_output = ks.layers.concatenate(outputs)
        return concat_output

    def get_config(self):
        layer_config = {"sizes": self._sizes}
        layer_config.update(super().get_config())
        return layer_config

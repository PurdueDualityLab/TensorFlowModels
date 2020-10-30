"""Contains common building blocks for yolo neural networks."""
import tensorflow as tf
import tensorflow.keras as ks


@ks.utils.register_keras_serializable(package='yolo')
class DarkShortcut(ks.layers.Layer):
    def __init__(self, activation='linear', **kwargs):
        '''
        Modified DarkShortcut layer to match that of the DarkNet Library.
        It adds input layers and passes the result through an activation
        function.

        It is used like the following:

        x = bottleneck_shortcut = DarkConv(64, (3, 3), (1, 1))(x)
        x = DarkConv(32, (1, 1), (1, 1))(x)
        x = DarkConv(64, (3, 3), (1, 1))(x)
        x = DarkShortcut('linear')([x, bottleneck_shortcut])

        Args:
            activation: string or None for activation function to use in layer,
                        if None activation is replaced by linear
            **kwargs: Keyword Arguments

        '''

        # activation params
        if activation is None:
            self._activation = 'linear'
        else:
            self._activation = activation

        super().__init__(**kwargs)

    def build(self, input_shape):
        self._activation_fn = ks.layers.Activation(activation=self._activation)
        super().build(input_shape)

    def call(self, input):
        x = ks.layers.add(input)
        x = self._activation_fn(x)
        return x

    def get_config(self):
        # used to store/share parameters to reconsturct the model
        layer_config = {"activation": self._activation}
        layer_config.update(super().get_config())
        return layer_config

"""Contains common building blocks for yolo neural networks."""
import tensorflow as tf
import tensorflow.keras as ks
from ._DarkConv import DarkConv


@ks.utils.register_keras_serializable(package='yolo')
class DarkUpsampleRoute(ks.layers.Layer):
    def __init__(
            self,
            filters=1,
            use_bias=True,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            bias_regularizer=None,
            l2_regularization=5e-4,  # default find where is it is stated
            use_bn=True,
            use_sync_bn=False,
            norm_moment=0.99,
            norm_epsilon=0.001,
            conv_activation='leaky',
            leaky_alpha=0.1,
            upsampling_size=(2, 2),
            **kwargs):

        # darkconv params
        self._filters = filters
        self._use_bias = use_bias
        self._kernel_initializer = kernel_initializer
        self._bias_initializer = bias_initializer
        self._bias_regularizer = bias_regularizer
        self._l2_regularization = l2_regularization
        self._use_bn = use_bn
        self._use_sync_bn = use_sync_bn

        # normal params
        self._norm_moment = norm_moment
        self._norm_epsilon = norm_epsilon

        # activation params
        self._conv_activation = conv_activation
        self._leaky_alpha = leaky_alpha
        self._upsampling_size = upsampling_size

        super().__init__(**kwargs)

    def build(self, input_shape):
        self._conv = DarkConv(filters=self._filters,
                              kernel_size=(1, 1),
                              strides=(1, 1),
                              padding='same',
                              use_bias=self._use_bias,
                              kernel_initializer=self._kernel_initializer,
                              bias_initializer=self._bias_initializer,
                              l2_regularization=self._l2_regularization,
                              use_bn=self._use_bn,
                              use_sync_bn=self._use_sync_bn,
                              norm_moment=self._norm_moment,
                              norm_epsilon=self._norm_epsilon,
                              activation=self._conv_activation,
                              leaky_alpha=self._leaky_alpha)
        self._upsample = ks.layers.UpSampling2D(size=self._upsampling_size)
        self._concat = tf.keras.layers.Concatenate()
        super().build(input_shape)
        return

    def call(self, inputs):
        # done this way to prevent confusion in the auto graph
        inputToConvolve, inputRouted = inputs

        x = self._conv(inputToConvolve)
        x = self._upsample(x)
        x = self._concat([x, inputRouted])
        return x

    def get_config(self):
        # used to store/share parameters to reconsturct the model
        layer_config = {
            "filters": self._filters,
            "use_bias": self._use_bias,
            "kernel_initializer": self._kernel_initializer,
            "bias_initializer": self._bias_initializer,
            "l2_regularization": self._l2_regularization,
            "use_bn": self._use_bn,
            "use_sync_bn": self._use_sync_bn,
            "norm_moment": self._norm_moment,
            "norm_epsilon": self._norm_epsilon,
            "conv_activation": self._conv_activation,
            "leaky_alpha": self._leaky_alpha,
            "upsampling_size": self._upsampling_size
        }
        layer_config.update(super().get_config())
        return layer_config

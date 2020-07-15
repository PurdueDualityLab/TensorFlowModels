"""Contains common building blocks for yolo neural networks."""
import tensorflow as tf
import tensorflow.keras as ks
from . import DarkConv

class DarkResidual(ks.layers.Layer):
    def __init__(self,
                 filters=1,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 use_bn=True,
                 use_sync_bn=False,
                 norm_moment=0.99,
                 norm_epsilon=0.001,
                 conv_activation='leaky',
                 leaky_alpha=0.1,
                 sc_activation='linear',
                 downsample = False, 
                 **kwargs):
        '''

        '''

        #downsample
        self._downsample = downsample

        #darkconv params
        self._filters = filters
        self._use_bias = use_bias
        self._kernel_initializer = kernel_initializer
        self._bias_initializer = bias_initializer
        self._use_bn = use_bn
        self._use_sync_bn = use_sync_bn

        #normal params 
        self._norm_moment = norm_moment
        self._norm_epsilon = norm_epsilon

        #activation params 
        self._conv_activation = conv_activation
        self._leak_alpha = leak_alpha
        self._sc_activation = sc_activation

        super().__init__(**kwargs)
        return

    def build(self, input_shape):
        if self._downsample: 
            self._dconv = DarkConv(
                filters = self._filters, 
                kernel_size=(3,3)
                strides  = (2,2), 
                padding = 'same', 
                use_bias = self._use_bias
                kernel_initializer = self._kernel_initializer, 
                bias_initializer = self._bias_initializer, 
                use_bn = self._use_bn, 
                use_sync_bn = self._use_sync_bn, 
                norm_moment= self._norm_moment, 
                norm_epsilon= self._norm_epsilon,
                activation= self._conv_activation,
                leaky_alpha= self._leak_alpha, 
                )
            
        self._conv1 = DarkConv()
        self._conv2 = DarkConv()

        self._shortcut = tf.layers.Add()
        self._activation_fn = ks.layers.Activation(activation=self._sc_activation)

        super().build(input_shape)
        return

    def call(self, input):
        x = self._adder(input)
        x = self._activation_fn(x)
        return x

    def get_config(self):
        # used to store/share parameters to reconsturct the model
        layer_config = {
            "activation": self._activation
        }
        layer_config.update(super().get_config())
        return layer_config


if __name__ == "__main__":
    # need to build proper unit tests below is temporary
    norm = tf.random_normal_initializer()
    # this variable needs to be a different size
    x = tf.Variable(
        initial_value=norm(
            shape=[
                1,
                224,
                224,
                3],
            dtype=tf.dtypes.float32))

    
    x = bottleneck_shortcut = DarkConv(64, (3, 3), (1, 1))(x)
    x = DarkConv(32, (1, 1), (1, 1))(x)
    x = DarkConv(64, (3, 3), (1, 1))(x)
    x = DarkShortcut('linear')([x, bottleneck_shortcut])

    test = DarkShortcut('linear')
    test1 = DarkShortcut().from_config(test.get_config())

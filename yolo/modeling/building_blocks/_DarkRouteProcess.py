import tensorflow as tf
import tensorflow.keras as ks
from yolo.modeling.building_blocks import DarkConv, DarkSpp


class DarkRouteProcess(ks.layers.Layer):
    def __init__(self,
                 filters=2,
                 repetitions=2,
                 insert_spp=False,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 norm_moment=0.99,
                 norm_epsilon=0.001,
                 activation='leaky',
                 leaky_alpha=0.1,
                 **kwargs):
        """
        process darknet outputs and connect back bone to head more generalizably
        Abstracts repetition of DarkConv objects that is common in YOLO.

        It is used like the following:

        x = DarkConv(1024, (3, 3), (1, 1))(x)
        proc = DarkRouteProcess(filters = 1024, repetitions = 3, insert_spp = False)(x)

        Args:
            filters: the number of filters to be used in all subsequent layers
                     filters should be the depth of the tensor input into this layer, as no downsampling can be done within this layer object
            repititions: number of times to repeat the processign nodes
                         for tiny: 1 repition, no spp allowed
                         for spp: insert_spp = True, and allow for 3+ repetitions
                         for regular: insert_spp = False, and allow for 3+ repetitions
            insert_spp: bool if true add the spatial pyramid pooling layer
            kernel_initializer: method to use to initializa kernel weights
            bias_initializer: method to use to initialize the bias of the conv layers
            norm_moment: batch norm parameter see Tensorflow documentation
            norm_epsilon: batch norm parameter see Tensorflow documentation
            activation: activation function to use in processing
            leaky_alpha: if leaky acitivation function, the alpha to use in processing the relu input

        Returns:
            callable tensorflow layer

        Raises:
            None
        """

        # darkconv params
        self._filters = filters
        self._kernel_initializer = kernel_initializer
        self._bias_initializer = bias_initializer

        # normal params
        self._norm_moment = norm_moment
        self._norm_epsilon = norm_epsilon

        # activation params
        self._activation = activation
        self._leaky_alpha = leaky_alpha

        # layer configs
        self._repetitions = repetitions
        self._insert_spp = insert_spp

        self.layer_list = self._get_layer_list()
        # print(self.layer_list)
        super().__init__(**kwargs)
        return

    def _get_layer_list(self):
        layer_config = ['block'] * self._repetitions
        if self._repetitions > 2 and self._insert_spp:
            layer_config[1] = 'spp'
        return layer_config

    def _block(self, filters):
        x1 = DarkConv(
            filters=filters // 2,
            kernel_size=(1,1),
            strides=(1,1),
            padding="same",
            use_bn= True,
            kernel_initializer=self._kernel_initializer,
            bias_initializer=self._bias_initializer,
            norm_moment=self._norm_moment,
            norm_epsilon=self._norm_epsilon,
            activation=self._activation,
            leaky_alpha=self._leaky_alpha)
        x2 = DarkConv(
            filters=filters,
            kernel_size=(3,3),
            strides=(1,1),
            padding="same",
            use_bn= True,
            kernel_initializer=self._kernel_initializer,
            bias_initializer=self._bias_initializer,
            norm_moment=self._norm_moment,
            norm_epsilon=self._norm_epsilon,
            activation=self._activation,
            leaky_alpha=self._leaky_alpha)
        return [x1, x2]

    def _spp(self, filters):
        x1 = DarkConv(
            filters=filters // 2,
            kernel_size=(1,1),
            strides=(1,1),
            padding="same",
            use_bn= True,
            kernel_initializer=self._kernel_initializer,
            bias_initializer=self._bias_initializer,
            norm_moment=self._norm_moment,
            norm_epsilon=self._norm_epsilon,
            activation=self._activation,
            leaky_alpha=self._leaky_alpha)
        # repalce with spp
        x2 = DarkSpp([5, 9, 13])
        return [x1, x2]

    def build(self, input_shape):
        self.layers = []
        for layer in self.layer_list:
            if layer == 'block':
                self.layers.extend(self._block(self._filters))
            else:
                self.layers.extend(self._spp(self._filters))

        self.outputs = [False] * self._repetitions * 2
        self.outputs[-2] = True
        self.outputs[-1] = True
        self.layers = list(zip(self.outputs, self.layers))
        super().build(input_shape)
        return

    def call(self, inputs):
        # check efficiency
        x = inputs
        outputs = []
        for out, layer in self.layers:
            x = layer(x)
            if out:
                outputs.append(x)
        return outputs
    
    def get_config(self):
        # used to store/share parameters to reconsturct the model
        layer_config = {
            "filters": self._filters,
            "kernel_initializer": self._kernel_initializer,
            "bias_initializer": self._bias_initializer,
            "repetitions": self._repetitions,
            "insert_spp": self._insert_spp,
            "norm_moment": self._norm_moment,
            "norm_epsilon": self._norm_epsilon,
            "leaky_alpha": self._leaky_alpha,
            "activation":self._activation,
        }
        layer_config.update(super().get_config())
        return layer_config



# x = tf.ones(shape = (1, 200, 200, 30))
# model = DarkRouteProcess(filters = 512, repetitions = 30, insert_spp = False)
# y_deep, y_out = model(x)

# print(y_deep, y_out)

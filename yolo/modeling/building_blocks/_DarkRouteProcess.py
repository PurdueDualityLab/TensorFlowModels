import tensorflow as tf
import tensorflow.keras as ks
from yolo.modeling.building_blocks import DarkConv

class DarkRouteProcess(ks.layers.Layer):
    def __init__(self,
                 filters = 2,
                 repetitions = 2,
                 insert_spp = False,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 norm_moment=0.99,
                 norm_epsilon=0.001,
                 activation='leaky',
                 leaky_alpha=0.1,
                 **kwargs):
        
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
        #print(self.layer_list)
        super().__init__(**kwargs)
        return
    
    def _get_layer_list(self):
        layer_config = ['block'] * self._repetitions
        if self._repetitions > 2 and self._insert_spp:
            layer_config[1] = 'spp'
        return layer_config

    def _block(self, filters):
        x1 = DarkConv(filters = filters//2, kernel_size = (1,1), strides = (1,1), padding = "same", kernel_initializer = self._kernel_initializer, bias_initializer = self._bias_initializer, norm_moment = self._norm_moment, norm_epsilon = self._norm_epsilon, activation = self._activation, leaky_alpha = self._leaky_alpha)
        x2 = DarkConv(filters = filters, kernel_size = (3,3), strides = (1,1), padding = "same", kernel_initializer = self._kernel_initializer, bias_initializer = self._bias_initializer, norm_moment = self._norm_moment, norm_epsilon = self._norm_epsilon, activation = self._activation, leaky_alpha = self._leaky_alpha)
        return [x1, x2]
    
    def _spp(self, filters):
        x1 = DarkConv(filters = filters//2, kernel_size = (1,1), strides = (1,1), padding = "same", kernel_initializer = self._kernel_initializer, bias_initializer = self._bias_initializer, norm_moment = self._norm_moment, norm_epsilon = self._norm_epsilon, activation = self._activation, leaky_alpha = self._leaky_alpha)
        # repalce with spp
        x2 = DarkConv(filters = filters, kernel_size = (3,3), strides = (1,1), padding = "same", kernel_initializer = self._kernel_initializer, bias_initializer = self._bias_initializer, norm_moment = self._norm_moment, norm_epsilon = self._norm_epsilon, activation = self._activation, leaky_alpha = self._leaky_alpha)
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


# x = tf.ones(shape = (1, 200, 200, 30))
# model = DarkRouteProcess(filters = 512, repetitions = 30, insert_spp = False)
# y_deep, y_out = model(x)

# print(y_deep, y_out)
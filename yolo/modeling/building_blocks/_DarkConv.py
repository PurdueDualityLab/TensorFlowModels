"""Contains common building blocks for yolo neural networks."""
import functools

import tensorflow as tf
import tensorflow.keras as ks
import tensorflow.keras.backend as K
from yolo.modeling.building_blocks._Identity import Identity


@ks.utils.register_keras_serializable(package='yolo')
class DarkConv(ks.layers.Layer):
    def __init__(self,
                 filters=1,
                 kernel_size=(1, 1),
                 strides=(1, 1),
                 padding='same',
                 dilation_rate=(1, 1),
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 bias_regularizer=None,
                 l2_regularization=5e-4,  # default find where is it is stated
                 use_bn=True,
                 use_sync_bn=False,
                 norm_moment=0.99,
                 norm_epsilon=0.001,
                 activation='leaky',
                 leaky_alpha=0.1,
                 **kwargs):
        '''
        Modified Convolution layer to match that of the DarkNet Library

        Args:
            filters: integer for output depth, or the number of features to learn
            kernel_size: integer or tuple for the shape of the weight matrix or kernel to learn
            strides: integer of tuple how much to move the kernel after each kernel use
            padding: string 'valid' or 'same', if same, then pad the image, else do not
            dialtion_rate: tuple to indicate how much to modulate kernel weights and
                            the how many pixels ina featur map to skip
            use_bias: boolean to indicate wither to use bias in convolution layer
            kernel_initializer: string to indicate which function to use to initialize weigths
            bias_initializer: string to indicate which function to use to initialize bias
            l2_regularization: float to use as a constant for weight regularization
            use_bn: boolean for wether to use batchnormalization
            use_sync_bn: boolean for wether sync batch normalization statistics
                         of all batch norm layers to the models global statistics (across all input batches)
            norm_moment: float for moment to use for batchnorm
            norm_epsilon: float for batchnorm epsilon
            activation: string or None for activation function to use in layer,
                        if None activation is replaced by linear
            leaky_alpha: float to use as alpha if activation function is leaky
            **kwargs: Keyword Arguments

        '''

        # convolution params
        self._filters = filters
        self._kernel_size = kernel_size
        self._strides = strides
        self._padding = padding.upper()
        self._dilation_rate = dilation_rate
        self._use_bias = use_bias
        self._kernel_initializer = kernel_initializer
        self._bias_initializer = bias_initializer
        self._l2_regularization = l2_regularization
        self._bias_regularizer = bias_regularizer

        # batchnorm params
        self._use_bn = use_bn
        if self._use_bn:
            self._use_bias = False
        self._use_sync_bn = use_sync_bn
        self._norm_moment = norm_moment
        self._norm_epsilon = norm_epsilon

        if tf.keras.backend.image_data_format() == 'channels_last':
            # format: (batch_size, height, width, channels)
            self._bn_axis = -1
        else:
            # format: (batch_size, channels, width, height)
            self._bn_axis = 1

        # activation params
        if activation is None:
            self._activation = 'linear'
        else:
            self._activation = activation
        self._leaky_alpha = leaky_alpha

        super(DarkConv, self).__init__(**kwargs)
        return

    def build(self, input_shape):
        if isinstance(self._kernel_size, int): 
            self._kernel = self.add_weight(shape=[self._kernel_size,self._kernel_size,input_shape[-1], self._filters], dtype = tf.float32, initializer=self._kernel_initializer, regularizer = ks.regularizers.l2(self._l2_regularization), trainable=True) 
        else:
            self._kernel = self.add_weight(shape=[self._kernel_size[0],self._kernel_size[1],input_shape[-1], self._filters], dtype = tf.float32, initializer=self._kernel_initializer, regularizer = ks.regularizers.l2(self._l2_regularization), trainable=True) 
        
        if self._use_bias:
            self._bias = self.add_weight(shape = [self._filters], dtype = tf.float32, initializer=self._bias_initializer, regularizer=self._bias_regularizer, trainable=True)
        
        # ks.layers.Conv2D(
        #     filters=self._filters,
        #     kernel_size=self._kernel_size,
        #     strides=self._strides,
        #     padding=self._padding,
        #     dilation_rate=self._dilation_rate,
        #     use_bias=self._use_bias,
        #     kernel_initializer=self._kernel_initializer,
        #     bias_initializer=self._bias_initializer,
        #     kernel_regularizer=ks.regularizers.l2(self._l2_regularization),
        #     bias_regularizer=self._bias_regularizer)
        
        if self._use_bn:
            if self._use_sync_bn:
                self.bn = tf.keras.layers.experimental.SyncBatchNormalization(
                    momentum=self._norm_moment,
                    epsilon=self._norm_epsilon,
                    axis=self._bn_axis)
            else:
                self.bn = ks.layers.BatchNormalization(
                    momentum=self._norm_moment,
                    epsilon=self._norm_epsilon,
                    axis=self._bn_axis)

            # # batch norm adjustments to move input batches closer to zero mean and unit variance
            # self._gamma = self.add_weight(shape = [self._filters], dtype = tf.float32, initializer='zeros', regularizer= None, trainable=True) #scales
            # self._beta= self.add_weight(shape = [self._filters], dtype = tf.float32, initializer='zeros', regularizer= None, trainable=True) #beta

            # # batch norm target is target is zero mean and unit variance, so these will remain as constant, trainable = False
            # self._rolling_mean = self.add_weight(shape = [self._filters], dtype = tf.float32, initializer = 'zeros', regularizer= None, trainable=False) 
            # self._rolling_variance = self.add_weight(shape = [self._filters], dtype = tf.float32, initializer = 'ones', regularizer= None, trainable=False)
        

        if self._activation != 'leaky':
            self._activation_fn = ks.layers.Activation(activation=self._activation)
        else:
            self._activation_fn = ks.layers.LeakyReLU(alpha=self._leaky_alpha)

        self.conv_op = functools.partial(tf.nn.conv2d, {"strides": self._strides, "padding": self._padding, "name": "dark_conv_op"})
        
        super(DarkConv, self).build(input_shape)
        return

    
    def call(self, inputs):
        x = tf.nn.conv2d(inputs, self._kernel, self._strides, self._padding, dilations=None, name="dark_conv_op")
        if self._use_bias:
            x = tf.nn.bias_add(x, self._bias, name = "dark_conv_bias_op")

        if self._use_bn:
            x = self.bn(x)

        return self._activation_fn(x)

    # def _assign_moving_average(self, variable, value, momentum, inputs_size):
    #     with K.name_scope('AssignMovingAvg') as scope:
    #         decay = tf.convert_to_tensor(1.0 - momentum, name='decay')
    #         if decay.dtype != variable.dtype.base_dtype:
    #             decay = tf.cast(decay, variable.dtype.base_dtype)
    #             update_delta = (variable - tf.cast(value, variable.dtype)) * decay
    #         else:
    #             update_delta = (variable - tf.cast(value, variable.dtype)) * decay
    #         if inputs_size is not None:
    #             update_delta = tf.where(inputs_size > 0, update_delta, K.zeros_like(update_delta))
    #         return tf.compat.v1.assign_sub(variable, update_delta, name=scope)

    # def _batch_norm(self, x):
    #     input_size = tf.size(x)
    #     #x, mean, variance = tf.compat.v1.nn.fused_batch_norm(x, self._gamma, self._beta, self._rolling_mean, self._rolling_variance,  self._norm_epsilon, is_training=True)
    #     x = tf.nn.batch_normalization(x, self._rolling_mean, self._rolling_variance, self._gamma, self._beta, self._norm_epsilon)
        
    #     # if self.trainable:
    #     #     def update_mean():
    #     #         return self._assign_moving_average(self._rolling_mean, mean, self._norm_moment, input_size)

    #     #     def update_var():
    #     #         return self._assign_moving_average(self._rolling_variance, variance, self._norm_moment, input_size)

    #     #     self.add_update(update_mean)
    #     #     self.add_update(update_var)
    #     return x


    def get_config(self):
        # used to store/share parameters to reconsturct the model
        layer_config = {
            "filters": self._filters,
            "kernel_size": self._kernel_size,
            "strides": self._strides,
            "padding": self._padding,
            "dilation_rate": self._dilation_rate,
            "use_bias": self._use_bias,
            "kernel_initializer": self._kernel_initializer,
            "bias_initializer": self._bias_initializer,
            "bias_regularizer": self._bias_regularizer,
            "l2_regularization": self._l2_regularization,
            "use_bn": self._use_bn,
            "use_sync_bn": self._use_sync_bn,
            "norm_moment": self._norm_moment,
            "norm_epsilon": self._norm_epsilon,
            "activation": self._activation,
            "leaky_alpha": self._leaky_alpha
        }
        layer_config.update(super(DarkConv, self).get_config())
        return layer_config


@ks.utils.register_keras_serializable(package='yolo')
class DarkConv_nested(ks.layers.Layer):
    def __init__(self,
                 filters=1,
                 kernel_size=(1, 1),
                 strides=(1, 1),
                 padding='same',
                 dilation_rate=(1, 1),
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 bias_regularizer=None,
                 l2_regularization=5e-4,  # default find where is it is stated
                 use_bn=True,
                 use_sync_bn=False,
                 norm_moment=0.99,
                 norm_epsilon=0.001,
                 activation='leaky',
                 leaky_alpha=0.1,
                 **kwargs):
        '''
        Modified Convolution layer to match that of the DarkNet Library

        Args:
            filters: integer for output depth, or the number of features to learn
            kernel_size: integer or tuple for the shape of the weight matrix or kernel to learn
            strides: integer of tuple how much to move the kernel after each kernel use
            padding: string 'valid' or 'same', if same, then pad the image, else do not
            dialtion_rate: tuple to indicate how much to modulate kernel weights and
                            the how many pixels ina featur map to skip
            use_bias: boolean to indicate wither to use bias in convolution layer
            kernel_initializer: string to indicate which function to use to initialize weigths
            bias_initializer: string to indicate which function to use to initialize bias
            l2_regularization: float to use as a constant for weight regularization
            use_bn: boolean for wether to use batchnormalization
            use_sync_bn: boolean for wether sync batch normalization statistics
                         of all batch norm layers to the models global statistics (across all input batches)
            norm_moment: float for moment to use for batchnorm
            norm_epsilon: float for batchnorm epsilon
            activation: string or None for activation function to use in layer,
                        if None activation is replaced by linear
            leaky_alpha: float to use as alpha if activation function is leaky
            **kwargs: Keyword Arguments

        '''

        # convolution params
        self._filters = filters
        self._kernel_size = kernel_size
        self._strides = strides
        self._padding = padding
        self._dilation_rate = dilation_rate
        self._use_bias = use_bias
        self._kernel_initializer = kernel_initializer
        self._bias_initializer = bias_initializer
        self._l2_regularization = l2_regularization
        self._bias_regularizer = bias_regularizer

        # batchnorm params
        self._use_bn = use_bn
        if self._use_bn:
            self._use_bias = False
        self._use_sync_bn = use_sync_bn
        self._norm_moment = norm_moment
        self._norm_epsilon = norm_epsilon

        if tf.keras.backend.image_data_format() == 'channels_last':
            # format: (batch_size, height, width, channels)
            self._bn_axis = -1
        else:
            # format: (batch_size, channels, width, height)
            self._bn_axis = 1

        # activation params
        if activation is None:
            self._activation = 'linear'
        else:
            self._activation = activation
        self._leaky_alpha = leaky_alpha

        super(DarkConv, self).__init__(**kwargs)
        return

    def build(self, input_shape):
        self.conv = ks.layers.Conv2D(
            filters=self._filters,
            kernel_size=self._kernel_size,
            strides=self._strides,
            padding=self._padding,
            dilation_rate=self._dilation_rate,
            use_bias=self._use_bias,
            kernel_initializer=self._kernel_initializer,
            bias_initializer=self._bias_initializer,
            kernel_regularizer=ks.regularizers.l2(self._l2_regularization),
            bias_regularizer=self._bias_regularizer)
        
        #self.conv =tf.nn.convolution(filters=self._filters, strides=self._strides, padding=self._padding
        if self._use_bn:
            if self._use_sync_bn:
                self.bn = tf.keras.layers.experimental.SyncBatchNormalization(
                    momentum=self._norm_moment,
                    epsilon=self._norm_epsilon,
                    axis=self._bn_axis)
            else:
                self.bn = ks.layers.BatchNormalization(
                    momentum=self._norm_moment,
                    epsilon=self._norm_epsilon,
                    axis=self._bn_axis)
        else:
            self.bn = Identity()
        

        if self._activation != 'leaky':
            self._activation_fn = ks.layers.Activation(
                activation=self._activation)
        else:
            self._activation_fn = ks.layers.LeakyReLU(alpha=self._leaky_alpha)

        super(DarkConv, self).build(input_shape)
        return

    def call(self, inputs):
        x = self.conv(inputs)
        #if self._use_bn:
        x = self.bn(x)
        x = self._activation_fn(x)
        return x

    def get_config(self):
        # used to store/share parameters to reconsturct the model
        layer_config = {
            "filters": self._filters,
            "kernel_size": self._kernel_size,
            "strides": self._strides,
            "padding": self._padding,
            "dilation_rate": self._dilation_rate,
            "use_bias": self._use_bias,
            "kernel_initializer": self._kernel_initializer,
            "bias_initializer": self._bias_initializer,
            "bias_regularizer": self._bias_regularizer,
            "l2_regularization": self._l2_regularization,
            "use_bn": self._use_bn,
            "use_sync_bn": self._use_sync_bn,
            "norm_moment": self._norm_moment,
            "norm_epsilon": self._norm_epsilon,
            "activation": self._activation,
            "leaky_alpha": self._leaky_alpha
        }
        layer_config.update(super(DarkConv, self).get_config())
        return layer_config
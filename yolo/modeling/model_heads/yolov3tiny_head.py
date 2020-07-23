import tensorflow as tf
from yolo.modeling.building_blocks import _DarkConv
from yolo.modeling.building_blocks import _DarkUpsampleRoute

@ks.utils.register_keras_serializable(package='yolo')

class yolo3tiny_head(tf.keras.Model):
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
                 downsample=False,
                 **kwargs):

        # darkconv params
        self._filters = filters
        self._use_bias = use_bias
        self._kernel_initializer = kernel_initializer
        self._bias_initializer = bias_initializer
        self._use_bn = use_bn
        self._use_sync_bn = use_sync_bn

        # normal params
        self._norm_moment = norm_moment
        self._norm_epsilon = norm_epsilon

        # activation params
        self._conv_activation = conv_activation
        self._leaky_alpha = leaky_alpha
        self._sc_activation = sc_activation


        super().__init__(**kwargs)
        return

    def build(self, input_shape):

       self.convlayer1 = DarkConv(filters=256,
                                  kernel_size=(1, 1),
                                  strides=(1, 1),
                                  padding=1,
                                  use_bias=self._use_bias,
                                  kernel_initializer=self._kernel_initializer,
                                  bias_initializer=self._bias_initializer,
                                  use_bn=self._use_bn,
                                  use_sync_bn=self._use_sync_bn,
                                  norm_moment=self._norm_moment,
                                  norm_epsilon=self._norm_epsilon,
                                  activation=self._conv_activation,
                                  leaky_alpha=self._leaky_alpha)

       self.convlayer2 = DarkConv(filters=512,
                                  kernel_size=(3, 3),
                                  strides=(1, 1),
                                  padding=1,
                                  use_bias=self._use_bias,
                                  kernel_initializer=self._kernel_initializer,
                                  bias_initializer=self._bias_initializer,
                                  use_bn=self._use_bn,
                                  use_sync_bn=self._use_sync_bn,
                                  norm_moment=self._norm_moment,
                                  norm_epsilon=self._norm_epsilon,
                                  activation=self._conv_activation,
                                  leaky_alpha=self._leaky_alpha)  

       self.convlayer3 = DarkConv(filters=255,
                                  kernel_size=(1, 1),
                                  strides=(1, 1),
                                  padding=1,
                                  use_bias=self._use_bias,
                                  kernel_initializer=self._kernel_initializer,
                                  bias_initializer=self._bias_initializer,
                                  use_bn=False,
                                  use_sync_bn=self._use_sync_bn,
                                  norm_moment=self._norm_moment,
                                  norm_epsilon=self._norm_epsilon,
                                  activation=self._sc_activation,
                                  leaky_alpha=self._leaky_alpha)

       self.convlayer4 = DarkConv(filters=256,
                                  kernel_size=(3, 3),
                                  strides=(1, 1),
                                  padding=1,
                                  use_bias=self._use_bias,
                                  kernel_initializer=self._kernel_initializer,
                                  bias_initializer=self._bias_initializer,
                                  use_bn=self._use_bn,
                                  use_sync_bn=self._use_sync_bn,
                                  norm_moment=self._norm_moment,
                                  norm_epsilon=self._norm_epsilon,
                                  activation=self._conv_activation,
                                  leaky_alpha=self._leaky_alpha)  

       self.convlayer5 = DarkConv(filters=255,
                                  kernel_size=(1, 1),
                                  strides=(1, 1),
                                  padding=1,
                                  use_bias=self._use_bias,
                                  kernel_initializer=self._kernel_initializer,
                                  bias_initializer=self._bias_initializer,
                                  use_bn=False,
                                  use_sync_bn=self._use_sync_bn,
                                  norm_moment=self._norm_moment,
                                  norm_epsilon=self._norm_epsilon,
                                  activation=self._sc_activation,
                                  leaky_alpha=self._leaky_alpha)

        self.upsamplelayer = DarkUpsampleRoute(self,
                                                filters=128,
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
                                                **kwargs)

        super().build(input_shape)
        return

    def call(self, inputs):
        route0 = inputs[0]
        route1 = inputs[1]
        output01 = self.convlayer1(route0)
        output02 = self.convlayer2(output01)
        output03 = self.convlayer3(output02)
        output05 = self.upsamplelayer(output04, route0)
        output11 = self.convlayer4(route1)
        output12 = self.convlayer5(output11)
        concat_output = ks.layers.concatenate(outputs)
        return concat_output

    def get_config(self):
        layer_config = {
            "sizes": self._sizes
        }
        layer_config.update(super().get_config())
        return layer_config
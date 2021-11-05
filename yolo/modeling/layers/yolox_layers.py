import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Layer,Conv2D,LeakyReLU,\
    MaxPool2D,UpSampling2D,Activation,ReLU,BatchNormalization,concatenate



class SiLU(Layer):
    def __init__(self, *args, **kwargs):
        super(SiLU,self).__init__(*args, **kwargs)

    def call(self,x,**kwargs):
        return x*K.sigmoid(x)

    def get_config(self):
        config = super(SiLU, self).get_config()
        return config

    def compute_output_shape(self, input_shape):
        return input_shape

def get_activation(name="silu", ):
    if name == "silu":
        module = SiLU()
    elif name == "relu":
        module = ReLU()
    elif name == "lrelu":
        module = LeakyReLU(0.1 )
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module

class BaseConv(Layer):
    def __init__(self,filters, kernel_size, strides=1,padding='same',groups=1,use_bias=False, apply_batchnorm=True, act="silu"):
        super(BaseConv, self).__init__()
        self.conv=Conv2D(filters,kernel_size=kernel_size,padding=padding,strides=strides,groups=groups,
                         use_bias=use_bias,
                         )
        self.apply_batchnorm=apply_batchnorm
        self.bn=BatchNormalization(momentum=0.03,epsilon=1e-3)
        self.act=get_activation(act)

    def call(self,inputs,**kwargs):
        x=self.conv(inputs)
        if self.apply_batchnorm:
            x=self.bn(x)
        x=self.act(x)

        return x

class DWConv(Layer):
    """Depthwise Conv + Conv"""
    def __init__(self, filters, kernel_size,padding='same',apply_batchnorm=True, strides=1, act="silu"):
        super().__init__()
        self.dconv = BaseConv(
            filters, kernel_size,
            strides=strides,padding=padding,apply_batchnorm=apply_batchnorm, act=act
        )
        self.pconv = BaseConv(
            filters, kernel_size=1,padding=padding,apply_batchnorm=apply_batchnorm,
            strides=1, act=act
        )
    def call(self,inputs,**kwargs):
        x=self.dconv(inputs)
        x=self.pconv(x)
        return x

class Bottleneck(Layer):
    # Standard bottleneck
    def __init__(
        self,filters, shortcut=True,
        expansion=0.5, depthwise=False, act="silu"
    ):
        super().__init__()
        hidden_filters = int(filters * expansion)
        Conv = DWConv if depthwise else BaseConv
        self.conv1 = BaseConv( hidden_filters, 1, strides=1, act=act)
        self.conv2 = Conv(hidden_filters, 3, strides=1, act=act)
        self.use_add = shortcut

    def call(self, x,**kwargs):
        y = self.conv2(self.conv1(x))
        if self.use_add:
            y = y + x
        return y


class SPP(Layer):
    def __init__(self,filters,activation="silu",pool_sizes=(5,9,13),name="spp",**kwargs):
        super(SPP, self).__init__(name=name,**kwargs)
        self.conv1 = BaseConv(filters//2, 1, strides=1, act=activation)
        self.poolings=[MaxPool2D(pool_size=pool_size,strides=1,padding="same") for pool_size in pool_sizes]
        self.conv2 = BaseConv(filters, 1, strides=1, act=activation)

    def call(self,inputs,**kwargs):
        inputs=self.conv1(inputs)
        features=[max_pooling(inputs) for max_pooling in self.poolings]
        features=concatenate([inputs]+features)
        features=self.conv2(features)

        return features


class CSPLayer(Layer):
    def __init__(self,filters,n=1,shortcut=True,
                 expansion=0.5,depthwise=False,act="silu"):
        super(CSPLayer, self).__init__()
        hidden_fiters=int(filters*expansion)
        self.conv1=BaseConv(hidden_fiters,1,strides=1,act=act)
        self.conv2= BaseConv(hidden_fiters, 1, strides=1, act=act)
        self.conv3 = BaseConv(filters, 1, strides=1, act=act)
        module_list = [
            Bottleneck(hidden_fiters, shortcut, 1.0, depthwise, act=act)
            for _ in range(n)
        ]
        self.m = Sequential([*module_list])
    def call(self,x,**kwargs):
        x_1=self.conv1(x)
        x_2=self.conv2(x)
        x_1=self.m(x_1)
        x=concatenate([x_1,x_2])

        return self.conv3(x)


class Focus(Layer):
    """Focus width and height information into channel space."""

    def __init__(self,filters, kernel_size=1, strides=1, act="silu"):
        super().__init__()
        self.conv = BaseConv(filters, kernel_size, strides=strides, act=act)

    def call(self, x,**kwargs):
        # shape of x (b,w,h,c) -> y(b,w/2,h/2,4c)
        patch_top_left = x[:, ::2, ::2,:]
        patch_top_right = x[:, ::2, 1::2,:,]
        patch_bot_left = x[:, 1::2, ::2,:,]
        patch_bot_right = x[:, 1::2, 1::2,:,]
        x = concatenate(
            [patch_top_left, patch_bot_left, patch_top_right, patch_bot_right],axis=-1
        )
        return self.conv(x)


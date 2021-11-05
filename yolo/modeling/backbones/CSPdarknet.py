import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import Sequential
from yolo.modeling.layers.yolox_layers import DWConv,BaseConv,Focus,CSPLayer,SPP
class CSPDarknet(Layer):
    def __init__(self,
                 dep_mul,wid_mul,
                 out_features=("dark3","dark4","dark5"),
                 depthwise=False,act="silu"):
        super(CSPDarknet, self).__init__()
        self.out_features = out_features
        Conv = DWConv if depthwise else BaseConv
        base_filters = int(wid_mul * 64)
        base_depth = max(round(dep_mul * 3), 1)

        # stem
        self.stem=Focus(base_filters,kernel_size=3,act=act)

        # dark2
        self.dark2=Sequential([
            Conv(base_filters*2,kernel_size=3,strides=2,act=act),
            CSPLayer(base_filters*2,n=base_depth,depthwise=depthwise,act=act)]
        )

        # dark3
        self.dark3=Sequential([
            Conv(base_filters*4,3,2,act=act),
            CSPLayer(base_filters*4,n=base_depth*3,depthwise=depthwise,act=act)
        ])

        # dark4
        self.dark4 =Sequential([
            Conv(base_filters * 8, 3, 2, act=act),
            CSPLayer(
                base_filters * 8,
                n=base_depth * 3, depthwise=depthwise, act=act,
            ),]
        )

        # dark5
        self.dark5 = Sequential([
            Conv(base_filters * 16, 3, 2, act=act),
            SPP(base_filters*16, activation=act),
            CSPLayer(
                base_filters * 16, n=base_depth,
                shortcut=False, depthwise=depthwise, act=act,
            ),]
        )

    def call(self,x,**kwargs):
        outputs = {}
        x = self.stem(x)
        outputs["stem"] = x
        x = self.dark2(x)
        outputs["dark2"] = x
        x = self.dark3(x)
        outputs["dark3"] = x
        x = self.dark4(x)
        outputs["dark4"] = x
        x = self.dark5(x)
        outputs["dark5"] = x
        return {k: v for k, v in outputs.items() if k in self.out_features}




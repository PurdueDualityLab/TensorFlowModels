import tensorflow as tf
from tensorflow.keras.layers import Layer,Concatenate
from yolo.modeling.backbones.CSPdarknet import CSPDarknet
from yolo.modeling.layers.yolox_layers import BaseConv, CSPLayer, DWConv



class YOLOPAFPN(Layer):
    def __init__(self, depth=1.0, width=1.0, in_features=("dark3", "dark4", "dark5"),
                 filters=(256, 512, 1024), depthwise=False, act="relu"):
        super(YOLOPAFPN, self).__init__()
        self.backbone=CSPDarknet(depth,width,depthwise=depthwise,act=act)
        self.in_features=in_features
        self.filters=filters
        Conv=DWConv if depthwise else BaseConv
        self.upsample=tf.keras.layers.UpSampling2D(size=(2,2),interpolation='nearest')
        self.lateral_conv0=BaseConv(
             int(filters[1] * width), 1, 1, act=act
        )
        self.C3_p4 = CSPLayer(
            int(filters[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )
        self.reduce_conv1 = BaseConv(
           int(filters[0] * width), 1, 1, act=act
        )
        self.C3_p3 = CSPLayer(
            int(filters[0] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )
        # bottom-up conv
        self.down_conv2 = Conv(
           int(filters[0] * width), 3, 2, act=act
        )
        self.C3_n3 = CSPLayer(
            int(filters[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.down_conv1 = Conv(
         int(filters[1] * width), 3, 2, act=act
        )
        self.C3_n4 = CSPLayer(
            int(filters[2] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )


    def call(self, inputs, *args, **kwargs):
        out_features = self.backbone(inputs)
        features = [out_features[f] for f in self.in_features]
        [x2, x1, x0] = features

        fpn_out0 = self.lateral_conv0(x0)  # 1024->512/32
        f_out0 = self.upsample(fpn_out0)  # 512/16
        f_out0 = Concatenate()([f_out0, x1])  # 512->1024/16
        f_out0 = self.C3_p4(f_out0)  # 1024->512/16

        fpn_out1 = self.reduce_conv1(f_out0)  # 512->256/16
        f_out1 = self.upsample(fpn_out1)  # 256/8
        f_out1 = Concatenate()([f_out1, x2])  # 256->512/8
        pan_out2 = self.C3_p3(f_out1)  # 512->256/8

        p_out1 = self.down_conv2(pan_out2)  # 256->256/16
        p_out1 = Concatenate()([p_out1, fpn_out1])  # 256->512/16
        pan_out1 = self.C3_n3(p_out1)  # 512->512/16

        p_out0 = self.down_conv1(pan_out1)  # 512->512/32
        p_out0 = Concatenate()([p_out0, fpn_out0])  # 512->1024/32
        pan_out0 = self.C3_n4(p_out0)  # 1024->1024/32

        outputs = (pan_out2, pan_out1, pan_out0)
        return outputs


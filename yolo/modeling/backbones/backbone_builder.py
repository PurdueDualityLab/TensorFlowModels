import tensorflow as tf
import tensorflow.keras as ks

import yolo.modeling.building_blocks as nn_blocks
from yolo.modeling.backbones.get_config import build_block_specs

@ks.utils.register_keras_serializable(package='yolo')
class Backbone_Builder(ks.Model):
    def __init__(self, name):
        self._layer_dict = {"DarkRes": nn_blocks.DarkResidual,
                            "DarkUpsampleRoute": nn_blocks.DarkUpsampleRoute,
                            "DarkBlock": None,
                            }

        # parameters required for tensorflow to recognize ks.Model as not a
        # subclass of ks.Model
        self._model_name = name
        self._input_shape = (None, None, None, 3)

        layer_specs = self._get_config(name)
        if layer_specs is None:
            raise Exception("config file not found")

        inputs = ks.layers.Input(shape=self._input_shape[1:])
        output = self._build_struct(layer_specs, inputs)
        super(Backbone_Builder, self).__init__(inputs=inputs, outputs=output)
        return

    def _get_config(self, name):
        if name == "darknet53":
            from yolo.modeling.backbones.configs.darknet_53 import darknet53_config
            return build_block_specs(darknet53_config)
        else:
            return None

    def _build_struct(self, net, inputs):
        endpoints = dict()
        count = 0
        x = inputs
        for i, config in enumerate(net):
            x = self._build_block(config, x, f"{config.name}_{i}")
            if config.output:
                endpoints[f"route{count}"] = x
                count += 1
        return endpoints

    def _build_block(self, config, inputs, name):
        x = inputs
        for i in range(config.repititions):
            if config.name == "DarkConv":
                x = nn_blocks.DarkConv(
                    filters=config.filters,
                    kernel_size=config.kernel_size,
                    strides=config.strides,
                    padding=config.padding,
                    name = f"{name}_{i}")(x)
            elif config.name == "MaxPool":
                x = ks.layers.MaxPool2D(
                    pool_size=config.kernel_size,
                    strides=config.strides,
                    padding=config.padding,
                    name = f"{name}_{i}")(x)
            else:
                layer = self._layer_dict[config.name]
                x = layer(
                    filters=config.filters,
                    downsample=config.downsample,
                    name = f"{name}_{i}")(x)
        return x


m = Backbone_Builder("darknet53")
m.summary()

for layer in m.layers:
    print()
    print(layer)
    for weight in layer.get_weights():
        print(weight.shape)

# x = tf.ones(shape = [1, 416, 416, 3])

# p = ks.layers.Conv2D(3, kernel_size = (3,3), strides = (1,1), padding = "same")
# l = ks.layers.BatchNormalization()
# y = p(x)
# z = l(y)

# print(p.get_weights())
# print(l.get_weights())


import tensorflow as tf
import tensorflow.keras as ks

import yolo.modeling.building_blocks as nn_blocks
from yolo.modeling.backbones.get_config import build_block_specs

@ks.utils.register_keras_serializable(package='yolo')
class Backbone_Builder(ks.Model):
    def __init__(self, name, config = None, **kwargs):
        self._layer_dict = {"DarkRes": nn_blocks.DarkResidual,
                            "DarkUpsampleRoute": nn_blocks.DarkUpsampleRoute,
                            "DarkBlock": None,
                            }

        # parameters required for tensorflow to recognize ks.Model as not a
        # subclass of ks.Model
        self._model_name = name
        self._input_shape = (None, None, None, 3)

        layer_specs = self._get_model_config(name)
        if layer_specs is None:
            raise Exception("config file not found")

        inputs = ks.layers.Input(shape=self._input_shape[1:])
        output = self._build_struct(layer_specs, inputs)
        print(kwargs)
        super().__init__(inputs=inputs, outputs=output, name = self._model_name)
        return

    def _get_model_config(self, name):
        if name == "darknet53":
            from yolo.modeling.backbones.configs.darknet_53 import darknet53_config
            return build_block_specs(darknet53_config)
        elif name == "yolov3_tiny":
            from yolo.modeling.backbones.configs.yolov3_tiny import yolov3_tiny_config
            return build_block_specs(yolov3_tiny_config)
        else:
            return None

    def _build_struct(self, net, inputs):
        endpoints = dict()
        x = inputs
        for i, config in enumerate(net):
            x = self._build_block(config, x, f"{config.name}_{i}")
            if config.output:
                endpoints[int(config.filters)] = x
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
            elif config.name == "darkyolotiny":
                x = nn_blocks.darkyolotiny(
                    filters=config.filters,
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

# model = Backbone_Builder("darknet53")
# config = model.to_json()
# print(config)
# with tf.keras.utils.CustomObjectScope({'Backbone_Builder': Backbone_Builder}):
#     data = tf.keras.models.model_from_json(config)
#     print(data)

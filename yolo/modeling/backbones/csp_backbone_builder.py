import tensorflow as tf
import tensorflow.keras as ks

import importlib
import collections
from typing import *

import yolo.modeling.building_blocks as nn_blocks
from yolo.modeling.backbones.get_config import csp_build_block_specs
from . import configs


@ks.utils.register_keras_serializable(package='yolo')
class CSP_Backbone_Builder(ks.Model):
    def __init__(self,
                 name="darknet53",
                 input_shape=(None, None, None, 3),
                 weight_decay = 5e-4, 
                 config=None,
                 **kwargs):
        self._layer_dict = {"DarkRes": nn_blocks.DarkResidual}
        # parameters required for tensorflow to recognize ks.Model as not a
        # subclass of ks.Model
        self._input_shape = input_shape
        self._model_name = "custom_csp_backbone"
        layer_specs = config

        if not isinstance(config, Dict):
            self._model_name = name
            layer_specs = self.get_model_config(name)

        self._weight_decay = weight_decay
        inputs = ks.layers.Input(shape=self._input_shape[1:])
        output = self._build_struct(layer_specs, inputs)
        super().__init__(inputs=inputs, outputs=output, name=self._model_name)
        return

    def _build_struct(self, net, inputs):
        endpoints = collections.OrderedDict()
        stack_outputs = [inputs]
        for i, config in enumerate(net):
            if not config.stack:
                x = self._build_block(stack_outputs[config.route],
                                      config,
                                      name=f"{config.layer}_{i}")
                stack_outputs.append(x)
            else:
                x = self._csp_stack(stack_outputs[config.route],
                                    config,
                                    name=f"{config.layer}_{i}")
                stack_outputs.append(x)
            if config.output_name != None:
                endpoints[config.output_name] = x
        return endpoints

    def _build_block(self, inputs, config, name):
        x = inputs
        i = 0
        while i < config.repetitions:
            if config.layer == "DarkConv":
                x = nn_blocks.DarkConv(filters=config.filters,
                                       kernel_size=config.kernel_size,
                                       strides=config.strides,
                                       padding=config.padding,
                                       l2_regularization=self._weight_decay,
                                       name=f"{name}_{i}")(x)
            else:
                layer = self._layer_dict[config.name]
                x = layer(filters=config.filters,
                          downsample=config.downsample,
                          l2_regularization=self._weight_decay,
                          name=f"{name}_{i}")(x)
            i += 1
        return x

    def _csp_stack(self, inputs, config, name):
        if config.bottleneck:
            csp_filter_reduce = 1
            residual_filter_reduce = 2
            scale_filters = 1
        else:
            csp_filter_reduce = 2
            residual_filter_reduce = 1
            scale_filters = 2
        x, x_route = nn_blocks.CSPDownSample(filters=config.filters,
                                             filter_reduce=csp_filter_reduce,
                                             activation=config.activation,
                                             l2_regularization=self._weight_decay,
                                             name=f"{name}_csp_down")(inputs)
        for i in range(config.repetitions):
            x = self._layer_dict[config.layer](
                filters=config.filters // scale_filters,
                filter_scale=residual_filter_reduce,
                conv_activation=config.activation,
                l2_regularization=self._weight_decay,
                name=f"{name}_{i}")(x)
        output = nn_blocks.CSPConnect(filters=config.filters,
                                      filter_reduce=csp_filter_reduce,
                                      activation=config.activation,
                                      l2_regularization=self._weight_decay,
                                      name=f"{name}_csp_connect")([x, x_route])
        return output

    @staticmethod
    def get_model_config(name):
        # if name == "darknet53":
        #     name = "darknet_53"

        try:
            backbone = importlib.import_module(
                '.csp_' + name, package=configs.__package__).backbone
        except ModuleNotFoundError as e:
            if e.name == configs.__package__ + '.' + name:
                raise ValueError(f"Invlid backbone '{name}'") from e
            else:
                raise

        return csp_build_block_specs(backbone)


if __name__ == "__main__":
    model = CSP_Backbone_Builder(name="darknet53")
    model.summary()
    tf.keras.utils.plot_model(model,
                              to_file='CSPDarknet53.png',
                              show_shapes=True,
                              show_layer_names=True,
                              rankdir='TB',
                              expand_nested=True,
                              dpi=96)

    def print_weights(weights):
        shapes = []
        for weight in weights:
            shapes.append(weight.shape)
        return shapes

    for layer in model.layers:
        weights = layer.get_weights()
        print(f"{layer.name}: {print_weights(weights)}")

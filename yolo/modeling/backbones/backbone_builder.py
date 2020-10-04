import tensorflow as tf
import tensorflow.keras as ks

import importlib
import collections
from typing import *

import yolo.modeling.building_blocks as nn_blocks
from yolo.modeling.backbones.get_config import build_block_specs
from . import configs


@ks.utils.register_keras_serializable(package='yolo')
class Backbone_Builder(ks.Model):
    def __init__(self,
                 name,
                 input_shape=(None, None, None, 3),
                 config=None,
                 weight_decay = 0.005, 
                 **kwargs):
        self._layer_dict = {
            "DarkRes": nn_blocks.DarkResidual,
            "DarkUpsampleRoute": nn_blocks.DarkUpsampleRoute,
            "DarkBlock": None
        }

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

    @staticmethod
    def get_model_config(name):
        if name == "darknet53":
            name = "darknet_53"

        try:
            backbone = importlib.import_module(
                '.' + name, package=configs.__package__).backbone
        except ModuleNotFoundError as e:
            if e.name == configs.__package__ + '.' + name:
                raise ValueError(f"Invlid backbone '{name}'") from e
            else:
                raise

        return build_block_specs(backbone)

    def _build_struct(self, net, inputs):
        endpoints = collections.OrderedDict()  #dict()
        x = inputs
        for i, config in enumerate(net):
            x = self._build_block(config, x, f"{config.name}_{i}")
            if config.output:
                endpoints[config.output_name] = x

        return endpoints

    def _build_block(self, config, inputs, name):
        x = inputs
        i = 0
        while i < config.repititions:
            if config.name == "DarkConv":
                x = nn_blocks.DarkConv(filters=config.filters,
                                       kernel_size=config.kernel_size,
                                       strides=config.strides,
                                       padding=config.padding,
                                       l2_regularization=self._weight_decay
                                       name=f"{name}_{i}")(x)
            elif config.name == "darkyolotiny":
                x = nn_blocks.DarkTiny(filters=config.filters,
                                       strides=config.strides,
                                       name=f"{name}_{i}")(x)
            elif config.name == "MaxPool":
                x = ks.layers.MaxPool2D(pool_size=config.kernel_size,
                                        strides=config.strides,
                                        padding=config.padding,
                                        name=f"{name}_{i}")(x)
            else:
                layer = self._layer_dict[config.name]
                x = layer(filters=config.filters,
                          downsample=config.downsample,
                          l2_regularization=self._weight_decay
                          name=f"{name}_{i}")(x)
            i += 1
        return x

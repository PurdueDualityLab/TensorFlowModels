from __future__ import annotations

import tensorflow as tf
import tensorflow.keras as ks

import importlib

import yolo.modeling.building_blocks as nn_blocks
from yolo.modeling.backbones import get_config
from yolo.utils import tf_shims
from . import configs

from typing import List


@ks.utils.register_keras_serializable(package='yolo')
class Backbone_Builder(ks.Model):
    """
    The backbone builder handles the construction of the backbones of YOLO
    models. The current backbone types are listed below:

     - Darknet53 (`darknet53`, `darknet_53`): backbone used for the YOLOv3 model
     - YOLOv3-tiny (`yolov3_tiny`, `darknet_tiny`): backbone used for the YOLOv3 tiny model

    The YOLOv3-tiny is NOT the same as the Tiny Darknet model found here: https://pjreddie.com/darknet/tiny-darknet/.
    """
    _updated_config = tf_shims.ks_Model___updated_config
    def __init__(self, name: str, config:dict=None, **kwargs):
        self._layer_dict = {"DarkRes": nn_blocks.DarkResidual,
                            "DarkUpsampleRoute": nn_blocks.DarkUpsampleRoute,
                            "DarkBlock": None,
                            }

        # parameters required for tensorflow to recognize ks.Model as not a
        # subclass of ks.Model
        self._model_name = name
        self._input_shape = (None, None, None, 3)

        layer_specs = self.get_model_config(name)
        if layer_specs is None:
            raise Exception("config file not found")

        inputs = ks.layers.Input(shape=self._input_shape[1:])
        output = self._build_struct(layer_specs, inputs)
        super().__init__(inputs=inputs, outputs=output, name=self._model_name)
        return

    @staticmethod
    def get_model_config(name: str) -> List[get_config.RawConfig]:
        """
        Read the Spinenet style configuration dictionary for a backbone with a
        given name.

        Arguments:
            name: name of the backbone

        Returns:
            list of dictionaries containing the configuration associated with
            the name provided
        """
        if name == "darknet53":
            name = "darknet_53"

        try:
            backbone = importlib.import_module('.' + name, package=configs.__package__).backbone
        except ModuleNotFoundError as e:
            if e.name == configs.__package__ + '.' + name:
                raise ValueError(f"Invlid backbone '{name}'") from e
            else:
                raise

        return get_config.BlockConfig.from_dicts(backbone)

    def _build_struct(self, net, inputs):
        endpoints = dict()
        x = inputs
        for i, config in enumerate(net):
            x = self._build_block(config, x, f"{config.name}_{i}")
            if config.output:
                endpoints[int(config.filters)] = x

        #endpoints = {key:endpoints[key] for key in reversed(list(endpoints.keys()))}
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
                    name=f"{name}_{i}")(x)
            elif config.name == "darkyolotiny":
                x = nn_blocks.DarkTiny(
                    filters=config.filters,
                    strides=config.strides,
                    name=f"{name}_{i}")(x)
            elif config.name == "MaxPool":
                x = ks.layers.MaxPool2D(
                    pool_size=config.kernel_size,
                    strides=config.strides,
                    padding=config.padding,
                    name=f"{name}_{i}")(x)
            else:
                layer = self._layer_dict[config.name]
                x = layer(
                    filters=config.filters,
                    downsample=config.downsample,
                    name=f"{name}_{i}")(x)
        return x

# model = Backbone_Builder("darknet53")
# model.summary()
# print(config)
# with tf.keras.utils.CustomObjectScope({'Backbone_Builder': Backbone_Builder}):
#     data = tf.keras.models.model_from_json(config)
#     print(data)

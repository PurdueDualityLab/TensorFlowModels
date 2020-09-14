import tensorflow as tf
import tensorflow.keras as ks

import importlib
import collections

import yolo.modeling.building_blocks as nn_blocks
from yolo.modeling.backbones.get_config import build_block_specs
from yolo.utils import tf_shims
from . import configs


@ks.utils.register_keras_serializable(package='yolo')
class Backbone_Builder(ks.Model):
    _updated_config = tf_shims.ks_Model___updated_config
    def __init__(self, name, input_shape = (None, None, None, 3), config=None, **kwargs):
        self._layer_dict = {"DarkRes": nn_blocks.DarkResidual,
                            "DarkUpsampleRoute": nn_blocks.DarkUpsampleRoute,
                            "DarkBlock": None}
        # parameters required for tensorflow to recognize ks.Model as not a
        # subclass of ks.Model
        self._model_name = name
        self._input_shape = input_shape

        if config is None:
            layer_specs = self.get_model_config(name)
        else:
            layer_specs = config

        inputs = ks.layers.Input(shape=self._input_shape[1:])
        output = self._build_struct(layer_specs, inputs)
        super().__init__(inputs=inputs, outputs=output, name=self._model_name)
        return

    @staticmethod
    def get_model_config(name):
        if name == "darknet53":
            name = "darknet_53"

        try:
            backbone = importlib.import_module('.' + name, package=configs.__package__).backbone
        except ModuleNotFoundError as e:
            if e.name == configs.__package__ + '.' + name:
                raise ValueError(f"Invlid backbone '{name}'") from e
            else:
                raise

        return build_block_specs(backbone)

    def _build_struct(self, net, inputs):
        endpoints = collections.OrderedDict()#dict()
        x = inputs
        for i, config in enumerate(net):
            x = self._build_block(config, x, f"{config.name}_{i}")
            if config.output:
                endpoints[config.output_name] = x
        return endpoints
    
    def _csp_block():
        return 
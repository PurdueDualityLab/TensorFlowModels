import tensorflow as tf
import tensorflow.keras as ks
from yolo.modeling.building_blocks import DarkConv
from yolo.modeling.building_blocks import DarkRouteProcess
from yolo.modeling.building_blocks import DarkUpsampleRoute
# for testing
from yolo.modeling.backbones.backbone_builder import Backbone_Builder

from . import configs

import importlib

#@ks.utils.register_keras_serializable(package='yolo')
class Yolov3Head(tf.keras.Model):
    def __init__(self, model="regular", classes=80, boxes=9, cfg_dict = None, **kwargs):
        self._cfg_dict = cfg_dict
        self._classes = classes
        self._boxes = boxes
        self._layer_dict = {"routeproc": DarkRouteProcess,
                            "upsampleroute": DarkUpsampleRoute}

        self._cfg_dict = self.load_dict_cfg(model)
        self._layer_keys = list(self._cfg_dict.keys())
        self._conv_depth = boxes//len(self._layer_keys) * (classes + 5)

        inputs, input_shapes, routes, upsamples, prediction_heads = self._get_attributes()
        outputs = self._connect_layers(routes, upsamples, prediction_heads, inputs)
        super().__init__(inputs=inputs, outputs=outputs, name=model, **kwargs)
        self._input_shape = input_shapes
        return

    @staticmethod
    def load_dict_cfg(model):
        try:
            return importlib.import_module('.yolov3_' + model, package=configs.__package__).head
        except ModuleNotFoundError as e:
            if e.name == configs.__package__ + '.yolov3_' + model:
                raise ValueError(f"Invlid backbone '{name}'") from e
            else:
                raise

    def _get_attributes(self):
        inputs = dict()
        input_shapes = dict()
        routes = dict()
        upsamples = dict()
        prediction_heads = dict()

        for key in self._layer_keys:
            path_keys = self._cfg_dict[key]

            inputs[key] = ks.layers.Input(shape=[None, None, path_keys["depth"]])
            input_shapes[key] = tf.TensorSpec([None, None, None, path_keys["depth"]])

            if type(path_keys["upsample"]) != type(None):
                args = path_keys["upsample_conditions"]
                layer = path_keys["upsample"]
                upsamples[key] = layer(**args)

            args = path_keys["processor_conditions"]
            layer = path_keys["processor"]
            routes[key] = layer(**args)

            args = path_keys["output_conditions"]
            prediction_heads[key] = DarkConv(filters=self._conv_depth + path_keys["output-extras"],**args)
        return inputs, input_shapes, routes, upsamples, prediction_heads

    def _connect_layers(self, routes, upsamples, prediction_heads, inputs):
        outputs = dict()
        layer_in = inputs[self._layer_keys[0]]
        for i in range(len(self._layer_keys)):
            x = routes[self._layer_keys[i]](layer_in)
            if i + 1 < len(self._layer_keys):
                x_next = inputs[self._layer_keys[i + 1]]
                layer_in = upsamples[self._layer_keys[i + 1]]([x[0], x_next])

            if type(x) == list or type(x) == tuple:
                outputs[self._layer_keys[i]] = prediction_heads[self._layer_keys[i]](x[1])
            else:
                outputs[self._layer_keys[i]] = prediction_heads[self._layer_keys[i]](x)
        return outputs

layer = Yolov3Head()

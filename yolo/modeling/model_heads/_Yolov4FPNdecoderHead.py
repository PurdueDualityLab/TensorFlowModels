import tensorflow as tf
import tensorflow.keras as ks
from typing import *
from yolo.modeling.building_blocks import DarkConv
from yolo.modeling.building_blocks import DarkRouteProcess
from yolo.modeling.building_blocks import DarkUpsampleRoute
# for testing
from ._Yolov4Neck import Yolov4Neck

from . import configs

import importlib
import more_itertools
import collections


@ks.utils.register_keras_serializable(package='yolo')
class Yolov4Head(tf.keras.Model):
    def __init__(self,
                 classes=80,
                 boxes_per_path=3,
                 min_level = 3, 
                 max_level = 5,
                 minimum_channels = 256, 
                 path_process_depth = 3, 
                 weight_decay = None,
                 **kwargs):


        self._classes = classes
        self._boxes = boxes_per_path
        self._weight_decay = weight_decay
        self._min_level = min_level
        self._max_level = max_level
        self._path_process_depth = 3
        self._minimum_channels = minimum_channels
        self._conv_depth = boxes_per_path * (classes + 5)

        print("WARNING: default anchor boxes may not work for this model")

        inputs, input_shapes, routes, resamples, prediction_heads = self._get_attributes()
        self._input_shape = input_shapes
        outputs = self._connect_layers(routes, resamples, prediction_heads, inputs)
        super().__init__(inputs=inputs,
                         outputs=outputs,
                         name=self._model_name,
                         **kwargs)
        return


    #TODO: not complete need to finish
    def _get_attributes(self, input_shape):
        """ use config dictionary to generate all important attributes for head construction """
        inputs = dict()
        routes = dict()
        resamples = dict()
        prediction_heads = dict()

        depth = self._minimum_channels

        for i, (key, path_keys) in enumerate(self._cfg_dict.items()):
            inputs[key] = ks.layers.Input(shape=[None, None, depth])
            input_shapes[key] = tf.TensorSpec(
                [None, start_width, start_height, path_keys["depth"]])

            if type(path_keys["resample"]) != type(None):
                args = path_keys["resample_conditions"]
                args['weight_decay'] = self._weight_decay
                layer = ks.utils.get_registered_object(path_keys["resample"])
                resamples[key] = layer(**args)

            args = path_keys["processor_conditions"].copy()
            args['weight_decay'] = self._weight_decay
            layer = ks.utils.get_registered_object(path_keys["processor"])
            # print(path_keys["processor"], ks.utils.get_registered_object(path_keys["processor"]))
            routes[key] = layer(**args)

            args = path_keys["output_conditions"].copy()
            args['weight_decay'] = self._weight_decay
            prediction_heads[key] = DarkConv(filters=self._conv_depth +
                                             path_keys["output-extras"],
                                             **args)
            depth *= 2

        return inputs, input_shapes, routes, resamples, prediction_heads

    def _connect_layers(self, routes, resamples, prediction_heads, inputs):
        """ connect all attributes the yolo way, if you want a different method of construction use something else """
        outputs = dict() #collections.OrderedDict()
        layer_keys = list(self._cfg_dict.keys())
        layer_in = inputs[layer_keys[0]]  # layer input to the next layer

        # print({key:inputs[key].shape for key in inputs.keys()})
        # using this loop is faster for some reason
        i = 0
        while i < len(layer_keys):
            x = routes[layer_keys[i]](layer_in)
            if i + 1 < len(layer_keys):
                x_next = inputs[layer_keys[i + 1]]
                if type(x) == tuple or type(x) == list:
                    layer_in = resamples[layer_keys[i + 1]]([x[0], x_next])
                else:
                    layer_in = resamples[layer_keys[i + 1]]([layer_in, x_next])

            if type(x) == tuple or type(x) == list:
                outputs[layer_keys[i]] = prediction_heads[layer_keys[i]](x[1])
            else:
                outputs[layer_keys[i]] = prediction_heads[layer_keys[i]](x)
            i += 1
        # print({key:outputs[key].shape for key in outputs.keys()})
        return outputs

    def get_config(self):
        layer_config = {
            "cfg_dict": self._cfg_dict,
            "classes": self._classes,
            "boxes": self._boxes,
            "model": self._model_name
        }
        layer_config.update(super().get_config())
        return layer_config


if __name__ == "__main__":
    #might be missing a layer?
    backbone = CSP_Backbone_Builder(input_shape=[1, 608, 608, 3])
    neck = Yolov4Neck(input_shape=[1, 608, 608, 3])
    head = Yolov4Head()
    inputs = tf.ones(shape=[1, 608, 608, 3], dtype=tf.float32)

    x = backbone(inputs)
    y = neck(x)
    z = head(y)

    head.summary()

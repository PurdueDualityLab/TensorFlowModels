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
                 model="regular",
                 classes=80,
                 boxes=9,
                 cfg_dict=None,
                 input_shape=(None, None, None, 3),
                 weight_decay=None,
                 **kwargs):
        """
        construct a detection head for an arbitrary back bone following the Yolo style

        config format:
            back bones can out put a head with many outputs that will be processed,
            for each backbone output yolo makes predictions for objects and N boxes

            back bone our pur should be a dictionary, that is what allow this config style to work

            {<backbone_output_name>:{
                                        "depth":<number of output channels>,
                                        "upsample":None or the name of a layer takes in 2 tensors and returns 1,
                                        "upsample_conditions":{dict conditions for layer above},
                                        "processor": Name of layer that will process output with this key and return 2 tensors,
                                        "processor_conditions": {dict conditions for layer above},
                                        "output_conditions": {dict conditions for detection or output layer},
                                        "output-extras": integer for the number of things to predict in addiction to the
                                                         4 items for  the bounding box, so maybe you want to do pose estimation,
                                                         and need the head to output 10 addictional values, place that here and for
                                                         each bounding box, we will predict 10 more values, be sure to modify the loss
                                                         function template to handle this modification, and calculate a loss for the
                                                         additional values.
                                    }
                ...
                <backbone_output_name>:{ ... }
            }

        Args:
            model: to generate a standard yolo head, we have 3 string key words that can accomplish this
                    regular -> corresponds to yolov3
                    spp -> corresponds to yolov3-spp
                    tiny -> corresponds to yolov3-tiny

                if you construct a custom backbone config, name it as follows:
                    yolov3_<name>.py and we will be able to find the model automaticially
                    in this case model corresponds to the value of <name>

            classes: integer for the number of classes in the prediction
            boxes: integer for the total number of anchor boxes, this will be devided by the number of paths
                   in the detection head config
            cfg_dict: dict, suppose you do not have the model_head saved config file in the configs folder,
                      you can provide us with a dictionary that will be used to generate the model head,
                      be sure to follow the correct format.


        """
        self._cfg_dict = cfg_dict
        self._classes = classes
        self._boxes = boxes
        self._weight_decay = weight_decay

        if not isinstance(self._cfg_dict, Dict):
            self._model_name = model
            self._cfg_dict = self.load_dict_cfg(model)
        else:
            self._model_name = "custom_head"

        self._conv_depth = boxes // len(self._cfg_dict) * (classes + 5)

        inputs, input_shapes, routes, resamples, prediction_heads = self._get_attributes(
            input_shape)
        self._input_shape = input_shapes
        outputs = self._connect_layers(routes, resamples, prediction_heads,
                                       inputs)
        super().__init__(inputs=inputs,
                         outputs=outputs,
                         name=self._model_name,
                         **kwargs)
        return

    @classmethod
    def load_dict_cfg(clz, model):
        """ find the config file and load it for use"""
        try:
            return importlib.import_module('.yolov4_' + model,
                                           package=configs.__package__).head
        except ModuleNotFoundError as e:
            if e.name == configs.__package__ + '.yolov4_' + model:
                raise ValueError(f"Invlid head '{model}'") from e
            else:
                raise

    def _get_attributes(self, input_shape):
        """ use config dictionary to generate all important attributes for head construction """
        inputs = collections.OrderedDict()
        input_shapes = collections.OrderedDict()
        routes = collections.OrderedDict()
        resamples = collections.OrderedDict()
        prediction_heads = collections.OrderedDict()

        start_width = input_shape[1]
        if input_shape[1] != None:
            start_width = start_width // 8

        start_height = input_shape[2]
        if input_shape[2] != None:
            start_height = start_height // 8

        for i, (key, path_keys) in enumerate(self._cfg_dict.items()):
            inputs[key] = ks.layers.Input(
                shape=[start_width, start_height, path_keys["depth"]])
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

            if start_width != None:
                start_width //= 2
            if start_height != None:
                start_height //= 2

        # print(routes)
        # print(resamples)
        return inputs, input_shapes, routes, resamples, prediction_heads

    def _connect_layers(self, routes, resamples, prediction_heads, inputs):
        """ connect all attributes the yolo way, if you want a different method of construction use something else """
        outputs = dict()  #collections.OrderedDict()
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

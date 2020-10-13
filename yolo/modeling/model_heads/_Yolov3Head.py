import tensorflow as tf
import tensorflow.keras as ks
from typing import *

from yolo.modeling.building_blocks import DarkConv
from yolo.modeling.building_blocks import DarkRouteProcess
from yolo.modeling.building_blocks import DarkUpsampleRoute
# for testing
from yolo.modeling.backbones.backbone_builder import Backbone_Builder

from . import configs

import importlib
import more_itertools
import collections


#@ks.utils.register_keras_serializable(package='yolo')
class Yolov3Head(tf.keras.Model):
    def __init__(self,
                 model="regular",
                 classes=80,
                 boxes=9,
                 cfg_dict=None,
                 input_shape=(None, None, None, 3),
                 weight_decay = 5e-4,
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
        self._model_name = model
        self._weight_decay = weight_decay

        if not isinstance(self._cfg_dict, Dict):
            self._model_name = model
            self._cfg_dict = self.load_dict_cfg(model)
        else:
            self._model_name = "custom_head"

        self._conv_depth = boxes // len(self._cfg_dict) * (classes + 5)

        inputs, input_shapes, routes, upsamples, prediction_heads = self._get_attributes(
            input_shape)
        self._input_shape = input_shapes
        outputs = self._connect_layers(routes, upsamples, prediction_heads,
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
            return importlib.import_module('.yolov3_' + model,
                                           package=configs.__package__).head
        except ModuleNotFoundError as e:
            if e.name == configs.__package__ + '.yolov3_' + model:
                raise ValueError(f"Invlid head '{model}'") from e
            else:
                raise

    def _get_attributes(self, input_shape):
        """ use config dictionary to generate all important attributes for head construction """
        inputs = collections.OrderedDict()  #dict()
        input_shapes = collections.OrderedDict()  #dict()
        routes = collections.OrderedDict()  #dict()
        upsamples = collections.OrderedDict()  #dict()
        prediction_heads = collections.OrderedDict()  #dict()

        start_width = input_shape[1]
        if input_shape[1] != None:
            start_width = start_width // 32

        start_height = input_shape[2]
        if input_shape[2] != None:
            start_height = start_height // 32

        for i, (key, path_keys) in enumerate(self._cfg_dict.items()):
            inputs[key] = ks.layers.Input(
                shape=[start_width, start_height, path_keys["depth"]])
            input_shapes[key] = tf.TensorSpec(
                [None, start_width, start_height, path_keys["depth"]])

            if type(path_keys["upsample"]) != type(None):
                args = path_keys["upsample_conditions"].copy()
                args['l2_regularization'] = self._weight_decay
                layer = ks.utils.get_registered_object(path_keys["upsample"])
                upsamples[key] = layer(**args)

            args = path_keys["processor_conditions"].copy()
            args['l2_regularization'] = self._weight_decay
            layer = ks.utils.get_registered_object(path_keys["processor"])
            routes[key] = layer(**args)

            args = path_keys["output_conditions"]
            args['l2_regularization'] = self._weight_decay
            prediction_heads[key] = DarkConv(filters=self._conv_depth +
                                             path_keys["output-extras"],
                                             **args)

            if start_width != None:
                start_width *= 2
            if start_height != None:
                start_height *= 2

        return inputs, input_shapes, routes, upsamples, prediction_heads

    def _connect_layers(self, routes, upsamples, prediction_heads, inputs):
        """ connect all attributes the yolo way, if you want a different method of construction use something else """
        outputs = dict() #collections.OrderedDict()
        layer_keys = list(self._cfg_dict.keys())
        layer_in = inputs[layer_keys[0]]

        i = 0
        while i < len(layer_keys):
            x = routes[layer_keys[i]](layer_in)
            if i + 1 < len(layer_keys):
                x_next = inputs[layer_keys[i + 1]]
                layer_in = upsamples[layer_keys[i + 1]]([x[0], x_next])

            if type(x) == tuple or type(x) == list:
                outputs[layer_keys[i]] = prediction_heads[layer_keys[i]](x[1])
            else:
                outputs[layer_keys[i]] = prediction_heads[layer_keys[i]](x)
            i += 1
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

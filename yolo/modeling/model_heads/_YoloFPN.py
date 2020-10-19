import tensorflow as tf
import tensorflow.keras as ks
from typing import *
from yolo.modeling.building_blocks import DarkConv
from yolo.modeling.building_blocks import DarkRouteProcess
from yolo.modeling.building_blocks import DarkRoute
# for testing


from . import configs

import importlib
import more_itertools
import collections


# @ks.utils.register_keras_serializable(package='yolo')
# generic yolo FPN do not use yet
class YoloFPN(tf.keras.Model):
    def __init__(self,
                 input_depth = 1024, 
                 fpn_path_len= 2, 
                 min_level = 3, 
                 max_level = 5, 
                 weight_decay = None,
                 activation = "leaky", 
                 **kwargs):
        # at the largest level insert a spp no up sampling
        # each subsequent level get 2 reps
        # the tailing conditions on the last path is half standard 
        self._input_depth = input_depth
        self._fpn_path_len = fpn_path_len
        self._min_level = min_level
        self._max_level = max_level
        self._weight_decay = weight_decay
        self._activation = activation

        inputs, preprocessors, resamples, tails = self._get_attributes()
        outputs = self._connect_layers(preprocessors, resamples, tails, inputs)
        super().__init__(inputs=inputs, outputs=outputs, **kwargs)
        return

    def _standard_block(self, filters):
        def block(inputs):
            x_route = DarkConv(filters=filters // 2,
                               kernel_size=(1, 1),
                               strides=(1, 1),
                               padding="same",
                               activation=self._activation,
                               weight_decay=self._weight_decay)(inputs)
            x = DarkConv(filters=filters // 4,
                         kernel_size=(1, 1),
                         strides=(1, 1),
                         padding="same",
                         activation=self._activation,
                         weight_decay=self._weight_decay)(x_route)
            x = ks.layers.UpSampling2D(size=2)(x)
            return x_route, x
        return block

    def _half_standard_block(self, filters):
        def block(inputs):
            x = DarkConv(filters=filters // 2,
                         kernel_size=(1, 1),
                         strides=(1, 1),
                         padding="same",
                         activation=self._activation,
                         weight_decay=self._weight_decay)(inputs)
            return x, None
        return block

    def _get_attributes(self):
        """ use config dictionary to generate all important attributes for head construction """
        inputs = {}
        upsamples = {}
        preprocessors = {}
        tails = {}
        depth = self._input_depth
        for i in reversed(range(self._min_level, self._max_level + 1)):
            print(i, depth)
            inputs[i] = ks.layers.Input(shape=[None, None, depth])
            if i != self._max_level: 
                upsamples[i] = DarkRoute(filters=depth//2)
            if i == self._max_level: 
                preprocessors[i] = DarkRouteProcess(filters = depth, repetitions=self._fpn_path_len + 1, insert_spp=True)
            else: 
                preprocessors[i] = DarkRouteProcess(filters = depth, repetitions=self._fpn_path_len, insert_spp=False)

            if i == self._min_level: 
                tails[i] = self._half_standard_block(depth)
            else: 
                tails[i] = self._standard_block(depth)
            depth //= 2
        return inputs, preprocessors, upsamples, tails
    
    def _connect_layers(self, routes, resamples, tails, inputs):
        outputs = {}
        layer_in = inputs[self._max_level]
        for i in reversed(range(self._min_level, self._max_level + 1)):
            _, x = routes[i](layer_in)
            x_route, x = tails[i](x)
            outputs[i] = x_route

            if i > self._min_level:
                x_next = inputs[i - 1]
                layer_in = resamples[i - 1]([x_next, x])
        return outputs

    


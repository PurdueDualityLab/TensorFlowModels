import importlib
import collections
from typing import *
import tensorflow as tf
import tensorflow.keras as ks
import yolo.modeling.building_blocks as nn_blocks
from official.vision.beta.modeling.backbones import factory
from . import configs


class CSPBlockConfig(object):
    def __init__(self, layer, stack, reps, bottleneck, filters, kernel_size,
                 strides, padding, activation, route, output_name, is_output):
        '''
        get layer config to make code more readable

        Args:
            layer: string layer name
            reps: integer for the number of times to repeat block
            filters: integer for the filter for this layer, or the output depth
            kernel_size: integer or none, if none, it implies that the the building block handles this automatically. not a layer input
            downsample: boolean, to down sample the input width and height
            output: boolean, true if the layer is required as an output
        '''
        self.layer = layer
        self.stack = stack
        self.repetitions = reps
        self.bottleneck = bottleneck
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.route = route
        self.output_name = output_name
        self.is_output = is_output
        return


def csp_build_block_specs(config, min_size, max_size):
    specs = []
    for layer in config:
        specs.append(CSPBlockConfig(*layer))
    return specs


def darkconv_config_todict(config, kwargs):
    dictvals = {
        "filters": config.filters,
        "kernel_size": config.kernel_size,
        "strides": config.strides,
        "padding": config.padding
    }
    dictvals.update(kwargs)
    return dictvals


def darktiny_config_todict(config, kwargs):
    dictvals = {"filters": config.filters, "strides": config.strides}
    dictvals.update(kwargs)
    return dictvals


def maxpool_config_todict(config, kwargs):
    return {
        "pool_size": config.kernel_size,
        "strides": config.strides,
        "padding": config.padding,
        "name": kwargs["name"]
    }


class layer_registry(object):
    def __init__(self):
        self._layer_dict = {
            "DarkTiny": (nn_blocks.DarkTiny, darktiny_config_todict),
            "DarkConv": (nn_blocks.DarkConv, darkconv_config_todict),
            "MaxPool": (tf.keras.layers.MaxPool2D, maxpool_config_todict)
        }
        return

    def _get_layer(self, key):
        return self._layer_dict[key]

    def __call__(self, config, kwargs):
        layer, get_param_dict = self._get_layer(config.layer)
        param_dict = get_param_dict(config, kwargs)
        return layer(**param_dict)


@ks.utils.register_keras_serializable(package='yolo')
class Darknet(ks.Model):
    def __init__(self,
                 model_id="darknet53",
                 input_shape=(None, None, None, 3),
                 min_size=None,
                 max_size=5,
                 activation=None,
                 use_sync_bn=False,
                 norm_momentum=0.99,
                 norm_epsilon=0.001,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 config=None,
                 **kwargs):

        self._model_name = "custom_csp_backbone"
        self._splits = None
        layer_specs = config

        if not isinstance(config, Dict):
            self._model_name = model_id.lower()
            layer_specs, splits = self.get_model_config(
                self._model_name, min_size, max_size)
            self._splits = splits

        if isinstance(input_shape, List) or isinstance(input_shape, Tuple):
            if len(input_shape) == 4:
                self._input_shape = tf.keras.layers.InputSpec(
                    shape=input_shape)
            else:
                self._input_shape = tf.keras.layers.InputSpec(
                    shape=[None] + list(input_shape))
        else:
            self._input_shape = input_shape

        # default layer look up
        self._min_size = min_size
        self._max_size = max_size
        self._registry = layer_registry()
        self._output_specs = None

        self._kernel_initializer = kernel_initializer
        self._bias_regularizer = bias_regularizer
        self._norm_momentum = norm_momentum
        self._norm_epislon = norm_epsilon
        self._use_sync_bn = use_sync_bn
        self._activation = activation
        self._kernel_regularizer = kernel_regularizer

        self._default_dict = {
            "kernel_initializer": self._kernel_initializer,
            "kernel_regularizer": self._kernel_regularizer,
            "bias_regularizer": self._bias_regularizer,
            "norm_momentum": self._norm_momentum,
            "norm_epsilon": self._norm_epislon,
            "use_sync_bn": self._use_sync_bn,
            "activation": self._activation,
            "name": None
        }

        inputs = ks.layers.Input(shape=self._input_shape.shape[1:])
        output = self._build_struct(layer_specs, inputs)

        super().__init__(inputs=inputs, outputs=output, name=self._model_name)
        return

    #TODO: build the model from arbitrary level numbers
    def _build_contiguous_struct(self, net, inputs):
        endpoints = collections.OrderedDict()
        stack_outputs = [inputs]

        return

    def _build_struct(self, net, inputs):
        endpoints = collections.OrderedDict()
        stack_outputs = [inputs]
        for i, config in enumerate(net):
            if config.stack == None:
                x = self._build_block(stack_outputs[config.route],
                                      config,
                                      name=f"{config.layer}_{i}")
                stack_outputs.append(x)
            elif config.stack == "residual":
                x = self._residual_stack(stack_outputs[config.route],
                                         config,
                                         name=f"{config.layer}_{i}")
                stack_outputs.append(x)
            elif config.stack == "csp":
                x = self._csp_stack(stack_outputs[config.route],
                                    config,
                                    name=f"{config.layer}_{i}")
                stack_outputs.append(x)
            elif config.stack == "csp_tiny":
                x_pass, x = self._tiny_stack(stack_outputs[config.route],
                                             config,
                                             name=f"{config.layer}_{i}")
                stack_outputs.append(x_pass)
            if (config.is_output and self._min_size
                    == None):  # or isinstance(config.output_name, str):
                endpoints[str(config.output_name)] = x
            elif self._min_size != None and config.output_name >= self._min_size and config.output_name <= self._max_size:
                endpoints[str(config.output_name)] = x

        self._output_specs = {
            l: endpoints[l].get_shape()
            for l in endpoints.keys()
        }
        return endpoints

    def _get_activation(self, activation):
        if self._activation == None:
            return activation
        else:
            return self._activation

    def _csp_stack(self, inputs, config, name):
        if config.bottleneck:
            csp_filter_reduce = 1
            residual_filter_reduce = 2
            scale_filters = 1
        else:
            csp_filter_reduce = 2
            residual_filter_reduce = 1
            scale_filters = 2
        self._default_dict["activation"] = self._get_activation(
            config.activation)
        self._default_dict["name"] = f"{name}_csp_down"
        x, x_route = nn_blocks.CSPDownSample(filters=config.filters,
                                             filter_reduce=csp_filter_reduce,
                                             **self._default_dict)(inputs)
        for i in range(config.repetitions):
            self._default_dict["name"] = f"{name}_{i}"
            x = nn_blocks.DarkResidual(filters=config.filters // scale_filters,
                                       filter_scale=residual_filter_reduce,
                                       **self._default_dict)(x)

        self._default_dict["name"] = f"{name}_csp_connect"
        output = nn_blocks.CSPConnect(filters=config.filters,
                                      filter_reduce=csp_filter_reduce,
                                      **self._default_dict)([x, x_route])
        self._default_dict["activation"] = self._activation
        self._default_dict["name"] = None
        return output

    def _tiny_stack(self, inputs, config, name):
        self._default_dict["activation"] = self._get_activation(
            config.activation)
        self._default_dict["name"] = f"{name}_tiny"
        x, x_route = nn_blocks.CSPTiny(filters=config.filters,
                                       **self._default_dict)(inputs)
        self._default_dict["activation"] = self._activation
        self._default_dict["name"] = None
        return x, x_route

    def _residual_stack(self, inputs, config, name):
        self._default_dict["activation"] = self._get_activation(
            config.activation)
        self._default_dict["name"] = f"{name}_residual_down"
        x = nn_blocks.DarkResidual(filters=config.filters,
                                   downsample=True,
                                   **self._default_dict)(inputs)
        for i in range(config.repetitions - 1):
            self._default_dict["name"] = f"{name}_{i}"
            x = nn_blocks.DarkResidual(filters=config.filters,
                                       **self._default_dict)(x)
        self._default_dict["activation"] = self._activation
        self._default_dict["name"] = None
        return x

    def _build_block(self, inputs, config, name):
        x = inputs
        i = 0
        self._default_dict["activation"] = self._get_activation(
            config.activation)
        while i < config.repetitions:
            self._default_dict["name"] = f"{name}_{i}"
            layer = self._registry(config, self._default_dict)
            x = layer(x)
            i += 1
        self._default_dict["activation"] = self._activation
        self._default_dict["name"] = None
        return x

    @property
    def input_specs(self):
        return self._input_shape

    @property
    def output_specs(self):
        return self._output_specs

    @property
    def splits(self):
        return self._splits

    @staticmethod
    def get_model_config(name, min_size, max_size):
        try:
            backbone_dict = getattr(configs, name).backbone
        except AttributeError as e:
            if e.name == configs.__package__ + '.' + name:
                raise ValueError(f"Invalid backbone '{name}'") from e
            else:
                raise
        backbone = backbone_dict["backbone"]
        splits = backbone_dict["splits"]
        return csp_build_block_specs(backbone, min_size, max_size), splits


@factory.register_backbone_builder('darknet')
def build_darknet(
    input_specs: tf.keras.layers.InputSpec,
    model_config,
    l2_regularizer: tf.keras.regularizers.Regularizer = None
) -> tf.keras.Model:

    backbone_type = model_config.backbone.type
    backbone_cfg = model_config.backbone.get()
    #norm_activation_config = model_config.norm_activation

    return Darknet(
        model_id=backbone_cfg.model_id,
        input_shape=input_specs,
        # activation=norm_activation_config.activation,
        # use_sync_bn=norm_activation_config.use_sync_bn,
        # norm_momentum=norm_activation_config.norm_momentum,
        # norm_epsilon=norm_activation_config.norm_epsilon,
        kernel_regularizer=l2_regularizer)


class temp():
    def __init__(self, backbone):
        self.backbone = backbone


# for generic usage have a depth of 8 for every layer except for the first 3 and the last.
# the last has 4 and the first 3 have 1 1 2
# tiny, 1 per depth and a final non down sampling layer with stride = 1

if __name__ == "__main__":
    from yolo.configs import backbones
    from official.core import registry

    model = backbones.Backbone(type="darknet",
                               darknet=backbones.DarkNet(model_id="darknet53"))
    cfg = temp(model)

    model = factory.build_backbone(
        tf.keras.layers.InputSpec(shape=[None, 416, 416, 3]), cfg, None)

    def print_weights(weights):
        shapes = []
        for weight in weights:
            shapes.append(weight.shape)
        return shapes

    for layer in model.layers:
        weights = layer.get_weights()
        print(f"{layer.name}: {print_weights(weights)}")

    print(model.output_specs)

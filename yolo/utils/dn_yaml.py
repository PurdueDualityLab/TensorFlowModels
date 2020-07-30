"""
Optional shortcuts for describing DarkNet models in YAML
"""

from __future__ import annotations

import copy
import os
import uuid

from yolo.modeling import building_blocks
from yolo.modeling.backbones import configs

try:
    import yaml
except ImportError:
    yaml = None

try:
    from importlib import resources as importlib_resources
except BaseException:
    # Shim for Python 3.6 and older
    import importlib_resources

import tensorflow as tf


class DarkNetYAMLLoaderBase:
    "A special YAML loader with the apply to use the repeat constructor"

    def repeat_constructor(self, tag_suffix: str, node) -> dict:
        """
        The repeat constructor allows for the repetition of a layer muliple
        times. It creates the same configuration and deep copies it so it can
        appear multiple times without being routed in multiple places.

        For more details about PyYAML multi-constructors, look at
        https://pyyaml.org/wiki/PyYAMLDocumentation#cb68
        """
        if ',' in tag_suffix:
            times_, name = tag_suffix.split(',', 1)
            times = int(times_)
        else:
            times = int(tag_suffix)
            name = "yaml_repeat_" + \
                str(tf.keras.backend.get_uid("yaml_repeat"))
        oneLayer = self.construct_mapping(node, deep=True)

        def myCopyWithName(d: dict, i: int):
            d = copy.deepcopy(d)
            d['config']['name'] = d['config']['name'] + "_" + str(i)
            return d

        allLayers = {
            'class_name': 'Sequential',
            'config': {
                'layers': (
                    [myCopyWithName(oneLayer, i) for i in range(times)]
                    if 'name' in oneLayer['config'] else
                    [copy.deepcopy(oneLayer) for i in range(times)]
                ),
                'name': name
            }
        }

        return allLayers

    def darknet(self, node):
        # https://stackoverflow.com/a/9577670
        filename = self.construct_scalar(node) + ".yml"
        with importlib_resources.open_text(configs, filename) as f:
            return yaml.load(f, self.__class__)


if yaml is None:
    DarkNetYAMLLoader = None
    DarkNetSafeYAMLLoader = None
else:
    class DarkNetUnsafeYAMLLoader(yaml.UnsafeLoader, DarkNetYAMLLoaderBase):
        pass

    class DarkNetSafeYAMLLoader(yaml.SafeLoader, DarkNetYAMLLoaderBase):
        def construct_python_tuple(self, node):
            return tuple(self.construct_sequence(node))

    # Add functionality to use python/tuple, even in safe mode.
    DarkNetSafeYAMLLoader.add_constructor(
        'tag:yaml.org,2002:python/tuple',
        DarkNetSafeYAMLLoader.construct_python_tuple)
    DarkNetSafeYAMLLoader.add_multi_constructor(
        '!repeat:', DarkNetSafeYAMLLoader.repeat_constructor)
    DarkNetSafeYAMLLoader.add_constructor(
        '!darknet', DarkNetSafeYAMLLoader.darknet)

    DarkNetUnsafeYAMLLoader.add_multi_constructor(
        '!repeat:', DarkNetUnsafeYAMLLoader.repeat_constructor)
    DarkNetUnsafeYAMLLoader.add_constructor(
        '!darknet', DarkNetUnsafeYAMLLoader.darknet)


def load_darknet_from_yaml(
        yaml_string,
        custom_objects: dict = None,
        loader: type = DarkNetSafeYAMLLoader) -> tf.keras.Model:
    """
    Parses a yaml model configuration file and returns a model instance. This
    function works the same way as model_from_yaml in the tf.keras.models
    package. The only difference is that it accepts an additional `loader`
    argument to specify the YAML loader.

    The name yaml_string is a misnomer from the Keras API. An open file object
    is also acceptable.

    Usage:

    >>> with open('yolo/modeling/backbones/configs/darknet_53.yml', mode='r') as file:
    ...     config = file.read()
    ... model = yolo.utils.dn_yaml.load_darknet_from_yaml(config)
    >>> with open('yolo/modeling/backbones/configs/darknet_53.yml', mode='r') as file:
    ...     model = yolo.utils.dn_yaml.load_darknet_from_yaml(file)

    Arguments:
        yaml_string:    A string, byte string, open binary file object, or open
                        text file that contains the YAML document.
        custom_objects: Optional dictionary mapping names (strings) to custom
                        classes or functions to be considered during deserialization.
        loader:         Class of the YAML Loader that will be used to read the
                        yaml_string

    Return:
        A deserialized Keras model specified by the yaml_string
    """
    if yaml is None:
        raise ImportError(
            'Requires yaml module installed (`pip install pyyaml`).')

    configDict = yaml.load(yaml_string, loader)
    #import pprint
    # pprint.pprint(configDict)
    return tf.keras.layers.deserialize(
        configDict, custom_objects=custom_objects)


if __name__ == '__main__':
    with open('yolo/modeling/backbones/configs/darknet_53.yml', mode='r') as file:
        config = file.read()
    loaded_model = load_darknet_from_yaml(config)
    print(loaded_model)
    with open('test.yml', mode='w') as file:
        print(loaded_model.to_yaml(), file=file)
    loaded_model(tf.keras.Input(shape=(300, 300, 3)))
    loaded_model.summary()
    #tf.keras.utils.plot_model(loaded_model, to_file='model.png', show_shapes=False, show_layer_names=True,rankdir='TB', expand_nested=False, dpi=96)

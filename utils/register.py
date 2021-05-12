import dataclasses
import functools
import inspect
import string
import warnings
from typing import ClassVar, Union

from official.core import exp_factory, registry, task_factory
from official.modeling import hyperparams
from official.vision.beta.configs import backbones, backbones_3d
from official.vision.beta.modeling import factory_3d as models_3d_factory
from official.vision.beta.modeling.backbones import \
    factory as backbones_factory


def _deduce_type(fn, config_param=1):
  if inspect.isclass(fn):
    fn = fn_or_cls.__init__
  if isinstance(config_param, int):
    sig = list(inspect.signature(fn).parameters.items())[
        config_param][1]
    return sig.annotation, sig.default
  elif isinstance(config_param, str):
    sig = inspect.signature(fn).parameters[config_param]
    return sig.annotation, sig.default
  else:
    return None, inspect.Signature.empty

def _snake_case(class_name: str):
  words = []

  for c in class_name:
    if c in string.ascii_uppercase:
      words.append('_' + c.lower())
    else:
      words.append(c)

  return ''.join(words).strip('_')

def _inject_dataclass(dcls, name, cls, value):
  del dcls.__init__, dcls.__repr__, dcls.__eq__
  dcls.__annotations__[name] = cls
  setattr(dcls, name, value)
  dataclasses.dataclass(dcls)

def _make_registry_shim(dataclass, register, config_param=1):
  def shim(name: str = None, config_class: type = None,
      default = inspect.Signature.empty):
    nonlocal dataclass, register

    def decorator(builder):
      nonlocal dataclass, register, name, config_class, default

      if config_class is None:
        config_class, deduced_default = _deduce_type(builder, config_param)
      else:
        deduced_default = inspect.Signature.empty

      if name is None:
        assert config_class is not None, 'Either the name or the class must ' \
            'be specified'
        name = _snake_case(config_class.__name__)

      if default is inspect.Signature.empty:
        if deduced_default is not inspect.Signature.empty:
          default = deduced_default
        elif config_class is not None:
          default = config_class()

      if config_class is not None:
        _inject_dataclass(dataclass, name, config_class, default)
      else:
        warnings.warn(f'Config class for {name} was not specified',
            stacklevel=2)
      return register(name)(builder)

    if callable(name):
      return decorator(name)
    return decorator

  return shim

backbone = _make_registry_shim(backbones.Backbone,
    backbones_factory.register_backbone_builder, None)
backbone_3d = _make_registry_shim(backbones_3d.Backbone3D,
    backbones_factory.register_backbone_builder, None)

task = task_factory.register_task_cls
experiment = exp_factory.register_config_factory


class Registry(dict):

  def register(self, key):
    return registry.register(self, key)

  def _lookup(self, key):
    return registry.lookup(self, key)


class RegistryOneOfConfigMetaclass(type):

  def __new__(cls, name, bases, dct):
    dct['_REGISTRY'] = Registry()
    if '__annotations__' not in dct:
      dct['__annotations__'] = {'_REGISTRY': ClassVar[dict]}
    else:
      dct['__annotations__']['_REGISTRY'] = ClassVar[dict]

    _CONFIG_PARAM = dct.get('_CONFIG_PARAM', 1)
    obj = super().__new__(cls, name, bases, dct)
    obj.register = _make_registry_shim(obj, obj._old_register,
        config_param=_CONFIG_PARAM)
    return obj


@dataclasses.dataclass
class RegistryOneOfConfig(
    hyperparams.OneOfConfig, metaclass=RegistryOneOfConfigMetaclass):
  _CONFIG_PARAM: ClassVar[Union[str, int]] = None

  @classmethod
  def _old_register(cls, key):
    return registry.register(cls._REGISTRY, key)

  @classmethod
  def _lookup(cls, key):
    return registry.lookup(cls._REGISTRY, key)

if __name__ == '__main__':
  from official.vision.beta.configs import backbones
  import tensorflow as tf

  @dataclasses.dataclass
  class Backbone(backbones.Backbone, RegistryOneOfConfig):
    # _CONFIG_PARAM = 1
    pass

  @dataclasses.dataclass
  class DarkNet(hyperparams.Config):
    """DarkNet config."""
    model_id: str = 'darknet53'

  mobilenet = Backbone({'type': 'mobilenet'})
  print(mobilenet)
  print(mobilenet.mobilenet)

  print(Backbone._REGISTRY)

  @Backbone.register('darknet')
  # @factory.register_backbone_builder("darknet")
  def build_darknet(
      input_specs: tf.keras.layers.InputSpec,
      model_config: DarkNet,
      l2_regularizer: tf.keras.regularizers.Regularizer = None
  ) -> tf.keras.Model:
    pass

  print(build_darknet)

  darknet = Backbone({'type': 'darknet'})
  print(darknet)
  print(mobilenet)
  print(Backbone._REGISTRY)
  print(Backbone._lookup('darknet'))

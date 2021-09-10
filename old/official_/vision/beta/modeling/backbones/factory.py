# Lint as: python3
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Backbone registers and factory method.

One can regitered a new backbone model by the following two steps:

1 Import the factory and register the build in the backbone file.
2 Import the backbone class and add a build in __init__.py.

```
# my_backbone.py

from modeling.backbones import factory

class MyBackbone():
  ...

@factory.register_backbone_builder('my_backbone')
def build_my_backbone():
  return MyBackbone()

# backbones/__init__.py adds import
from modeling.backbones.my_backbone import MyBackbone
```

If one wants the MyBackbone class to be used only by those binary
then don't imported the backbone module in backbones/__init__.py, but import it
in place that uses it.


"""
# Import libraries
import tensorflow as tf

from official.core import registry


_REGISTERED_BACKBONE_CLS = {}


def register_backbone_builder(key: str):
  """Decorates a builder of backbone class.

  The builder should be a Callable (a class or a function).
  This decorator supports registration of backbone builder as follows:

  ```
  class MyBackbone(tf.keras.Model):
    pass

  @register_backbone_builder('mybackbone')
  def builder(input_specs, config, l2_reg):
    return MyBackbone(...)

  # Builds a MyBackbone object.
  my_backbone = build_backbone_3d(input_specs, config, l2_reg)
  ```

  Args:
    key: the key to look up the builder.

  Returns:
    A callable for use as class decorator that registers the decorated class
    for creation from an instance of task_config_cls.
  """
  return registry.register(_REGISTERED_BACKBONE_CLS, key)


def build_backbone(input_specs: tf.keras.layers.InputSpec,
                   model_config,
                   l2_regularizer: tf.keras.regularizers.Regularizer = None):
  """Builds backbone from a config.

  Args:
    input_specs: tf.keras.layers.InputSpec.
    model_config: a OneOfConfig. Model config.
    l2_regularizer: tf.keras.regularizers.Regularizer instance. Default to None.

  Returns:
    tf.keras.Model instance of the backbone.
  """
  backbone_builder = registry.lookup(_REGISTERED_BACKBONE_CLS,
                                     model_config.backbone.type)

  return backbone_builder(input_specs, model_config, l2_regularizer)

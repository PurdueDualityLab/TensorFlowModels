from official.core import registry
import tensorflow as tf
import tensorflow.keras as ks
from typing import *

from yolo.configs import yolo

from official.vision.beta.modeling.backbones import factory
from yolo.modeling.backbones.darknet import build_darknet
from yolo.modeling.backbones.darknet import Darknet
from yolo.modeling.decoders.yolo_decoder import YoloDecoder
from yolo.modeling.heads.yolo_head import YoloHead
from yolo.modeling.layers.detection_generator import YoloLayer
from yolo.modeling.layers import nn_blocks

# static base Yolo Models that do not require configuration
# similar to a backbone model id.

# this is done greatly simplify the model config
# the structure is as follows. model version, {v3, v4, v#, ... etc}
# the model config type {regular, tiny, small, large, ... etc}
YOLO_MODELS = {
    "v4":
        dict(
            regular=dict(
                embed_spp=False,
                use_fpn=True,
                max_level_process_len=None,
                path_process_len=6),
            tiny=dict(
                embed_spp=False,
                use_fpn=False,
                max_level_process_len=2,
                path_process_len=1),
            csp=dict(
                embed_spp=False,
                use_fpn=True,
                max_level_process_len=None,
                csp_stack=5,
                fpn_depth=5,
                path_process_len=6),
            csp_large=dict(
                embed_spp=False,
                use_fpn=True,
                max_level_process_len=None,
                csp_stack=7,
                fpn_depth=7,
                path_process_len=8,
                fpn_filter_scale=2),
        ),
    "v3":
        dict(
            regular=dict(
                embed_spp=False,
                use_fpn=False,
                max_level_process_len=None,
                path_process_len=6),
            tiny=dict(
                embed_spp=False,
                use_fpn=False,
                max_level_process_len=2,
                path_process_len=1),
            spp=dict(
                embed_spp=True,
                use_fpn=False,
                max_level_process_len=2,
                path_process_len=1),
        ),
}


class Yolo(ks.Model):
  """The YOLO model class."""

  def __init__(self,
               backbone=None,
               decoder=None,
               head=None,
               filter=None,
               **kwargs):
    """Detection initialization function.
    Args:
      backbone: `tf.keras.Model` a backbone network.
      decoder: `tf.keras.Model` a decoder network.
      head: `RetinaNetHead`, the RetinaNet head.
      filter: the detection generator.
      **kwargs: keyword arguments to be passed.
    """
    super(Yolo, self).__init__(**kwargs)

    self._config_dict = {
        'backbone': backbone,
        'decoder': decoder,
        'head': head,
        'filter': filter
    }

    # model components
    self._backbone = backbone
    self._decoder = decoder
    self._head = head
    self._filter = filter
    return

  def call(self, inputs, training=False):
    maps = self._backbone(inputs)
    decoded_maps = self._decoder(maps)
    raw_predictions = self._head(decoded_maps)
    if training:
      return {"raw_output": raw_predictions}
    else:
      # Post-processing.
      predictions = self._filter(raw_predictions)
      predictions.update({"raw_output": raw_predictions})
      return predictions

  @property
  def backbone(self):
    return self._backbone

  @property
  def decoder(self):
    return self._decoder

  @property
  def head(self):
    return self._head

  @property
  def filter(self):
    return self._filter

  def get_config(self):
    return self._config_dict

  @classmethod
  def from_config(cls, config):
    return cls(**config)

  def get_weight_groups(self, train_vars):
    """Sequence of trainable variables owned by this module and its submodules.
    Note: this method uses reflection to find variables on the current instance
    and submodules. For performance reasons you may wish to cache the result
    of calling this method if you don't expect the return value to change.
    Returns:
      A sequence of variables for the current module (sorted by attribute
      name) followed by variables from all submodules recursively (breadth
      first).
    """
    bias = []
    weights = []
    other = []
    for var in train_vars:
      if "bias" in var.name:
        bias.append(var)
      elif "beta" in var.name:
        bias.append(var)
      elif "kernel" in var.name or "weight" in var.name:
        weights.append(var)
      else:
        other.append(var)
    return weights, bias, other

  def get_weight_groups_names(self):

    variables = self.trainable_variables

    name = []
    for var in variables:
      name.append(var.ref())
    return name

  def print(self):
    """Sequence of trainable variables owned by this module and its submodules.
    Note: this method uses reflection to find variables on the current instance
    and submodules. For performance reasons you may wish to cache the result
    of calling this method if you don't expect the return value to change.
    Returns:
      A sequence of variables for the current module (sorted by attribute
      name) followed by variables from all submodules recursively (breadth
      first).
    """
    modules = self.submodules

    for module in modules: 
      if isinstance(module, nn_blocks.ConvBN):
        print(module)
        print(module._kernel_regularizer)
        print(module._kernel_initializer)
        if module._use_bn:
          print(module.bn)
    return 

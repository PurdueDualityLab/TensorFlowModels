"""Backbones configurations."""
# Import libraries
import dataclasses
from typing import Optional
from official.modeling import hyperparams
from official.vision.beta.configs import backbones


@dataclasses.dataclass
class Darknet(hyperparams.Config):
  """DarkNet config."""
  model_id: str = 'cspdarknet53'
  width_scale: int = 1.0
  depth_scale: int = 1.0
  dilate: bool = False
  min_level: int = 3
  max_level: int = 5
  use_separable_conv: bool = False


@dataclasses.dataclass
class Backbone(backbones.Backbone):
  darknet: Darknet = Darknet()

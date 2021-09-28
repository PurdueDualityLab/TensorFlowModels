"""Backbones configurations."""
# Import libraries
import dataclasses
from typing import Optional, List
from official.modeling import hyperparams
from official.vision.beta.configs import backbones

@dataclasses.dataclass
class Darknet(hyperparams.Config):
  """DarkNet config."""
  model_id: str = 'cspdarknet53'
  width_scale: float = 1.0
  depth_scale: float = 1.0
  dilate: bool = False
  min_level: int = 3
  max_level: int = 5
  use_separable_conv: bool = False
  use_reorg_input: bool = False

@dataclasses.dataclass
class Swin(hyperparams.Config):
  min_level: int = 3
  max_level: Optional[int] = None
  mlp_ratio: int = 4
  patch_size: int = 4 
  embed_dims: int = 96 
  window_size: List[int] = dataclasses.field(default_factory=lambda:[7, 7, 7, 7])
  depths: List[int] = dataclasses.field(default_factory=lambda:[2, 2, 6, 2])
  num_heads: List[int] = dataclasses.field(default_factory=lambda:[3, 6, 12, 24])

@dataclasses.dataclass
class Backbone(backbones.Backbone):
  darknet: Darknet = Darknet()
  swin: Swin = Swin()

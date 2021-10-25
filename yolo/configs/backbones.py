"""Backbones configurations."""
# Import libraries
import dataclasses
from typing import Optional, List, Union
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
  min_level: Optional[int] = None
  max_level: Optional[int] = None
  embed_dims: int = 96
  depths: List[int] = dataclasses.field(default_factory=lambda:[2, 2, 6, 2])
  num_heads: List[int] = dataclasses.field(default_factory=lambda:[3, 6, 12, 24]) 
  window_size: List[int] = dataclasses.field(default_factory=lambda:[7, 7, 7, 7]) 
  patch_size: int = 4
  mlp_ratio: float = 4
  qkv_bias: bool = True
  qk_scale: bool = None
  dropout: float = 0.0
  attention_dropout: float = 0.0
  drop_path: float = 0.1
  absolute_positional_embed: bool = False
  normalize_endpoints: bool = True
  norm_layer: str = 'layer_norm'

@dataclasses.dataclass
class Backbone(backbones.Backbone):
  darknet: Darknet = Darknet()
  swin: Swin = Swin()

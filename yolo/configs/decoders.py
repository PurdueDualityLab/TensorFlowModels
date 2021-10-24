"""Backbones configurations."""
# Import libraries
import dataclasses
from typing import Optional, List
from official.modeling import hyperparams
from official.vision.beta.configs import decoders

@dataclasses.dataclass
class YoloDecoder(hyperparams.Config):
  """Builds Yolo decoder.

  If the name is specified, or version is specified we ignore input parameters
  and use version and name defaults.
  """
  version: Optional[str] = None
  type: Optional[str] = None
  use_fpn: Optional[bool] = None
  use_spatial_attention: bool = False
  use_separable_conv: bool = False
  csp_stack: Optional[bool] = None
  fpn_depth: Optional[int] = None
  max_fpn_depth: Optional[int] = None
  max_csp_stack: Optional[int] = None
  fpn_filter_scale: Optional[int] = None
  path_process_len: Optional[int] = None
  max_level_process_len: Optional[int] = None
  embed_spp: Optional[bool] = None
  activation: Optional[str] = 'same'

@dataclasses.dataclass
class TBiFPN(hyperparams.Config):
  """Builds Yolo decoder.

  If the name is specified, or version is specified we ignore input parameters
  and use version and name defaults.
  """
  fpn_only: bool = False
  repititions: int = 1
  include_detokenization: bool = True 
  use_separable_conv: bool = True
  window_size: int = 8
  token_size: int  = 32
  mlp_ratio: int  = 2
  kernel_size: int = 1
  activation: Optional[str] = 'mish'
  use_patch_expansion: bool = False
  shift: bool = False
  expansion_kernel_size: bool = 1

@dataclasses.dataclass
class Decoder(decoders.Decoder):
  type: Optional[str] = 'yolo_decoder'
  yolo_decoder: YoloDecoder = YoloDecoder()
  tbifpn_decoder: TBiFPN = TBiFPN()

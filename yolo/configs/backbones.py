"""Backbones configurations."""
from typing import Optional

# Import libraries
import dataclasses

from official.modeling import hyperparams


@dataclasses.dataclass
class ResNet(hyperparams.Config):
  """ResNet config."""
  model_id: int = 50


@dataclasses.dataclass
class EfficientNet(hyperparams.Config):
  """EfficientNet config."""
  model_id: str = 'b0'
  stochastic_depth_drop_rate: float = 0.0
  se_ratio: float = 0.0


@dataclasses.dataclass
class MobileNet(hyperparams.Config):
  """Mobilenet config."""
  model_id: str = 'MobileNetV2'
  filter_size_scale: float = 1.0
  stochastic_depth_drop_rate: float = 0.0


@dataclasses.dataclass
class SpineNet(hyperparams.Config):
  """SpineNet config."""
  model_id: str = '49'
  stochastic_depth_drop_rate: float = 0.0

@dataclasses.dataclass
class RevNet(hyperparams.Config):
  """RevNet config."""
  # Specifies the depth of RevNet.
  model_id: int = 56

@dataclasses.dataclass
class DarkNet53(hyperparams.Config):
  """RevNet config."""
  # Specifies the depth of RevNet.
  model_id: str = "regular"

@dataclasses.dataclass
class CSPDarkNet53(hyperparams.Config):
  """RevNet config."""
  # Specifies the depth of RevNet.
  model_id: str = "regular"

@dataclasses.dataclass
class Backbone(hyperparams.OneOfConfig):
  """Configuration for backbones.
  Attributes:
    type: 'str', type of backbone be used, one the of fields below.
    resnet: resnet backbone config.
    revnet: revnet backbone config.
    efficientnet: efficientnet backbone config.
    spinenet: spinenet backbone config.
    mobilenet: mobilenet backbone config.
  """
  type: Optional[str] = None
  resnet: ResNet = ResNet()
  revnet: RevNet = RevNet()
  efficientnet: EfficientNet = EfficientNet()
  spinenet: SpineNet = SpineNet()
  mobilenet: MobileNet = MobileNet()
  darknet53: DarkNet53 = DarkNet53()
  cspdarknet53: CSPDarkNet53 = CSPDarkNet53()


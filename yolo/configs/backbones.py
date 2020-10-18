"""Backbones configurations."""
from typing import Optional, Dict
# Import libraries
import dataclasses
from official.modeling import hyperparams

@dataclasses.dataclass
class DarkNet(hyperparams.Config):
    """RevNet config."""
    # Specifies the depth of RevNet.
    model_id: str = "darknet53"

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
    darknet: DarkNet = DarkNet()

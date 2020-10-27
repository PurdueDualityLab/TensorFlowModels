"""Backbones configurations."""
# Import libraries
import dataclasses
from typing import Optional
from official.modeling import hyperparams


@dataclasses.dataclass
class ResNet(hyperparams.Config):
    """ResNet config."""
    model_id: int = 50


@dataclasses.dataclass
class DilatedResNet(hyperparams.Config):
    """DilatedResNet config."""
    model_id: int = 50
    output_stride: int = 16


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
        dilated_resnet: dilated resnet backbone for semantic segmentation config.
        revnet: revnet backbone config.
        efficientnet: efficientnet backbone config.
        spinenet: spinenet backbone config.
        mobilenet: mobilenet backbone config.
    """
    type: Optional[str] = None
    resnet: ResNet = ResNet()
    dilated_resnet: DilatedResNet = DilatedResNet()
    revnet: RevNet = RevNet()
    efficientnet: EfficientNet = EfficientNet()
    spinenet: SpineNet = SpineNet()
    mobilenet: MobileNet = MobileNet()
    darknet: DarkNet = DarkNet()

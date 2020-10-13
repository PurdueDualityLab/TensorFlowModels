"""Backbones configurations."""
from typing import Optional, Dict
# Import libraries
import dataclasses
from official.modeling import hyperparams

def get_darknet_splits(version, name):
    try: 
        values = {"Darknet53": {"regular":{"backbone_head": 76}, 
                                "spp":{"backbone_head": 76}, 
                                "tiny":{"backbone_head": 14}}, 
                "CSPDarknet53": {"regular":{"backbone_neck": 106, "neck_head": 138}}}
        return values[version][name]
    except: 
        return None

@dataclasses.dataclass
class DarkNet53(hyperparams.Config):
    """RevNet config."""
    # Specifies the depth of RevNet.
    model_id: str = "regular"
    splits: Dict = get_darknet_splits("Darknet53", model_id)

@dataclasses.dataclass
class CSPDarkNet53(hyperparams.Config):
    """RevNet config."""
    # Specifies the depth of RevNet.
    model_id: str = "regular"
    splits: Dict = get_darknet_splits("CSPDarknet53", model_id)

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
    darknet53: DarkNet53 = DarkNet53()
    cspdarknet53: CSPDarkNet53 = CSPDarkNet53()


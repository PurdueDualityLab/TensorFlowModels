"""Backbones configurations."""
from typing import Optional

# Import libraries
import dataclasses
from official.modeling import hyperparams


@dataclasses.dataclass
class Yolov3head(hyperparams.Config):
    model_id: str = "regular"


@dataclasses.dataclass
class Yolov4head(hyperparams.Config):
    model_id: str = "regular"


@dataclasses.dataclass
class head(hyperparams.OneOfConfig):
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
    v3: Yolov3head = Yolov3head()
    v4: Yolov4head = Yolov4head()

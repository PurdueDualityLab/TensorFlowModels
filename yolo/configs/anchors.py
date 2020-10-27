"""Backbones configurations."""
from typing import Optional

# Import libraries
import dataclasses
from official.modeling import hyperparams
from yolo.configs import cfg_defs as cfg


@dataclasses.dataclass
class Yolov3Anchors(cfg.AnchorCFG):
    """RevNet config."""
    # Specifies the depth of RevNet.
    _boxes: List[str] = dataclasses.field(default_factory=lambda: [
        "10, 13", "16, 30", "33, 23", "30, 61", "62, 45", "59, 119", "116, 90",
        "156, 198", "373, 326"
    ])
    masks: Dict = dataclasses.field(default_factory=lambda: {
        5: [6, 7, 8],
        4: [3, 4, 5],
        3: [0, 1, 2]
    })
    path_scales: Dict = dataclasses.field(default_factory=lambda: {
        5: 32,
        4: 16,
        3: 8
    })
    x_y_scales: Dict = dataclasses.field(default_factory=lambda: {
        5: 1.0,
        4: 1.0,
        3: 1.0
    })


@dataclasses.dataclass
class Yolov4Anchors(cfg.AnchorCFG):
    """RevNet config."""
    # Specifies the depth of RevNet.
    _boxes: List[str] = dataclasses.field(default_factory=lambda: [
        "12, 16", "19, 36", "40, 28", "36, 75", "76, 55", "72, 146",
        "142, 110", "192, 243", "459, 401"
    ])
    masks: Dict = dataclasses.field(default_factory=lambda: {
        5: [6, 7, 8],
        4: [3, 4, 5],
        3: [0, 1, 2]
    })
    path_scales: Dict = dataclasses.field(default_factory=lambda: {
        5: 32,
        4: 16,
        3: 8
    })
    x_y_scales: Dict = dataclasses.field(default_factory=lambda: {
        5: 1.05,
        4: 1.1,
        3: 1.2
    })


@dataclasses.dataclass
class Yolov3TinyAnchors(cfg.AnchorCFG):
    """RevNet config."""
    # Specifies the depth of RevNet.
    _boxes: List[str] = dataclasses.field(
        default_factory=lambda:
        ["10, 14", "23, 27", "37, 58", "81, 82", "135, 169", "344, 319"])
    masks: Dict = dataclasses.field(default_factory=lambda: {
        5: [3, 4, 5],
        3: [0, 1, 2]
    })
    path_scales: Dict = dataclasses.field(default_factory=lambda: {
        5: 32,
        3: 8
    })
    x_y_scales: Dict = dataclasses.field(default_factory=lambda: {
        5: 1.0,
        3: 1.0
    })


@dataclasses.dataclass
class Anchors(hyperparams.OneOfConfig):
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
    v3: Yolov3Anchors = Yolov3Anchors()
    v3_spp: Yolov3Anchors = Yolov3Anchors()
    v3_tiny: Yolov3TinyAnchors = Yolov3TinyAnchors()
    v4: Yolov4Anchors = Yolov4Anchors()

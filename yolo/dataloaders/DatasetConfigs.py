from dataclasses import dataclass, field
from yolo.dataloaders.ops.custom_dataset_ops import _yolo_coco_id_parser, _yolo_coco_label_parser


@dataclass
class Dataset():
    dataset_type: str = field(init=True, repr=True, default="tfds")
    dataset_name: str = field(init=True, repr=True, default="coco")
    train_set_path: str = field(
        init=True, repr=True,
        default="train")  # if custim it is a list of file paths
    val_set_path: str = field(init=True, repr=True, default="validation")

    # custom datasets  path/to/labels/<image_id>.*
    label_to_image: bool = field(init=True, repr=True, default=False)
    train_labels_paths: List = field(init=True, repr=True,
                                     default=None)  # add a key for each one
    val_labels_paths: List = field(init=True, repr=True,
                                   default=None)  # add a key for each one
    id_parser: Callable = field(
        init=True, repr=True,
        default=_yolo_coco_id_parser)  # function to get the id of the labels
    label_parser: Union[Callable,
                        Dict] = field(init=True,
                                      repr=True,
                                      default=_yolo_coco_label_parser
                                      )  # function to get the id of the labels

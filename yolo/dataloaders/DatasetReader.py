import tensorflow as tf 

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import * #Tuple, Sequence, List, Callable, Dict, Union

import re


def _yolo_coco_id_parser(value):
    #im = value.split('/')[-1].split(".")[0].split("_")[-1]
    value = tf.strings.split(input = tf.strings.split(input = tf.strings.split(input = value, sep="/")[-1], sep=".")[0], sep = "_")[-1]
    return tf.strings.join([value, ".*"])

def _yolo_coco_label_parser(id, paths):
    #im = value.split('/')[-1].split(".")[0].split("_")[-1]
    ret_dict = dict.fromkeys(["bbox", "labels"])
    for path in paths:
        path = tf.strings.join([path, "/" ,id])
        ret_dict["bbox"] = path
        ret_dict["labels"] = path        
    return ret_dict

@dataclass
class Dataset():
    dataset_type: str = field(init=True, repr=True, default="tfds")
    dataset_name: str = field(init=True, repr=True, default="coco")
    train_set_path: str = field(init=True, repr=True, default="train") # if custim it is a list of file paths 
    val_set_path: str = field(init=True, repr=True, default="validation")

    # custom datasets  path/to/labels/<image_id>.*
    label_to_image:bool = field(init=True, repr=True, default=False)
    train_labels_paths: List = field(init=True, repr=True, default=None) # add a key for each one
    val_labels_paths: List = field(init=True, repr=True, default=None) # add a key for each one
    id_parser: Callable = field(init=True, repr=True, default=_yolo_coco_id_parser) # function to get the id of the labels
    label_parser: Union[Callable, Dict] = field(init=True, repr=True, default=_yolo_coco_label_parser) # function to get the id of the labels

def _get_images(file_paths):
    paths = []
    for path_instance in file_paths:
        path_instance = path_instance.strip().split(",")
        for path in path_instance:
            if "*" in path or "?" in path:
                temp_paths = tf.io.gfile.glob(path)
                if temp_paths == None:
                    raise IOError("no matches found to dataset path")
                paths.extend(temp_paths)
            else:
                paths.append(path)
    return paths

def _str_to_list(path):
    if isinstance(path, Dict):
        return paths
    if isinstance(path, str):
        paths = [path]
    else:
        paths = path
    return paths

class DatasetReader():
    def __init__(self, 
                 config: Dataset):
        self._tfds_builder = None
        self._matched_files = None
        self._is_sharder = True if config.dataset_type == "custom" else False

        self._train_set = []
        self._val_set = []
        self._train_labels = _str_to_list(config.train_labels_paths)
        self._val_labels = _str_to_list(config.val_labels_paths) 
        if config.dataset_type == "custom":
            if not config.train_set_path == None:
                train_paths = _str_to_list(config.train_set_path)
                self._train_set = _get_images(train_paths)

            if not config.val_set_path == None:
                val_paths = _str_to_list(config.val_set_path)
                self._val_set = _get_images(val_paths)
            print(self._train_set[0:1])
            print(self._val_set[0:1])
        
        self._get_id_fn = config.id_parser
        self._parse_label_fn = config.label_parser
        

    def _parse_custom(self, im_path, is_training = False):
        image = tf.io.decode_jpeg(tf.io.read_file(im_path))
        im_id = self._get_id_fn(im_path)
        if is_training:
            labels = self._train_labels
        else:
            labels = self._val_labels
        
        labels = self._parse_label_fn(im_id, self._val_labels)

        return {"image": image, "label": labels}
    
    def build_dataset(self):
        if self._is_sharder:
            train = tf.data.Dataset.from_tensor_slices(self._train_set)
            val = tf.data.Dataset.from_tensor_slices(self._val_set)
            train = train.shuffle(1024)
            val = val.shuffle(1024)

            train = train.map(lambda x: self._parse_custom(x, is_training = True))
            val = val.map(lambda x: self._parse_custom(x))
        
        dataset = train.concatenate(val)
        return dataset

if __name__ == "__main__":
    config = Dataset(dataset_type= "custom", 
                     dataset_name= "coco", 
                     train_set_path= "/home/vishnu/datasets/coco/images/train2014/*", 
                     val_set_path= "/home/vishnu/datasets/coco/images/val2014/*",
                     train_labels_paths= "/home/vishnu/datasets/coco/labels/train2014/*", 
                     val_labels_paths= "/home/vishnu/datasets/coco/labels/val2014/*", 
                     id_parser=_yolo_coco_id_parser)
    dsset = DatasetReader(config)

    s = dsset.build_dataset()

    for value in s.take(12):
        print(value)
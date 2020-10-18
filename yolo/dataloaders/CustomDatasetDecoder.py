import tensorflow as tf
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import *

from .DatasetConfigs import Dataset
from .op.custom_dataset_ops import _get_images, _str_to_list

class DatasetReader():
    def __init__(self, config: Dataset):
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

    def _parse_custom(self, im_path, is_training=False):
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

            train = train.map(
                lambda x: self._parse_custom(x, is_training=True))
            val = val.map(lambda x: self._parse_custom(x))

        dataset = train.concatenate(val)
        return dataset
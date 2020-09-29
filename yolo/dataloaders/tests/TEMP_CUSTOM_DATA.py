# WILL BE MERGING SOON TO DATASETREADER.PY, THE REGULAR JSON PARSER IS IN THIS FILE

import tensorflow as tf
import tensorflow_datasets as tfds
import glob
import matplotlib.pyplot as plt
import xmltodict
import json
import os
import sys
import shutil

labels = {1: 0, 2: 0, 3: 0}
print(str(labels))

IMAGE_PATH = ""
LABEL_PATH = ""


def xml_json_converter(label_folder_path):
    json_dir = os.path.dirname(label_folder_path) + "_json/"
    if os.path.isdir(json_dir[:-1]):
        shutil.rmtree(json_dir[:-1])
    os.mkdir(json_dir)
    for x in glob.glob(label_folder_path):
        with open(x) as xml_file:
            data_dict = xmltodict.parse(xml_file.read())
            xml_file.close()
        with open(
                json_dir + os.path.basename(os.path.normpath(x))[:-4] +
                ".json", "w") as json_file:
            json_file.write(json.dumps(data_dict))
            json_file.close()
    return json_dir + "*.json"


def json_sereloizer(json_string):
    dictionary = json.loads(
        json.loads(json.dumps(json_string.numpy().decode('utf-8'))))
    obj = dictionary["annotation"]["object"]
    area = int(dictionary["annotation"]["size"]["width"]) * int(
        dictionary["annotation"]["size"]["height"])
    return obj["name"], 1, int(obj["bndbox"]["xmin"]), int(
        obj["bndbox"]["ymin"]), int(obj["bndbox"]["xmax"]), int(
            obj["bndbox"]["ymax"]), area


def parser(fp):
    label = tf.strings.split(fp, '/')[-1]
    f = tf.io.read_file(fp)
    name, identifier, xmin, ymin, xmax, ymax, area = tf.py_function(
        json_sereloizer, [f], [
            tf.string, tf.int64, tf.float32, tf.float32, tf.float32,
            tf.float32, tf.int64
        ])
    return {
        'id': identifier,
        'name': name,
        'bbox': tf.concat([xmin, ymin, xmax, ymax], 0),
        'area': area
    }


class customdataloader:
    def __init__(self, image_folder_path, label_folder_path, label_parser):
        self.image_folder_path = image_folder_path
        self.label_folder_path = label_folder_path
        self.label_parser = label_parser
        self.map = None

    def construct_map(self):
        def map(file_paths):
            image_path = file_paths[0]
            label_path = file_paths[1]
            image = tf.image.decode_image(tf.io.read_file(image_path),
                                          channels=3)
            label = self.label_parser(label_path)
            return image, label

        return map

    def construct_dataset(self):
        if not self.map:
            self.map = self.construct_map()
        fp = list(
            zip(glob.glob(self.image_folder_path),
                glob.glob(self.label_folder_path)))
        ds = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(fp))
        ds = ds.map(self.map)
        return ds


LABEL_PATH = xml_json_converter(LABEL_PATH)
ldr = customdataloader(IMAGE_PATH, LABEL_PATH, parser)
ds = ldr.construct_dataset()

for image, label in ds.take(5):
    print(image.shape)
    print(label)

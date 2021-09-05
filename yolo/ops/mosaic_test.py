# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for mosaic.py."""

import tensorflow_datasets as tfds
import tensorflow as tf
import matplotlib.pyplot as plt

from yolo.dataloaders.decoders import tfds_coco_decoder
from yolo.ops import mosaic
from yolo.utils.demos import utils, coco

def test_mosaic():
    mos = mosaic.Mosaic(output_size=(512,512),
                        mosaic_crop_mode='crop')
    mos_fn = mos.mosaic_fn()
    ds = tfds.load('coco', split = 'train')
    decoder = tfds_coco_decoder.MSCOCODecoder()
    ds = ds.map(decoder.decode)
    ds = mos_fn(ds)
    for i in ds.take(10):
        drawer = utils.DrawBoxes(
        labels=coco.get_coco_names(
            path="yolo/dataloaders/dataset_specs/coco.names"
        ),
        thickness=2,
        classes=91)
        image = i['image'] / 255.
        draw_dict = {
            'bbox': i['groundtruth_boxes'],
            'classes': i['groundtruth_classes'],
            'confidence': tf.ones(tf.shape(i['groundtruth_classes']),
                dtype=tf.int64),
        }
        image = drawer(image, draw_dict)
        plt.imshow(image)
        plt.show()

if __name__ == "__main__":
    test_mosaic()    
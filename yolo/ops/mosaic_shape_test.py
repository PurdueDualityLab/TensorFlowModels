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

"""Shape test for mosaic.py."""

import tensorflow_datasets as tfds
import tensorflow as tf
from absl.testing import parameterized

from yolo.dataloaders.decoders import tfds_coco_decoder
from yolo.ops import mosaic

class InputUtilsTest(parameterized.TestCase, tf.test.TestCase):
    @parameterized.parameters(
        ((512, 512), 'crop'),
        ((416, 416), 'scale'),
        ((608, 608), 'crop'),
    )
    def testMosaic(self, 
                   output_size,
                   mosaic_crop_mode):
        mos = mosaic.Mosaic(output_size = output_size,
                            mosaic_crop_mode = mosaic_crop_mode)
        mos_fn = mos.mosaic_fn()
        ds = tfds.load('coco', split = 'train')
        decoder = tfds_coco_decoder.MSCOCODecoder()
        ds = ds.map(decoder.decode)
        ds = mos_fn(ds)
        for i in ds.take(10):
            image_shape = i['image'].shape
            box_shape = i['groundtruth_boxes'].shape
            class_shape = i['groundtruth_classes'].shape
            self.assertAllEqual(image_shape[:-1], output_size)
            self.assertAllEqual(box_shape[-1], 4)
            self.assertAllEqual(box_shape[:-1], class_shape)

if __name__ == '__main__':
  tf.test.main()
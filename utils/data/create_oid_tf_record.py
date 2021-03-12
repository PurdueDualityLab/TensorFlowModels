# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
r"""Creates TFRecords of Open Images dataset for object detection.

Example usage:
  python object_detection/dataset_tools/create_oid_tf_record.py \
    --input_box_annotations_csv=/path/to/input/annotations-human-bbox.csv \
    --input_image_label_annotations_csv=/path/to/input/annotations-label.csv \
    --input_images_directory=/path/to/input/image_pixels_directory \
    --input_label_map=/path/to/input/labels_bbox_545.labelmap \
    --output_tf_record_path_prefix=/path/to/output/prefix.tfrecord

CSVs with bounding box annotations and image metadata (including the image URLs)
can be downloaded from the Open Images GitHub repository:
https://github.com/openimages/dataset

This script will include every image found in the input_images_directory in the
output TFRecord, even if the image has no corresponding bounding box annotations
in the input_annotations_csv. If input_image_label_annotations_csv is specified,
it will add image-level labels as well. Note that the information of whether a
label is positivelly or negativelly verified is NOT added to tfrecord.
"""

import collections
import logging
import os

from absl import app  # pylint:disable=unused-import
from absl import flags
import pandas as pd

import tensorflow as tf

import multiprocessing as mp
from official.vision.beta.data import tfrecord_lib


flags.DEFINE_string(
    'input_box_annotations_csv', None,
    'Path to CSV containing image bounding box annotations')
flags.DEFINE_string(
    'input_images_directory', None,
    'Directory containing the image pixels '
    'downloaded from the OpenImages GitHub repository.')
flags.DEFINE_string(
    'input_image_label_annotations_csv', None,
    'Path to CSV containing image-level labels annotations')
flags.DEFINE_string('input_label_map', None, 'Path to the label map proto')
flags.DEFINE_string(
    'output_tf_record_path_prefix', None,
    'Path to the output TFRecord. The shard index and the number of shards '
    'will be appended for each output shard.')
flags.DEFINE_integer('num_shards', 100, 'Number of TFRecord shards')

FLAGS = flags.FLAGS

logger = tf.get_logger()
logger.setLevel(logging.INFO)


def bbox_annotations_to_feature_dict(
    bbox_annotations, image_height, image_width, id_to_name_map, include_masks):
  """Convert COCO annotations to an encoded feature dict."""

  data, num_skipped = oid_annotations_to_lists(
      bbox_annotations, id_to_name_map, image_height, image_width,
      include_masks)
  feature_dict = {
      'image/object/bbox/xmin':
          tfrecord_lib.convert_to_feature(data['xmin']),
      'image/object/bbox/xmax':
          tfrecord_lib.convert_to_feature(data['xmax']),
      'image/object/bbox/ymin':
          tfrecord_lib.convert_to_feature(data['ymin']),
      'image/object/bbox/ymax':
          tfrecord_lib.convert_to_feature(data['ymax']),
      'image/object/class/text':
          tfrecord_lib.convert_to_feature(data['category_names']),
      'image/object/class/label':
          tfrecord_lib.convert_to_feature(data['category_id']),
      'image/object/is_crowd':
          tfrecord_lib.convert_to_feature(data['is_crowd']),
      'image/object/area':
          tfrecord_lib.convert_to_feature(data['area']),
  }
  if include_masks:
    feature_dict['image/object/mask'] = (
        tfrecord_lib.convert_to_feature(data['encoded_mask_png']))

  return feature_dict, num_skipped


def encode_caption_annotations(caption_annotations):
  captions = []
  for caption_annotation in caption_annotations:
    captions.append(caption_annotation['caption'].encode('utf8'))

  return captions


def create_tf_example(image,
                      image_dir,
                      bbox_annotations=None,
                      id_to_name_map=None,
                      caption_annotations=None,
                      include_masks=False):
  """Converts image and annotations to a tf.Example proto.

  Args:
    image: dict with keys: [u'license', u'file_name', u'coco_url', u'height',
      u'width', u'date_captured', u'flickr_url', u'id']
    image_dir: directory containing the image files.
    bbox_annotations:
      list of dicts with keys: [u'segmentation', u'area', u'iscrowd',
        u'image_id', u'bbox', u'category_id', u'id'] Notice that bounding box
        coordinates in the official COCO dataset are given as [x, y, width,
        height] tuples using absolute coordinates where x, y represent the
        top-left (0-indexed) corner.  This function converts to the format
        expected by the Tensorflow Object Detection API (which is which is
        [ymin, xmin, ymax, xmax] with coordinates normalized relative to image
        size).
    id_to_name_map: a dict mapping category IDs to string names.
    caption_annotations:
      list of dict with keys: [u'id', u'image_id', u'str'].
    include_masks: Whether to include instance segmentations masks
      (PNG encoded) in the result. default: False.

  Returns:
    example: The converted tf.Example
    num_annotations_skipped: Number of (invalid) annotations that were ignored.

  Raises:
    ValueError: if the image pointed to by data['filename'] is not a valid JPEG
  """
  image_height = image['height']
  image_width = image['width']
  filename = image['file_name']
  image_id = image['id']

  full_path = os.path.join(image_dir, filename)
  with tf.io.gfile.GFile(full_path, 'rb') as fid:
    encoded_jpg = fid.read()

  feature_dict = tfrecord_lib.image_info_to_feature_dict(
      image_height, image_width, filename, image_id, encoded_jpg, 'jpg')

  num_annotations_skipped = 0
  if bbox_annotations:
    box_feature_dict, num_skipped = bbox_annotations_to_feature_dict(
        bbox_annotations, image_height, image_width, id_to_name_map,
        include_masks)
    num_annotations_skipped += num_skipped
    feature_dict.update(box_feature_dict)

  if caption_annotations:
    encoded_captions = encode_caption_annotations(caption_annotations)
    feature_dict.update(
        {'image/caption': tfrecord_lib.convert_to_feature(encoded_captions)})

  example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
  return example, num_annotations_skipped


def _create_tf_record_from_oid_annotations(images_info_file,
                                            image_dir,
                                            output_path,
                                            num_shards,
                                            object_annotations_file=None,
                                            caption_annotations_file=None,
                                            include_masks=False):
    # TODO: edit documentation
    """Loads COCO annotation json files and converts to tf.Record format.

    Args:
        images_info_file: JSON file containing image info. The number of tf.Examples
        in the output tf Record files is exactly equal to the number of image info
        entries in this file. This can be any of train/val/test annotation json
        files Eg. 'image_info_test-dev2017.json',
        'instance_annotations_train2017.json',
        'caption_annotations_train2017.json', etc.
        image_dir: Directory containing the image files.
        output_path: Path to output tf.Record file.
        num_shards: Number of output files to create.
        object_annotations_file: JSON file containing bounding box annotations.
        caption_annotations_file: JSON file containing caption annotations.
        include_masks: Whether to include instance segmentations masks
        (PNG encoded) in the result. default: False.
    """

  logging.info('writing to output path: %s', output_path)


  logging.info('Finished writing, skipped %d annotations.', num_skipped)


def main(_):
  assert FLAGS.image_dir, '`image_dir` missing.'
  assert (FLAGS.image_info_file or FLAGS.object_annotations_file or
          FLAGS.caption_annotations_file), ('All annotation files are '
                                            'missing.')
  if FLAGS.image_info_file:
    images_info_file = FLAGS.image_info_file
  elif FLAGS.object_annotations_file:
    images_info_file = FLAGS.object_annotations_file
  else:
    images_info_file = FLAGS.caption_annotations_file

  directory = os.path.dirname(FLAGS.output_file_prefix)
  if not tf.io.gfile.isdir(directory):
    tf.io.gfile.makedirs(directory)

  _create_tf_record_from_oid_annotations(images_info_file, FLAGS.image_dir,
                                          FLAGS.output_file_prefix,
                                          FLAGS.num_shards,
                                          FLAGS.object_annotations_file,
                                          FLAGS.caption_annotations_file,
                                          FLAGS.include_masks)


if __name__ == '__main__':
  app.run(main)

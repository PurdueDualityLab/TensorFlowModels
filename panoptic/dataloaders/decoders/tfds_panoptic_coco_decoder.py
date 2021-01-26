"""Tensorflow Example proto decoder for object detection.
A decoder to decode string tensors containing serialized tensorflow.Example
protos for object detection.
"""
import csv
# Import libraries
import tensorflow as tf

from official.vision.beta.dataloaders import decoder

def _generate_source_id(image_bytes):
  return tf.strings.as_string(
      tf.strings.to_hash_bucket_fast(image_bytes, 2**63 - 1))


class MSCOCODecoder(decoder.Decoder):
  """Tensorflow Example proto decoder."""

  def __init__(self, include_mask=False, regenerate_source_id=False):
    self._include_mask = include_mask
    self._regenerate_source_id = regenerate_source_id
    self._object_key = "objects"

    if include_mask:
      self._object_key = "panoptic_objects"
      print("masks are only included when using coco/2017_panoptic")

  def _decode_image(self, parsed_tensors):
    """Decodes the image and set its static shape."""
    return parsed_tensors['image']

  def _decode_boxes(self, parsed_tensors):
    """Concat box coordinates in the format of [ymin, xmin, ymax, xmax]."""
    return parsed_tensors[self._object_key]['bbox']

  def _decode_classes(self, parsed_tensors):
    return parsed_tensors[self._object_key]['label']

  def _decode_areas(self, parsed_tensors):
    ymin = parsed_tensors[self._object_key]['bbox'][..., 0]
    xmin = parsed_tensors[self._object_key]['bbox'][..., 1]
    ymax = parsed_tensors[self._object_key]['bbox'][..., 2]
    xmax = parsed_tensors[self._object_key]['bbox'][..., 3]
    shape = tf.cast(tf.shape(parsed_tensors['image']), tf.float32)
    width = shape[0]
    height = shape[1]
    return (ymax - ymin) * (xmax - xmin) * width * height

  def _get_instance_id(self, image_mask):
    image_mask = tf.cast(image_mask, tf.int64)
    id = image_mask[..., 0] + image_mask[..., 1] * 256 + image_mask[..., 2] * 256 ** 2
    return id

  def _decode_masks(self, parsed_tensors):
    """Decode a set of PNG masks to the tf.float32 tensors."""
    id_mask = self._get_instance_id(parsed_tensors["panoptic_image"])
    ids = parsed_tensors["panoptic_objects"]['id']  
    classes = parsed_tensors["panoptic_objects"]['label']  

    ids = tf.expand_dims(ids, axis = 0)
    ids = tf.expand_dims(ids, axis = 0)
    ids = tf.repeat(ids, tf.shape(id_mask)[0], axis = 0)
    ids = tf.repeat(ids, tf.shape(id_mask)[1], axis = 1)

    classes = tf.expand_dims(classes, axis = 0)
    classes = tf.expand_dims(classes, axis = 0)
    classes = tf.repeat(classes, tf.shape(id_mask)[0], axis = 0)
    classes = tf.repeat(classes, tf.shape(id_mask)[1], axis = 1)

    bool_mask = tf.cast(tf.math.equal(tf.expand_dims(id_mask, axis = -1), ids), tf.int64)
    class_mask = tf.reduce_sum(bool_mask * classes, axis = -1)
    return class_mask, id_mask

  def decode(self, serialized_example):
    """Decode the serialized example.
    Args:
      serialized_example: a single serialized tf.Example string.
    Returns:
      decoded_tensors: a dictionary of tensors with the following fields:
        - source_id: a string scalar tensor.
        - image: a uint8 tensor of shape [None, None, 3].
        - height: an integer scalar tensor.
        - width: an integer scalar tensor.
        - groundtruth_classes: a int64 tensor of shape [None].
        - groundtruth_is_crowd: a bool tensor of shape [None].
        - groundtruth_area: a float32 tensor of shape [None].
        - groundtruth_boxes: a float32 tensor of shape [None, 4].
        - groundtruth_instance_masks: a float32 tensor of shape
            [None, None, None].
        - groundtruth_instance_masks_png: a string tensor of shape [None].
    """
    parsed_tensors = serialized_example
    image = self._decode_image(parsed_tensors)

    if 'image/id' in parsed_tensors.keys():
      source_id = parsed_tensors['image/id']
    else:
      ime = tf.io.encode_jpeg(image, quality=100)
      source_id = _generate_source_id(ime)

    boxes = self._decode_boxes(parsed_tensors)
    classes = self._decode_classes(parsed_tensors)
    areas = self._decode_areas(parsed_tensors)

    if 'is_crowd' in parsed_tensors[self._object_key].keys():
      is_crowd = tf.cond(
          tf.greater(tf.shape(classes)[0], 0),
          lambda: tf.cast(parsed_tensors[self._object_key]['is_crowd'], dtype=tf.bool),
          lambda: tf.zeros_like(classes, dtype=tf.bool))
    else:
      is_crowd = tf.zeros_like(classes, dtype=tf.bool)

    decoded_tensors = {
        'source_id': source_id,
        'image': image,
        'width': tf.shape(parsed_tensors['image'])[0],
        'height': tf.shape(parsed_tensors['image'])[1],
        'groundtruth_classes': classes,
        'groundtruth_is_crowd': is_crowd,
        'groundtruth_area': areas,
        'groundtruth_boxes': boxes,
    }

    if self._include_mask:
      semantic_mask, pantopic_id = self._decode_masks(parsed_tensors)

      decoded_tensors.update({
          'groundtruth_semantic_mask': semantic_mask,
          'groundtruth_instance_id': pantopic_id,
          'groundtruth_pantopic_mask_png': parsed_tensors["panoptic_image/filename"],
          'groundtruth_pantopic_mask': parsed_tensors["panoptic_image"],
      })
    # tf.print(decoded_tensors)
    return decoded_tensors

""" bounding box utils file """

# import libraries
import tensorflow as tf
from typing import Tuple, Union


def yxyx_to_xcycwh(box: tf.Tensor):
  """Converts boxes from ymin, xmin, ymax, xmax to x_center, y_center, width, height.
    Args:
      box: a `Tensor` whose last dimension is 4 representing the coordinates of boxes
        in ymin, xmin, ymax, xmax.
    Returns:
      box: a `Tensor` whose shape is the same as `box` in new format.
    """
  with tf.name_scope('yxyx_to_xcycwh'):
    ymin, xmin, ymax, xmax = tf.split(box, 4, axis=-1)
    x_center = (xmax + xmin) / 2
    y_center = (ymax + ymin) / 2
    width = xmax - xmin
    height = ymax - ymin
    box = tf.concat([x_center, y_center, width, height], axis=-1)
  return box


def xcycwh_to_yxyx(box: tf.Tensor, split_min_max: bool = False):
  """Converts boxes from x_center, y_center, width, height to ymin, xmin, ymax, xmax.
    Args:
      box: a `Tensor` whose last dimension is 4 representing the coordinates of boxes in
        x_center, y_center, width, height.
    Returns:
      box: a `Tensor` whose shape is the same as `box` in new format.
    """
  with tf.name_scope('xcycwh_to_yxyx'):
    xy, wh = tf.split(box, 2, axis=-1)
    xy_min = xy - wh / 2
    xy_max = xy + wh / 2
    x_min, y_min = tf.split(xy_min, 2, axis=-1)
    x_max, y_max = tf.split(xy_max, 2, axis=-1)
    box = tf.concat([y_min, x_min, y_max, x_max], axis=-1)
    if split_min_max:
      box = tf.split(box, 2, axis=-1)
  return box


def xcycwh_to_xyxy(box: tf.Tensor, split_min_max: bool = False):
  """Converts boxes from x_center, y_center, width, height to xmin, ymin, xmax, ymax.
    Args:
      box: a `Tensor` whose last dimension is 4 representing the coordinates of boxes in
        x_center, y_center, width, height.
    Returns:
      box: a `Tensor` whose shape is the same as `box` in new format.
    """
  with tf.name_scope('xcycwh_to_yxyx'):
    xy, wh = tf.split(box, 2, axis=-1)
    xy_min = xy - wh / 2
    xy_max = xy + wh / 2
    box = (xy_min, xy_max)
    if not split_min_max:
      box = tf.concat(box, axis=-1)
  return box


def center_distance(center_1: tf.Tensor, center_2: tf.Tensor):
  """Calculates the squared distance between two points.
    Args:
      center_1: a `Tensor` that represents a point.
      center_2: a `Tensor` that represents a point.
    Returns:
      dist: a `Tensor` whose value represents the squared distance
        between center_1 and center_2.
    """
  with tf.name_scope('center_distance'):
    dist = (center_1[..., 0] - center_2[..., 0])**2 + (center_1[..., 1] -
                                                       center_2[..., 1])**2
  return dist

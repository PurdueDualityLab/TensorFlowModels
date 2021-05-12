""" bounding box utils file """

import math
from typing import Tuple, Union

# import libraries
import tensorflow as tf
import tensorflow.keras.backend as K

from yolo.ops import math_ops


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


# IOU
def compute_iou(box1, box2, yxyx=False):
  """Calculates the intersection of union between box1 and box2.
    Args:
        box1: a `Tensor` whose last dimension is 4 representing the coordinates of boxes in
            x_center, y_center, width, height.
        box2: a `Tensor` whose last dimension is 4 representing the coordinates of boxes in
            x_center, y_center, width, height.
    Returns:
        iou: a `Tensor` who represents the intersection over union.
    """
  # get box corners
  with tf.name_scope('iou'):
    if not yxyx:
      box1 = xcycwh_to_yxyx(box1)
      box2 = xcycwh_to_yxyx(box2)

    b1mi, b1ma = tf.split(box1, 2, axis=-1)
    b2mi, b2ma = tf.split(box2, 2, axis=-1)
    intersect_mins = tf.math.maximum(b1mi, b2mi)
    intersect_maxes = tf.math.minimum(b1ma, b2ma)
    intersect_wh = tf.math.maximum(intersect_maxes - intersect_mins,
                                   tf.zeros_like(intersect_mins))
    intersection = tf.reduce_prod(
        intersect_wh, axis=-1)  # intersect_wh[..., 0] * intersect_wh[..., 1]

    box1_area = tf.math.abs(tf.reduce_prod(b1ma - b1mi, axis=-1))
    box2_area = tf.math.abs(tf.reduce_prod(b2ma - b2mi, axis=-1))
    union = box1_area + box2_area - intersection

    iou = math_ops.divide_no_nan(intersection, union)
    iou = tf.clip_by_value(iou, clip_value_min=0.0, clip_value_max=1.0)
  return iou


def compute_giou(box1, box2, yxyx=False):
  """Calculates the generalized intersection of union between box1 and box2.
    Args:
        box1: a `Tensor` whose last dimension is 4 representing the coordinates of boxes in
            x_center, y_center, width, height.
        box2: a `Tensor` whose last dimension is 4 representing the coordinates of boxes in
            x_center, y_center, width, height.
    Returns:
        iou: a `Tensor` who represents the generalized intersection over union.
    """
  with tf.name_scope('giou'):
    # get box corners
    if not yxyx:
      box1 = xcycwh_to_yxyx(box1)
      box2 = xcycwh_to_yxyx(box2)

    # compute IOU
    b1mi, b1ma = tf.split(box1, 2, axis=-1)
    b2mi, b2ma = tf.split(box2, 2, axis=-1)
    intersect_mins = tf.math.maximum(b1mi, b2mi)
    intersect_maxes = tf.math.minimum(b1ma, b2ma)
    intersect_wh = tf.math.maximum(intersect_maxes - intersect_mins,
                                   tf.zeros_like(intersect_mins))
    intersection = tf.reduce_prod(
        intersect_wh, axis=-1)  # intersect_wh[..., 0] * intersect_wh[..., 1]

    box1_area = tf.math.abs(tf.reduce_prod(b1ma - b1mi, axis=-1))
    box2_area = tf.math.abs(tf.reduce_prod(b2ma - b2mi, axis=-1))
    union = box1_area + box2_area - intersection

    iou = math_ops.divide_no_nan(intersection, union)
    iou = tf.clip_by_value(iou, clip_value_min=0.0, clip_value_max=1.0)

    # find the smallest box to encompase both box1 and box2
    c_mins = tf.math.minimum(b1mi, b2mi)  
    c_maxes = tf.math.maximum(b1ma, b2ma)  
    c = tf.math.abs(tf.reduce_prod(c_mins - c_maxes, axis=-1))

    # compute giou
    regularization = math_ops.divide_no_nan((c - union), c)
    giou = iou - regularization
  return iou, giou


def compute_diou(box1, box2, yxyx=False):
  """Calculates the distance intersection of union between box1 and box2.
    Args:
        box1: a `Tensor` whose last dimension is 4 representing the coordinates of boxes in
            x_center, y_center, width, height.
        box2: a `Tensor` whose last dimension is 4 representing the coordinates of boxes in
            x_center, y_center, width, height.
    Returns:
        iou: a `Tensor` who represents the distance intersection over union.
    """
  with tf.name_scope('diou'):
    # compute center distance
    if yxyx:
      box1 = yxyx_to_xcycwh(box1)
      box2 = yxyx_to_xcycwh(box2)

    # dist = center_distance(box1[..., 0:2], box2[..., 0:2])
    dist = tf.reduce_sum((box1[..., 0:2] - box2[..., 0:2])**2, axis=-1)

    # get box corners
    box1 = xcycwh_to_yxyx(box1)
    box2 = xcycwh_to_yxyx(box2)

    # compute IOU
    b1mi, b1ma = tf.split(box1, 2, axis=-1)
    b2mi, b2ma = tf.split(box2, 2, axis=-1)
    intersect_mins = tf.math.maximum(b1mi, b2mi)
    intersect_maxes = tf.math.minimum(b1ma, b2ma)
    intersect_wh = tf.math.maximum(intersect_maxes - intersect_mins,
                                   tf.zeros_like(intersect_mins))
    intersection = tf.reduce_prod(intersect_wh, axis=-1)

    box1_area = tf.math.abs(tf.reduce_prod(b1ma - b1mi, axis=-1))
    box2_area = tf.math.abs(tf.reduce_prod(b2ma - b2mi, axis=-1))
    union = box1_area + box2_area - intersection

    iou = math_ops.divide_no_nan(intersection, union)
    iou = tf.clip_by_value(iou, clip_value_min=0.0, clip_value_max=1.0)

    # compute max diagnal of the smallest enclosing box
    c_mins = tf.math.minimum(b1mi, b2mi)  # box1[..., 0:2], box2[..., 0:2])
    c_maxes = tf.math.maximum(b1ma, b2ma)  # box1[..., 2:4], box2[..., 2:4])

    diag_dist = tf.reduce_sum((c_maxes - c_mins)**2, axis=-1)

    regularization = math_ops.divide_no_nan(dist, diag_dist)
    diou = iou - regularization
  return iou, diou


def compute_ciou(box1, box2, yxyx=False):
  """Calculates the complete intersection of union between box1 and box2.
    Args:
        box1: a `Tensor` whose last dimension is 4 representing the coordinates of boxes in
            x_center, y_center, width, height.
        box2: a `Tensor` whose last dimension is 4 representing the coordinates of boxes in
            x_center, y_center, width, height.
    Returns:
        iou: a `Tensor` who represents the complete intersection over union.
    """
  with tf.name_scope('ciou'):
    # compute DIOU and IOU

    iou, diou = compute_diou(box1, box2, yxyx=yxyx)

    if yxyx:
      box1 = yxyx_to_xcycwh(box1)
      box2 = yxyx_to_xcycwh(box2)

    # computer aspect ratio consistency
    arcterm = tf.square(
        tf.math.atan(math_ops.divide_no_nan(box1[..., 2], box1[..., 3])) -
        tf.math.atan(math_ops.divide_no_nan(box2[..., 2], box2[..., 3])))
    v = 4 * arcterm / (math.pi**2)

    a = math_ops.divide_no_nan(v, ((1 - iou) + v))
    ciou = diou - (v * a)
  return iou, ciou


def aggregated_comparitive_iou(boxes1, boxes2=None, iou_type=0, xyxy=True):
  k = tf.shape(boxes1)[-2]

  boxes1 = tf.expand_dims(boxes1, axis=-2)
  boxes1 = tf.tile(boxes1, [1, 1, k, 1])

  if boxes2 is not None:
    boxes2 = tf.expand_dims(boxes2, axis=-2)
    boxes2 = tf.tile(boxes2, [1, 1, k, 1])
    boxes2 = tf.transpose(boxes2, perm=(0, 2, 1, 3))
  else:
    boxes2 = tf.transpose(boxes1, perm=(0, 2, 1, 3))

  if iou_type == 0:  #diou
    _, iou = compute_diou(boxes1, boxes2, yxyx=True)
  elif iou_type == 1:  #giou
    _, iou = compute_giou(boxes1, boxes2, yxyx=True)
  else:
    iou = box_ops.compute_iou(boxes1, boxes2, yxyx=True)
  return iou

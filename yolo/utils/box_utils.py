import tensorflow as tf 
import tensorflow.keras.backend as K
import math
from typing import *

def _yxyx_to_xcycwh(box : tf.Tensor):
    """convert the box to the proper yolo format"""
    with tf.name_scope("yxyx_to_xcycwh"):
        ymin, xmin, ymax, xmax = tf.split(box, 4, axis = -1)
        x_center = (xmax + xmin)/2
        y_center = (ymax + ymin)/2
        width = xmax - xmin
        height = ymax - ymin
        box = tf.concat([x_center, y_center, width, height], axis = -1)
    return box

def _xcycwh_to_yxyx(box : tf.Tensor, split_min_max:bool  = False):
    with tf.name_scope("xcycwh_to_yxyx"):
        xy, wh = tf.split(box, 2, axis = -1)
        xy_min = xy - wh/2
        xy_max = xy + wh/2
        box = (tf.reverse(xy_min, axis = [-1]), tf.reverse(xy_max, axis = [-1]))
        if not split_min_max:
            box = tf.concat(box, axis = -1)
    return box   

def _xcycwh_to_xyxy(box : tf.Tensor, split_min_max:bool  = False):
    with tf.name_scope("xcycwh_to_yxyx"):
        xy, wh = tf.split(box, 2, axis = -1)
        xy_min = xy - wh/2
        xy_max = xy + wh/2
        box = (xy_min, xy_max)
        if not split_min_max:
            box = tf.concat(box, axis = -1)
    return box  

def _intersection_and_union(box1: tf.Tensor, box2: tf.Tensor):
    with tf.name_scope("intersection_and_union"):
        intersect_mins = K.maximum(box1[..., 0:2], box2[..., 0:2])
        intersect_maxes = K.minimum(box1[..., 2:4], box2[..., 2:4])
        intersect_wh = K.maximum(intersect_maxes - intersect_mins, K.zeros_like(intersect_mins))
        intersection = intersect_wh[..., 0] * intersect_wh[..., 1]

        box1_area = _get_area(box1)
        box2_area = _get_area(box2)
        union = box1_area + box2_area - intersection
    return intersection, union

def _get_area(box: Union[tf.Tensor, Tuple] , xywh:bool = False, use_tuple:bool  = False):
    with tf.name_scope("box_area"):
        if use_tuple:
            area = _get_area_tuple(box = box, xywh=xywh)
        else:
            area = _get_area_tensor(box = box, xywh=xywh)
    return area

def _get_area_tensor(box: tf.Tensor , xywh:bool  = False):
    with tf.name_scope("tensor_area"):
        if xywh: 
            area = tf.reduce_prod(box[..., 2:4], axis = -1)
        else:
            area = tf.math.abs(tf.reduce_prod(box[..., 2:4] - box[..., 0:2], axis = -1))
    return area

def _get_area_tuple(box: Tuple, xywh:bool = False):
    with tf.name_scope("tuple_area"):
        if xywh: 
            area = tf.reduce_prod(box[1], axis = -1)
        else:
            area = tf.math.abs(tf.reduce_prod(box[1] - box[0], axis = -1))
    return area

def _center_distance(center_1: tf.Tensor, center_2: tf.Tensor):
    with tf.name_scope("center_distance"):
        dist = (center_1[..., 0] - center_2[..., 0]) ** 2 + (center_1[..., 1] - center_2[..., 1]) ** 2
    return dist

def _aspect_ratio_consistancy(w_gt: tf.Tensor, h_gt: tf.Tensor, w: tf.Tensor, h: tf.Tensor):
    arcterm = (tf.math.atan(tf.math.divide_no_nan(w_gt,h_gt)) - tf.math.atan(tf.math.divide_no_nan(w,h))) ** 2
    return 4 * arcterm / (math.pi)**2
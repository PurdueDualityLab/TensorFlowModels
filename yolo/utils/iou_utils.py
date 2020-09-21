import math
import tensorflow.keras.backend as K
import tensorflow as tf

def box_iou(box_1, box_2, dtype = tf.float32):
    box1_xy = box_1[..., :2]
    box1_wh = box_1[..., 2:4]
    box1_mins = box1_xy - box1_wh / 2.
    box1_maxes = box1_xy + box1_wh / 2.

    box2_xy = box_2[..., :2]
    box2_wh = box_2[..., 2:4]
    box2_mins = box2_xy - box2_wh / 2.
    box2_maxes = box2_xy + box2_wh / 2.

    intersect_mins = K.maximum(box1_mins, box2_mins)
    intersect_maxes = K.minimum(box1_maxes, box2_maxes)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, K.zeros_like(intersect_mins))
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    box1_area = box1_wh[..., 0] * box1_wh[..., 1]
    box2_area = box2_wh[..., 0] * box2_wh[..., 1]
    iou = intersect_area/(box1_area + box2_area - intersect_area)
    iou = tf.where(tf.math.is_nan(iou), tf.cast(0.0, dtype = dtype), iou)
    return iou


def giou(box_1, box_2, dtype = tf.float32):
    box1_xy = box_1[..., :2]
    box1_wh = box_1[..., 2:4]
    box1_mins = box1_xy - box1_wh / 2.
    box1_maxes = box1_xy + box1_wh / 2.

    box2_xy = box_2[..., :2]
    box2_wh = box_2[..., 2:4]
    box2_mins = box2_xy - box2_wh / 2.
    box2_maxes = box2_xy + box2_wh / 2.

    intersect_mins = K.minimum(box1_mins, box2_mins)
    intersect_maxes = K.maximum(box1_maxes, box2_maxes)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
    C = intersect_wh[..., 0] * intersect_wh[..., 1]
    box1_area = box1_wh[..., 0] * box1_wh[..., 1]
    box2_area = box2_wh[..., 0] * box2_wh[..., 1]
    IOU = tf.convert_to_tensor(box_iou(box_1, box_2))
    giou = IOU - (C - (box1_area + box2_area) / (IOU + 1)) / C

    giou = tf.where(tf.math.is_nan(giou), 0.0, giou)
    giou = tf.where(tf.math.is_inf(giou), 0.0, giou)
    return giou


def ciou(box_1, box_2):
    ### NOT COMPLETED
    box1_xy = box_1[..., :2]
    box1_wh = box_1[..., 2:4]
    box1_mins = box1_xy - box1_wh / 2.
    box1_maxes = box1_xy + box1_wh / 2.

    box2_xy = box_2[..., :2]
    box2_wh = box_2[..., 2:4]
    box2_mins = box2_xy - box2_wh / 2.
    box2_maxes = box2_xy + box2_wh / 2.

    intersect_mins = K.minimum(box1_mins, box2_mins)
    intersect_maxes = K.maximum(box1_maxes, box2_maxes)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
    C = intersect_wh[..., 0] * intersect_wh[..., 1]
    box1_area = box1_wh[..., 0] * box1_wh[..., 1]
    box2_area = box2_wh[..., 0] * box2_wh[..., 1]
    IOU = tf.convert_to_tensor(box_iou(box_1, box_2))
    giou = IOU - (C - (box1_area + box2_area) / (IOU + 1)) / C

    giou = tf.where(tf.math.is_nan(giou), 0.0, giou)
    giou = tf.where(tf.math.is_inf(giou), 0.0, giou)
    return giou
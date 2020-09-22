import math
import tensorflow.keras.backend as K
import tensorflow as tf

def _distance(center_1, center_2):
    sqr_pt = K.square(center_1 - center_2)
    dist = tf.reduce_sum(sqr_pt, axis = -1)
    return dist

def _get_corners(box):
    x, y, w, h = tf.split(box, 4, axis = -1)
    x_min = x - w / 2
    x_max = x + w / 2
    y_min = y - h / 2
    y_max = y + h / 2
    return y_min, x_min, y_max, x_max

def _aspect_ratio_consistancy(w_gt, h_gt, w, h):
    arcterm = (tf.math.atan(w_gt/h_gt) - tf.math.atan(w/h)) ** 2
    return 4 * arcterm / (math.pi)**2

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

def diou(output, target):
    iou = box_iou(output, target)
    dist = _distance(output[..., 0:2], target[..., 0:2])

    ty_min, tx_min, ty_max, tx_max = _get_corners(target)
    y_min, x_min, y_max, x_max = _get_corners(output)
    xmin_diag = K.maximum(x_min, tx_min)
    xmax_diag = K.maximum(x_max, tx_max)
    ymin_diag = K.maximum(y_min, ty_min)
    ymax_diag = K.maximum(y_max, ty_max)
    diag_dist = ((xmax_diag - xmin_diag) ** 2) + ((ymax_diag - ymin_diag) ** 2) + 1e-16
    regularization = dist/diag_dist  
    return iou + regularization

def ciou(output, target):
    iou = box_iou(output, target)
    iou_reg = diou(output, target) 
    v = _aspect_ratio_consistancy(target[..., 3],target[..., 4], output [..., 3], output[..., 4])
    a = v/((1 - iou) + v)
    return iou_reg + v * a
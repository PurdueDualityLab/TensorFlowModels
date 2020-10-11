"""Preprocessing opeartions."""

# Import libraries
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow_addons.image import utils as img_utils

from yolo.utils.iou_utils import *


def _scale_image(image, resize=False, w=None, h=None):
    """Normalize the image to bound the pixel value betweeen 0 and 1.

    Args:
        image: a `Tensor` for image.
        resize: a `bool` that determines resizing.
        w: a `int` or `Tensor` for image width
        h: a `int` or `Tensor` for image height

    Returns:
        image: a `Tensor` for normalized image.
    """
    with tf.name_scope("scale_image"):
        image = tf.convert_to_tensor(image)
        if resize:
            image = tf.image.resize(image, size=(w, h))
        image = image / 255
    return image

def random_jitter_boxes(boxes, box_jitter, seed = 10):
    """Performs random jitter on boxes.

    Args:
        boxes: a `Tensor` whose last dimension is 4 representing the coordinates of boxes in
            x_center, y_center, width, height.
        box_jitter: a `float` that is the maximum jitter applied to the bounding
            box for data augmentation during training.
        seed: an `int` for the seed used by tf.random
    
    Returns:
        boxes: a `Tensor` whose shape is the same as `boxes`.
    """
    num_gen = tf.shape(boxes)[0]
    jx = tf.random.uniform(minval=-box_jitter,
                             maxval=box_jitter,
                             shape=(num_gen, ),
                             seed = seed, 
                             dtype=tf.float32)
    jy = tf.random.uniform(minval=-box_jitter,
                             maxval=box_jitter,
                             shape=(num_gen, ),
                             seed = seed, 
                             dtype=tf.float32)
    jw = tf.random.uniform(minval=-box_jitter,
                             maxval=box_jitter,
                             shape=(num_gen, ),
                             seed = seed, 
                             dtype=tf.float32)+1
    jh = tf.random.uniform(minval=-box_jitter,
                             maxval=box_jitter,
                             shape=(num_gen, ),
                             seed = seed, 
                             dtype=tf.float32)+1
    #tf.print(tf.math.reduce_sum(jx))
    boxes = _jitter_boxes(boxes, jx, jy, jw, jh)
    return boxes

def _jitter_boxes(box, j_cx, j_cy, j_w, j_h):
    """Jittering the boxes."""
    with tf.name_scope("jitter_boxs"):
        x = tf.clip_by_value(tf.math.add(box[..., 0], j_cx), clip_value_min=0.0, clip_value_max=1.0)
        y = tf.clip_by_value(tf.math.add(box[..., 1], j_cy), clip_value_min=0.0, clip_value_max=1.0)
        w = box[..., 2] * j_w
        h = box[..., 3] * j_h
        box = tf.stack([x, y, w, h], axis = -1)
        box.set_shape([None, 4])
    return box

def random_translate(image, box, t, seed = 10):
    """Performs random translate on the image and the box.

    Args:
        image: image: a `Tensor` for image.
        box: a `Tensor` whose last dimension is 4 representing the coordinates of boxes in
            x_center, y_center, width, height.
        t: a `float` that is the maximum translation applied to the bounding
            box for data augmentation during training.
        seed: an `int` for the seed used by tf.random
    
    Returns:
        image: a `Tensor` whose shape is the same as `image`.
        box: a `Tensor` whose shape is the same as `box`.
    """
    t_x = tf.random.uniform(minval=-t,
                            maxval=t,
                            shape=(),
                            dtype=tf.float32)
    t_y = tf.random.uniform(minval=-t,
                            maxval=t,
                            shape=(),
                            dtype=tf.float32)
    #tf.print(t_x+t_y)
    box = _translate_boxes(box, t_x, t_y)
    image = _translate_image(image, t_x, t_y)
    return image, box

def _translate_boxes(box, translate_x, translate_y):
    """Translating the boxes"""
    with tf.name_scope("translate_boxs"):
        x = box[..., 0] + translate_x
        y = box[..., 1] + translate_y
        box = tf.stack([x, y, box[..., 2], box[..., 3]], axis = -1)
        box.set_shape([None, 4])
    return box

def _translate_image(image, translate_x, translate_y):
    """Translating the image"""
    with tf.name_scope("translate_image"):
        if (translate_x != 0 and translate_y != 0):
            image_jitter = tf.convert_to_tensor([translate_x, translate_y])
            image_jitter.set_shape([2])
            image = tfa.image.translate(
                image, image_jitter * tf.cast(tf.shape(image)[1], tf.float32))
    return image

def random_flip(image, box, seed = 10):
    """Randomly flips input image and bounding boxes.

    Args:
        image: image: a `Tensor` for image.
        box: a `Tensor` whose last dimension is 4 representing the coordinates of boxes in
            x_center, y_center, width, height.
        seed: an `int` for the seed used by tf.random
    
    Returns:
        image: a `Tensor` whose shape is the same as `image`.
        box: a `Tensor` whose shape is the same as `box`.
    """
    do_flip_x = tf.greater(tf.random.uniform([], seed=seed), 0.5)
    x = box[..., 0]
    y = box[..., 1]
    if do_flip_x:
        image = tf.image.flip_left_right(image)
        x = 1 - x
    do_flip_y = tf.greater(tf.random.uniform([], seed=seed), 0.5)
    if do_flip_y:
        image = tf.image.flip_up_down(image)
        y = 1 - y
    box = tf.stack([x, y, box[..., 2], box[..., 3]], axis = -1)
    return image, box

def pad_max_instances(value, instances, pad_value = 0):
    """Pads data to a fixed length at the first dimension.
    
    Args:
        value: `Tensor` with any dimension.
        instances: `int` number for the first dimension of output Tensor.
        pad_value: `int` value assigned to the paddings.

    Returns:
        value: `Tensor` with the first dimension padded to `size`.
    """
    shape = tf.shape(value)
    dim1 = shape[0]

    if dim1 > instances: 
        value = value[:instances, ...]
        return value
    else: 
        nshape = tf.tensor_scatter_nd_update(shape, [[0]], [instances - dim1])
        pad_tensor = tf.ones(nshape, dtype=value.dtype) * pad_value
        value = tf.concat([value, pad_tensor], axis = 0)
        return value 

def _get_best_anchor(y_true, anchors, width, height):
    """Get the correct anchor that is assoiciated with each box
    using IOU between input anchors and ground truth.

    Args:
        y_true: a `Tensor` for the list of bounding boxes in the yolo format.
        anchors: a `list` or `Tensor` for the anchor boxes to be used in prediction found via Kmeans.
        width: a `Tensor` or `int` for width of input image.
        height: a `Tensor` or `int` for height of input image.

    return:
        best_anchor: y_true with the anchor associated with each ground truth box known
    """
    with tf.name_scope("get_anchor"):
        width = tf.cast(width, dtype=tf.float32)
        height = tf.cast(height, dtype=tf.float32)

        anchor_xy = y_true[..., 0:2]
        true_wh = y_true[..., 2:4]

        # scale thhe boxes
        anchors = tf.convert_to_tensor(anchors, dtype=tf.float32)
        anchors_x = anchors[..., 0]/width
        anchors_y = anchors[..., 1]/height
        anchors = tf.stack([anchors_x, anchors_y], axis = -1)

        # build a matrix of anchor boxes
        anchors = tf.transpose(anchors, perm=[1, 0])
        anchor_xy = tf.tile(tf.expand_dims(anchor_xy, axis=-1),
                            [1, 1, tf.shape(anchors)[-1]])
        anchors = tf.tile(tf.expand_dims(anchors, axis=0),
                          [tf.shape(anchor_xy)[0], 1, 1])

        # stack the xy so, each anchor is asscoaited once with each center from the ground truth input
        anchors = tf.concat([anchor_xy, anchors], axis=1)
        anchors = tf.transpose(anchors, perm=[2, 0, 1])

        # copy the gt n times so that each anchor from above can be compared to input ground truth
        truth_comp = tf.tile(tf.expand_dims(y_true[..., 0:4], axis=-1),
                             [1, 1, tf.shape(anchors)[0]])
        truth_comp = tf.transpose(truth_comp, perm=[2, 0, 1])

        # compute intersection over union of the boxes, and take the argmax of comuted iou for each box.
        # thus each box is associated with the largest interection over union
        iou_raw = compute_iou(truth_comp, anchors)

        gt_mask = tf.cast(iou_raw > 0.213, dtype=iou_raw.dtype)

        num_k = tf.reduce_max(
            tf.reduce_sum(tf.transpose(gt_mask, perm=[1, 0]), axis=1))
        if num_k <= 0:
            num_k = 1.0

        values, indexes = tf.math.top_k(tf.transpose(iou_raw, perm=[1, 0]),
                                        k=tf.cast(num_k, dtype=tf.int32),
                                        sorted=True)
        ind_mask = tf.cast(values > 0.213, dtype=indexes.dtype)
        iou_index = tf.concat([
            tf.expand_dims(indexes[..., 0], axis=-1),
            ((indexes[..., 1:] + 1) * ind_mask[..., 1:]) - 1
        ], axis=-1)

        stack = tf.zeros(
            [tf.shape(iou_index)[0],
             tf.cast(1, dtype=iou_index.dtype)],
            dtype=iou_index.dtype) - 1
        while num_k < 5:
            iou_index = tf.concat([iou_index, stack], axis=-1)
            num_k += 1
        iou_index = iou_index[..., :5]

        values = tf.concat([
            tf.expand_dims(values[..., 0], axis=-1),
            ((values[..., 1:]) * tf.cast(ind_mask[..., 1:], dtype=tf.float32))
        ], axis=-1)
        best_anchor = tf.cast(iou_index, dtype=tf.float32)
    return best_anchor

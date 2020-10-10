import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow_addons.image import utils as img_utils
import tensorflow.keras.backend as K
from yolo.utils.iou_utils import *


@tf.function
def _angles_to_projective_transforms(angle, image_w, image_h):
    """Generate projective transform matrix for tfa.image.transform.
    Args:
        angle(tensorflow.python.framework.ops.EagerTensor): The rotation angle.
        image_w(tensorflow.python.framework.ops.EagerTensor): The width of the image.
        image_h(tensorflow.python.framework.ops.EagerTensor): The height of the image.
    Returns:
        projective transform matrix(tensorflow.python.framework.ops.EagerTensor)
    """
    with tf.name_scope("rotate_parent"):
        angle_or_angles = tf.convert_to_tensor(angle,
                                               name="angles",
                                               dtype=tf.dtypes.float32)
        angles = angle_or_angles[None]
        x_offset = ((image_w - 1) - (tf.math.cos(angles) *
                                     (image_w - 1) - tf.math.sin(angles) *
                                     (image_h - 1))) / 2.0
        y_offset = ((image_h - 1) - (tf.math.sin(angles) *
                                     (image_w - 1) + tf.math.cos(angles) *
                                     (image_h - 1))) / 2.0
        num_angles = tf.shape(angles)[0]
    return tf.concat([
        tf.math.cos(angles)[:, None], -tf.math.sin(angles)[:, None],
        x_offset[:, None],
        tf.math.sin(angles)[:, None],
        tf.math.cos(angles)[:, None], y_offset[:, None],
        tf.zeros((1, 2))
    ],
                     axis=1)


@tf.function
def _rotate(image, angle):
    """Generates a rotated image with the use of tfa.image.transform
    Args:
        image(tensorflow.python.framework.ops.Tensor): The image.
        angle(tensorflow.python.framework.ops.EagerTensor): The rotation angle.
    Returns:
        The rotated image.
    """
    with tf.name_scope("rotate"):
        image = tf.convert_to_tensor(image)
        img = img_utils.to_4D_image(image)
        ndim = image.get_shape().ndims
        image_h = tf.cast(img.shape[0], tf.dtypes.float32)
        image_w = tf.cast(img.shape[1], tf.dtypes.float32)
        rotation_key = _angles_to_projective_transforms(
            angle, image_w, image_h)
        output = tfa.image.transform(img,
                                     rotation_key,
                                     interpolation="NEAREST")
    return img_utils.from_4D_image(output, ndim)

def _scale_image(image, resize=False, w=None, h=None):
    """Image Normalization.
    Args:
        image(tensorflow.python.framework.ops.Tensor): The image.
    Returns:
        A Normalized Function.
    """
    with tf.name_scope("scale_image"):
        image = tf.convert_to_tensor(image)
        if resize:
            image = tf.image.resize(image, size=(w, h))
        image = image / 255
    return image

def random_jitter_boxes(boxes, box_jitter, seed = 10):
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
    with tf.name_scope("jitter_boxs"):
        x = tf.clip_by_value(tf.math.add(box[..., 0], j_cx), clip_value_min=0.0, clip_value_max=1.0)
        y = tf.clip_by_value(tf.math.add(box[..., 1], j_cy), clip_value_min=0.0, clip_value_max=1.0)
        w = box[..., 2] * j_w
        h = box[..., 3] * j_h
        box = tf.stack([x, y, w, h], axis = -1)
        box.set_shape([None, 4])
    return box

def random_translate(image, box, t, seed = 10):
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
    with tf.name_scope("translate_boxs"):
        x = box[..., 0] + translate_x
        y = box[..., 1] + translate_y
        box = tf.stack([x, y, box[..., 2], box[..., 3]], axis = -1)
        box.set_shape([None, 4])
    return box

def _translate_image(image, translate_x, translate_y):
    with tf.name_scope("translate_image"):
        if (translate_x != 0 and translate_y != 0):
            image_jitter = tf.convert_to_tensor([translate_x, translate_y])
            image_jitter.set_shape([2])
            image = tfa.image.translate(
                image, image_jitter * tf.cast(tf.shape(image)[1], tf.float32))
    return image

def random_flip(image, box, seed = 10):
    do_flip_x = tf.greater(tf.random.uniform([], seed=seed), 0.5)
    x = box[..., 0]
    y = box[..., 1]
    if do_flip_x:
        image = tf.image.flip_left_right(image)
        x = 1 - x
    # do_flip_y = tf.greater(tf.random.uniform([], seed=seed), 0.5)
    # if do_flip_y:
    #     image = tf.image.flip_up_down(image)
    #     y = 1 - y
    return image, tf.stack([x, y, box[..., 2], box[..., 3]], axis = -1)

def pad_max_instances(value, instances, pad_value = 0):
    shape = tf.shape(value)
    dim1 = shape[0]

    if dim1 > instances: 
        value = value[:instances, ...]
        return value
    else: 
        nshape = tf.tensor_scatter_nd_update(shape, [[0]], [instances - dim1])# 
        pad_tensor = tf.ones(nshape, dtype=value.dtype) * pad_value
        value = tf.concat([value, pad_tensor], axis = 0)
        return value 

def _get_best_anchor(y_true, anchors, width, height):
    """
    get the correct anchor that is assoiciated with each box using IOU betwenn input anchors and gt
    Args:
        y_true: tf.Tensor[] for the list of bounding boxes in the yolo format
        anchors: list or tensor for the anchor boxes to be used in prediction found via Kmeans
        size: size of the image that the bounding boxes were selected at 416 is the default for the original YOLO model
    return:
        tf.Tensor: y_true with the anchor associated with each ground truth box known
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
        anchors = K.concatenate([anchor_xy, anchors], axis=1)
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
            K.expand_dims(indexes[..., 0], axis=-1),
            ((indexes[..., 1:] + 1) * ind_mask[..., 1:]) - 1
        ],
                              axis=-1)

        stack = tf.zeros(
            [tf.shape(iou_index)[0],
             tf.cast(1, dtype=iou_index.dtype)],
            dtype=iou_index.dtype) - 1
        #tf.print(tf.shape(iou_index))
        while num_k < 5:
            iou_index = tf.concat([iou_index, stack], axis=-1)
            num_k += 1
        iou_index = iou_index[..., :5]

        values = tf.concat([
            K.expand_dims(values[..., 0], axis=-1),
            ((values[..., 1:]) * tf.cast(ind_mask[..., 1:], dtype=tf.float32))
        ],
                           axis=-1)
        # iou_anchors = K.argmax(iou_raw, axis = 0)
        # iou_anchors = K.expand_dims(tf.cast(iou_anchors, dtype = tf.float32), axis = -1)
        # tf.print(iou_index, values)
        #flatten the list from above and attach to the end of input y_true, then return it
        #y_true = K.concatenate([y_true, K.expand_dims(iou_anchors, axis = -1)], axis = -1)
    return tf.cast(iou_index, dtype=tf.float32)

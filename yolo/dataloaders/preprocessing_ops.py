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
        angle_or_angles = tf.convert_to_tensor(angle, name="angles", dtype=tf.dtypes.float32)
        angles = angle_or_angles[None]
        x_offset = ((image_w - 1) - (tf.math.cos(angles) * (image_w - 1) - tf.math.sin(angles) * (image_h - 1))) / 2.0
        y_offset = ((image_h - 1)- (tf.math.sin(angles) * (image_w - 1) + tf.math.cos(angles) * (image_h - 1))) / 2.0
        num_angles = tf.shape(angles)[0]
    return tf.concat([tf.math.cos(angles)[:, None],-tf.math.sin(angles)[:, None],x_offset[:, None],tf.math.sin(angles)[:, None],tf.math.cos(angles)[:, None],y_offset[:, None],tf.zeros((1, 2))],axis=1)

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
        rotation_key = _angles_to_projective_transforms(angle, image_w, image_h)
        output = tfa.image.transform(img, rotation_key, interpolation="NEAREST")
    return img_utils.from_4D_image(output, ndim)

def _scale_image(image, square = False, square_w = None):
    """Image Normalization.
    Args:
        image(tensorflow.python.framework.ops.Tensor): The image.
    Returns:
        A Normalized Function.
    """
    with tf.name_scope("scale_image"):
        image = tf.convert_to_tensor(image)
        if square: 
            image = tf.image.resize(image, size = (square_w, square_w))
        image = image / 255
    return image

def _get_yolo_box(box):
    """convert the box to the proper yolo format"""
    with tf.name_scope("yolo_box"):
        ymin, xmin, ymax, xmax = tf.split(box, 4, axis = -1)
        x_center = (xmax + xmin)/2
        y_center = (ymax + ymin)/2
        width = xmax - xmin
        height = ymax - ymin
        box = tf.concat([x_center, y_center, width, height], axis = -1)
    return box

def _get_tf_box(box):
    with tf.name_scope("tf_box"):
        x, y, w, h = tf.split(box, 4, axis = -1)
        x_min = x - w/2
        y_min = y - h/2
        x_max = x + w/2
        y_max = y + h/2
        box = tf.concat([y_min, x_min, y_max, x_max], axis = -1)
    return box
def _jitter_boxes(box, translate_x, translate_y, j_cx, j_cy, j_w, j_h):
    with tf.name_scope("jitter_boxs"):

        x = tf.nn.relu(tf.math.add(box[..., 0], j_cx))
        if translate_x != 0.0:
            x = tf.math.add(x, translate_x)
        x = tf.expand_dims(x, axis = -1)

        y = tf.nn.relu(tf.math.add(box[..., 1], j_cy))
        if translate_y != 0.0:
            y = tf.math.add(y, translate_y)
        y = tf.expand_dims(y, axis = -1)

        w = box[..., 2] * j_w
        w = tf.expand_dims(w, axis = -1)
        h = box[..., 3] * j_h
        h = tf.expand_dims(h, axis = -1)
        box = tf.concat([x, y, w, h], axis = -1)
    return box

def _translate_image(image, translate_x, translate_y):
    with tf.name_scope("translate_image"):
        if (translate_x != 0 and translate_y != 0):
            image_jitter = tf.concat([translate_x, translate_y], axis = 0)
            image_jitter.set_shape([2])
            image = tfa.image.translate(image, image_jitter * tf.cast(tf.shape(image)[1], tf.float32))
    return image

def _get_best_anchor(y_true, anchors, size):
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
        size = tf.cast(size, dtype = tf.float32)

        anchor_xy = y_true[..., 0:2]
        true_wh = y_true[..., 2:4]

        # scale thhe boxes 
        anchors = tf.convert_to_tensor(anchors, dtype=tf.float32)/size
        
        # build a matrix of anchor boxes
        anchors = tf.transpose(anchors, perm=[1, 0])
        anchor_xy = tf.tile(tf.expand_dims(anchor_xy, axis = -1), [1,1, tf.shape(anchors)[-1]])
        anchors = tf.tile(tf.expand_dims(anchors, axis = 0), [tf.shape(anchor_xy)[0], 1, 1])
        
        # stack the xy so, each anchor is asscoaited once with each center from the ground truth input
        anchors = K.concatenate([anchor_xy, anchors], axis = 1)
        anchors = tf.transpose(anchors, perm = [2, 0, 1])

        # copy the gt n times so that each anchor from above can be compared to input ground truth 
        truth_comp = tf.tile(tf.expand_dims(y_true[..., 0:4], axis = -1), [1,1, tf.shape(anchors)[0]])
        truth_comp = tf.transpose(truth_comp, perm = [2, 0, 1])

        # compute intersection over union of the boxes, and take the argmax of comuted iou for each box. 
        # thus each box is associated with the largest interection over union 
        iou_anchors = K.expand_dims(tf.cast(K.argmax(box_iou(truth_comp, anchors), axis = 0), dtype = tf.float32), axis = -1)

        #flatten the list from above and attach to the end of input y_true, then return it
        #y_true = K.concatenate([y_true, K.expand_dims(iou_anchors, axis = -1)], axis = -1)
    return iou_anchors

@tf.function
def _build_grided_gt(y_true, mask, size, use_tie_breaker):
    """
    convert ground truth for use in loss functions
    Args: 
        y_true: tf.Tensor[] ground truth [box coords[0:4], classes_onehot[0:-1], best_fit_anchor_box]
        mask: list of the anchor boxes choresponding to the output, ex. [1, 2, 3] tells this layer to predict only the first 3 anchors in the total. 
        size: the dimensions of this output, for regular, it progresses from 13, to 26, to 52
    
    Return:
        tf.Tensor[] of shape [batch, size, size, #of_anchors, 4, 1, num_classes]
    """
    batches = tf.shape(y_true)[0]
    num_boxes = tf.shape(y_true)[1]
    len_masks = tf.shape(mask)[0]

    full = tf.zeros([batches, size, size, len_masks, tf.shape(y_true)[-1]])
    #if use_tie_breaker:
    depth_track = tf.zeros((batches, size, size, len_masks), dtype=tf.int32)

    x = tf.cast(y_true[..., 0] * tf.cast(size, dtype = tf.float32), dtype = tf.int32)
    y = tf.cast(y_true[..., 1] * tf.cast(size, dtype = tf.float32), dtype = tf.int32)

    anchors = tf.repeat(tf.expand_dims(y_true[..., -1], axis = -1), len_masks, axis = -1)

    update_index = tf.TensorArray(tf.int32, size=0, dynamic_size=True)
    update = tf.TensorArray(tf.float32, size=0, dynamic_size=True)

    i = 0
    for batch in range(batches):
        for box_id in range(num_boxes):
            if K.all(tf.math.equal(y_true[batch, box_id, 2:4], 0)):
                continue
            if K.any(tf.math.less(y_true[batch, box_id, 0:2], 0.0)) or K.any(tf.math.greater_equal(y_true[batch, box_id, 0:2], 1.0)): 
                continue
            index = tf.math.equal(anchors[batch, box_id], mask)
            if K.any(index):
                p = tf.cast(K.argmax(tf.cast(index, dtype = tf.int32)), dtype = tf.int32)
                
                if use_tie_breaker:
                    # find the index of the box
                    uid = 1
                    used = depth_track[batch, y[batch, box_id], x[batch, box_id], p]
                    count = 0
                    # check if the next anchor is used used == 1, if so find another box 
                    while tf.math.equal(used, 1) and tf.math.less(count, 3):
                        uid = 2
                        count += 1
                        p = (p + 1)%3
                        used = depth_track[batch, x[batch, box_id], y[batch, box_id], p]
                    if tf.math.equal(used, 1):
                        tf.print("skipping")
                        continue
                    # set the current index to used  = 2, to indicate that it is occupied by something that should not be there, so if another box fits that anchor
                    # it will be prioritized over the current box.
                    depth_track = tf.tensor_scatter_nd_update(depth_track, [(batch, y[batch, box_id], x[batch, box_id], p)], [uid])

                # write the box to the update list 
                # the boxes output from yolo are for some reason have the x and y indexes swapped for some reason, I am not sure why 
                """peculiar"""
                update_index = update_index.write(i, [batch, y[batch, box_id], x[batch, box_id], p])
                value = K.concatenate([y_true[batch, box_id, 0:4], tf.convert_to_tensor([1.]), y_true[batch, box_id, 4:-1]])
                update = update.write(i, value)
                i += 1

            """
            used can be:
                0 not used
                1 used with the correct anchor
                2 used with offset anchor
            if used is 0 or 2:
                do not enter tie breaker (count = 0)
                edit box index with the most recent box
            if tie breaker was used:
                set used to 2
            else:
                set used to 1
            E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:741] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
            raised likely due to a memory issue? reduced batch size to 2 and it solved the problem? odd
            W tensorflow/core/grappler/optimizers/loop_optimizer.cc:906] Skipping loop optimization for Merge node with control input: cond/branch_executed/_11
            idk should look into this
            18 seconds for 2000 images
            """

    # if the size of the update list is not 0, do an update, other wise, no boxes and pass an empty grid
    if tf.math.greater(update_index.size(), 0):
        update_index = update_index.stack()
        update = update.stack()
        full = tf.tensor_scatter_nd_add(full, update_index, update)
    return full
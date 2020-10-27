import tensorflow as tf


def build_gridded_gt_v1(y_true: dict, num_classes: int, size: int,
                        num_boxes: int, dtype=tf.float64) -> tf.Tensor:
    """
    Convert ground truth for use in loss functions.
    Args:
        y_true: dict[tf.Tensor] containing 'bbox':boxes, 'label':classes
        num_classes: number of classes
        size: dimensions of grid S*S
        dtype: type of float
        num_boxes: number of boxes in each grid
    Return:
        tf.Tensor[batch, size, size, num_boxes, num_classes + 5]
    """
    boxes = tf.cast(y_true["bbox"], dtype)  # [xcenter, ycenter, width, height]
    classes = tf.one_hot(tf.cast(y_true["label"], dtype=tf.int32),
                         depth=num_classes, dtype=dtype)
    batches = boxes.get_shape().as_list()[0]
    gt_boxes = boxes.get_shape().as_list()[1]
    full = tf.zeros([batches, size, size, num_boxes,
                     num_classes + 4 + 1], dtype=dtype)
    # x centered coords
    x = tf.cast(boxes[..., 0] * tf.cast(size, dtype=dtype), dtype=tf.int32)
    # y centered coords
    y = tf.cast(boxes[..., 1] * tf.cast(size, dtype=dtype), dtype=tf.int32)
    i = 0  # index for tensor array
    full_confidence = tf.cast(tf.convert_to_tensor([1.]), dtype=dtype)
    update_index = tf.TensorArray(tf.int32, size=0, dynamic_size=True)
    update = tf.TensorArray(dtype, size=0, dynamic_size=True)

    tf.print(boxes)
    for batch in range(batches):
        for box_id in range(gt_boxes):
            if tf.math.reduce_all(tf.math.equal(boxes[batch, box_id, 2:4], 0),
                                  axis=None):
                continue
            if tf.math.reduce_any(tf.math.less(
                                  boxes[batch, box_id, 0:2], 0.0)):
                continue
            if tf.math.reduce_any(tf.math.greater_equal(
                                  boxes[batch, box_id, 0:2], 1.0)):
                continue
            for n in range(num_boxes):
                # index for box in the output [batch, x, y, box number]
                update_index = update_index.write(i,
                                                  [batch,
                                                   y[batch, box_id],
                                                   x[batch, box_id], n])
                # value of box to be updated: [x, y, w, h, confidence, classes]
                value = tf.concat([boxes[batch, box_id],
                                   full_confidence,
                                   classes[batch, box_id]],
                                  -1)
                update = update.write(i, value)
                i += 1
    # if no updates, return empty grid
    if tf.math.greater(update_index.size(), 0):
        update_index = update_index.stack()
        update = update.stack()
        full = tf.tensor_scatter_nd_add(full, update_index, update)
    return full

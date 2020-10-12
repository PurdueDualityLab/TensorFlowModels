import tensorflow as tf


@tf.function
def build_gridded_gt_v1(y_true: dict, size: int, dtype) -> tf.Tensor:
    """
    Convert ground truth for use in loss functions.
    Args:
        y_true: dict[tf.Tensor] containing 'bbox':boxes, 'classes':classes
        size: dimensions of grid
    Return:
        tf.Tensor[batch, size, size, 4, 1, num_classes]
    """
    boxes = tf.cast(y_true["bbox"], dtype)  # [xcenter, ycenter, width, height]
    classes = tf.one_hot(tf.cast(y_true["classes"], dtype=tf.int32),
                         depth=classes, dtype=dtype)
    batches = tf.shape(boxes)[0]
    num_boxes = tf.shape(boxes)[1]
    full = tf.zeros([batches, size, size, tf.shape(classes)[0]], dtype=dtype)
    # x centered coords
    x = tf.cast(boxes[..., 0] * tf.cast(size, dtype=dtype), dtype=tf.int32)
    # y centered coords
    y = tf.cast(boxes[..., 1] * tf.cast(size, dtype=dtype), dtype=tf.int32)
    i = 0
    full_confidence = tf.cast(tf.convert_to_tensor([1.]), dtype=dtype)
    for batch in range(batches):
        for box_id in range(num_boxes):
            if tf.math.reduce_all(tf.math.equal(boxes[batch, box_id, 2:4], 0)):
                continue
            if tf.math.reduce_any(tf.math.less(boxes[batch, box_id, 0:2], 0.0)) or tf.math.reduce_any(tf.math.greater_equal(boxes[batch, box_id, 0:2], 1.0)):
                continue
            update_index = update_index.write(i, [batch, y[batch, box_id], x[batch, box_id]])
            value = tf.concat([boxes[batch, box_id], full_confidence, classes[batch, box_id]])
            update = update.write(i, value)
            i += 1
    # if no updates, return empty grid
    if tf.math.greater(update_index.size(), 0):
        update_index = update_index.stack()
        update = update.stack()
        full = tf.tensor_scatter_nd_add(full, update_index, update)
    return full

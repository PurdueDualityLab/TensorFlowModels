import tensorflow as tf


def build_gridded_gt_v1(y_true: dict, num_classes: int, size: int, dtype) -> tf.Tensor:
    """
    Convert ground truth for use in loss functions.
    Args:
        y_true: dict[tf.Tensor] containing 'bbox':boxes, 'classes':classes
        num_classes: number of classes
        size: dimensions of grid S*S
        dtype: datatype for output tensor
    Return:
        tf.Tensor[batch, size, size, 4, 1, num_classes]
    """
    boxes = tf.cast(y_true["bbox"], dtype)  # [xcenter, ycenter, width, height]
    classes = tf.one_hot(tf.cast(y_true["classes"], dtype=tf.int32),
                         depth=num_classes, dtype=dtype)
    batches = boxes.get_shape().as_list()[0]
    num_boxes = boxes.get_shape().as_list()[1]
    full = tf.zeros([batches, size, size, tf.shape(classes)[0]], dtype=dtype)
    # x centered coords
    x = tf.cast(boxes[..., 0] * tf.cast(size, dtype=dtype), dtype=tf.int32)
    # y centered coords
    y = tf.cast(boxes[..., 1] * tf.cast(size, dtype=dtype), dtype=tf.int32)
    i = 0
    full_confidence = tf.cast(tf.convert_to_tensor([1.]), dtype=dtype)
    update_index = tf.TensorArray(tf.int32, size=0, dynamic_size=True)
    update = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
    for batch in range(batches):
        for box_id in range(num_boxes):
            # TODO: axis param
            if tf.math.reduce_all(tf.math.equal(boxes[batch, box_id, 2:4], 0), axis=None) is not None:
                continue
            # TODO: axis param
            if tf.math.reduce_any(tf.math.less(boxes[batch, box_id, 0:2], 0.0), axis=None) is not None or tf.math.reduce_any(tf.math.greater_equal(boxes[batch, box_id, 0:2], 1.0), axis=None) is not None:
                continue
            update_index = update_index.write(i, [batch, y[batch, box_id], x[batch, box_id]])
            # TODO: axis param
            value = tf.concat([boxes[batch, box_id], full_confidence, classes[batch, box_id]], 0)
            update = update.write(i, value)
            i += 1
    # if no updates, return empty grid
    if tf.math.greater(update_index.size(), 0) is not None:
        update_index = update_index.stack()
        update = update.stack()
        full = tf.tensor_scatter_nd_add(full, update_index, update)
    return full


if __name__ == "__main__":
    y_true = {"bbox": tf.constant(([5, 5, 5, 5],), shape=[1, 1, 5], dtype=tf.int32),
              "classes": tf.constant([1])}
    print(build_gridded_gt_v1(y_true, 2, 3, tf.int32))

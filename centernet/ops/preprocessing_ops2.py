def draw_gaussian(heatmap, blobs, scaling_factor=1, dtype=tf.float32):
    """
    Draws a gaussian heatmap around a center point given a radius.
    Params:
        heatmap (tf.Tensor): heatmap placeholder to fill
        blobs (tf.Tensor): a tensor whose last dimension is 4 integers for
          the category of the object, center (x, y), and for radius of the
          gaussian
        scaling_factor (int): scaling factor for gaussian
    """
    blobs = tf.cast(blobs, tf.int32)
    category = blobs[..., 0]
    x = blobs[..., 1]
    y = blobs[..., 2]
    radius = blobs[..., 3]

    diameter = 2 * radius + 1

    num_boxes = tf.shape(radius)[0]

    heatmap_shape = tf.shape(heatmap)
    height, width = heatmap_shape[-2], heatmap_shape[-1]

    left, right = tf.math.minimum(x, radius), tf.math.minimum(width - x, radius + 1)
    top, bottom = tf.math.minimum(y, radius), tf.math.minimum(height - y, radius + 1)

    update_count = tf.reduce_sum((bottom + top) * (right + left))
    masked_gaussian_ta = tf.TensorArray(dtype, size=update_count)
    heatmap_mask_ta = tf.TensorArray(tf.int32, element_shape=tf.TensorShape((3,)), size=update_count)
    i = 0
    for j in range(num_boxes):
      cat = category[j]
      X = x[j]
      Y = y[j]
      R = radius[j]
      l = left[j]
      r = right[j]
      t = top[j]
      b = bottom[j]

      gaussian = _gaussian_penalty(R, dtype=dtype)
      masked_gaussian_instance = tf.reshape(gaussian[R - t:R + b, R - l:R + r], (-1,))
      heatmap_mask_instance = cartesian_product([cat], tf.range(Y - t, Y + b), tf.range(X - l, X + r))
      masked_gaussian_ta, _ = write_all(masked_gaussian_ta, i, masked_gaussian_instance)
      heatmap_mask_ta, i = write_all(heatmap_mask_ta, i, heatmap_mask_instance)
    masked_gaussian = masked_gaussian_ta.stack()
    heatmap_mask = heatmap_mask_ta.stack()
    heatmap_mask = tf.reshape(heatmap_mask, (-1, 3))
    heatmap = tf.tensor_scatter_nd_max(heatmap, heatmap_mask, tf.cast(masked_gaussian * scaling_factor, heatmap.dtype))
    return heatmap

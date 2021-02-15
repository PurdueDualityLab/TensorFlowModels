import tensorflow as tf

def _smallest_positive_root(a, b, c) -> tf.Tensor:
  """
    Returns the smallest positive root of a quadratic equation.
    This implements the fixed version in https://github.com/princeton-vl/CornerNet.
  """

  discriminant = tf.sqrt(b ** 2 - 4 * a * c)

  root1 = (-b - discriminant) / (2 * a)
  root2 = (-b + discriminant) / (2 * a)

  return (-b + discriminant) / (2) #tf.where(tf.less(root1, 0), root2, root1)

def gaussian_radius(det_size, min_overlap=0.7) -> int:
  """
    Given a bounding box size, returns a lower bound on how far apart the
    corners of another bounding box can lie while still maintaining the given
    minimum overlap, or IoU. Modified from implementation found in
    https://github.com/tensorflow/models/blob/master/research/object_detection/core/target_assigner.py.

    Params:
        det_size (tuple): tuple of integers representing height and width
        min_overlap (tf.float32): minimum IoU desired
    Returns:
        int representing desired gaussian radius
    """
  height, width = det_size

  # Case where detected box is offset from ground truth and no box completely
  # contains the other.

  a1  = 1
  b1  = -(height + width)
  c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
  r1 = _smallest_positive_root(a1, b1, c1)

  # Case where detection is smaller than ground truth and completely contained
  # in it.

  a2  = 4
  b2  = -2 * (height + width)
  c2  = (1 - min_overlap) * width * height
  r2 = _smallest_positive_root(a2, b2, c2)

  # Case where ground truth is smaller than detection and completely contained
  # in it.

  a3  = 4 * min_overlap
  b3  = 2 * min_overlap * (height + width)
  c3  = (min_overlap - 1) * width * height
  r3 = _smallest_positive_root(a3, b3, c3)
  # TODO discuss whether to return scalar or tensor
  # return tf.reduce_min([r1, r2, r3], axis=0)

  return tf.reduce_min([r1, r2, r3], axis=0)

def gaussian_penalty(radius: int, type=tf.float32) -> tf.Tensor:
    """
    This represents the penalty reduction around a point.
    Params:
        radius (int): integer for radius of penalty reduction
        type (tf.dtypes.DType): datatype of returned tensor
    Returns:
        tf.Tensor of shape (2 * radius + 1, 2 * radius + 1).
    """
    width = 2 * radius + 1
    sigma = radius / 3
    x = tf.reshape(tf.range(width, dtype=type) - radius, (width, 1))
    y = tf.reshape(tf.range(width, dtype=type) - radius, (1, width))
    exponent = (-1 * (x ** 2) - (y ** 2)) / (2 * sigma ** 2)
    return tf.math.exp(exponent)

def draw_gaussian(heatmap, center, radius, k=1):
    """
    Draws a gaussian heatmap around a center point given a radius.
    Params:
        heatmap (tf.Tensor): heatmap placeholder to fill
        center (int): integer for center of gaussian
        radius (int): integer for radius of gaussian
        k (int): scaling factor for gaussian
    """

    diameter = 2 * radius + 1
    gaussian = gaussian_penalty(radius)

    x, y = center

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap  = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    # TODO: make sure this replicates original functionality
    # np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    masked_heatmap = tf.math.maximum(masked_heatmap, masked_gaussian * k)

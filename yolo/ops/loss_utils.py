import tensorflow as tf
from tensorflow.keras import backend as K


@tf.function(experimental_relax_shapes=True)
def _build_grid_points(lwidth, lheight, num, dtype):
  """ generate a grid that is used to detemine the relative centers of the bounding boxs """
  with tf.name_scope('center_grid'):
    y = tf.range(0, lheight)
    x = tf.range(0, lwidth)
    #x_left, y_left = tf.meshgrid(y, x)

    x_left = tf.repeat(
        tf.transpose(tf.expand_dims(y, axis=-1), perm=[1, 0]), [lwidth], axis=0)
    y_left = tf.repeat(tf.expand_dims(x, axis=-1), [lheight], axis=1)
    x_y = K.stack([x_left, y_left], axis=-1)
    x_y = tf.cast(x_y, dtype=dtype) / tf.cast(lwidth, dtype=dtype)
    x_y = tf.expand_dims(
        tf.repeat(tf.expand_dims(x_y, axis=-2), num, axis=-2), axis=0)
  return x_y


@tf.function(experimental_relax_shapes=True)
def _build_anchor_grid(width, height, anchors, num, dtype):
  with tf.name_scope('anchor_grid'):
    """ get the transformed anchor boxes for each dimention """
    anchors = tf.cast(anchors, dtype=dtype)
    anchors = tf.reshape(anchors, [1, -1])
    anchors = tf.repeat(anchors, width * height, axis=0)
    anchors = tf.reshape(anchors, [1, width, height, num, -1])
  return anchors


class GridGenerator(object):

  def __init__(self, anchors, masks=None, scale_anchors=None):
    self.dtype = tf.keras.backend.floatx()
    if masks is not None:
      self._num = len(masks)
    else:
      self._num = tf.shape(anchors)[0]

    if masks is not None:
      anchors = [anchors[mask] for mask in masks]

    self._scale_anchors = scale_anchors
    self._anchors = tf.convert_to_tensor(anchors)
    return

  @tf.function(experimental_relax_shapes=True)
  def _extend_batch(self, grid, batch_size):
    return tf.repeat(grid, batch_size, axis=0)

  @tf.function(experimental_relax_shapes=True)
  def __call__(self, width, height, batch_size, dtype=None):
    if dtype is None:
      self.dtype = tf.keras.backend.floatx()
    else:
      self.dtype = dtype
    grid_points = _build_grid_points(width, height, self._num, self.dtype)
    anchor_grid = _build_anchor_grid(
        width, height,
        tf.cast(self._anchors, self.dtype) /
        tf.cast(self._scale_anchors * width, self.dtype), self._num, self.dtype)
    grid_points = self._extend_batch(grid_points, batch_size)
    anchor_grid = self._extend_batch(anchor_grid, batch_size)
    return tf.stop_gradient(grid_points), tf.stop_gradient(anchor_grid)

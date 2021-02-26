import tensorflow as tf
import numpy as np

import centernet.utils.groundtruth as preprocessing_ops

class CenterNetBoxTargetAssignerTest(tf.test.TestCase):

  def __init__(self, *args, **kwargs):
    super(CenterNetBoxTargetAssignerTest, self).__init__(*args, **kwargs)
    self._box_center = [0.0, 0.0, 1.0, 1.0]
    self._box_center_small = [0.25, 0.25, 0.75, 0.75]
    self._box_lower_left = [0.5, 0.0, 1.0, 0.5]
    self._box_center_offset = [0.1, 0.05, 1.0, 1.0]
    self._box_odd_coordinates = [0.1625, 0.2125, 0.5625, 0.9625]

  def test_max_distance_for_overlap(self):
    """Test that the distance ensures the IoU with random boxes."""

    # TODO(vighneshb) remove this after the `_smallest_positive_root`
    # function if fixed.
    self.skipTest(('Skipping test because we are using an incorrect version of'
                   'the `max_distance_for_overlap` function to reproduce'
                   ' results.'))

    rng = np.random.RandomState(0)
    n_samples = 100

    width = rng.uniform(1, 100, size=n_samples)
    height = rng.uniform(1, 100, size=n_samples)
    min_iou = rng.uniform(0.1, 1.0, size=n_samples)

    max_dist = preprocessing_ops.gaussian_radius((height, width), min_iou)
    
    xmin1 = np.zeros(n_samples)
    ymin1 = np.zeros(n_samples)
    xmax1 = np.zeros(n_samples) + width
    ymax1 = np.zeros(n_samples) + height

    xmin2 = max_dist * np.cos(rng.uniform(0, 2 * np.pi))
    ymin2 = max_dist * np.sin(rng.uniform(0, 2 * np.pi))
    xmax2 = width + max_dist * np.cos(rng.uniform(0, 2 * np.pi))
    ymax2 = height + max_dist * np.sin(rng.uniform(0, 2 * np.pi))

    boxes1 = np.vstack([ymin1, xmin1, ymax1, xmax1]).T
    boxes2 = np.vstack([ymin2, xmin2, ymax2, xmax2]).T

    iou = np.diag(np_box_ops.iou(boxes1, boxes2))

    self.assertTrue(np.all(iou >= min_iou))

  def test_max_distance_for_overlap_centernet(self):
    """Test the version of the function used in the CenterNet paper."""
    distance = preprocessing_ops.gaussian_radius((10, 5), 0.5)
    self.assertAlmostEqual(2.807764064, distance.numpy())





  def coordinates_to_heatmap(y_grid,
                           x_grid,
                           y_coordinates,
                           x_coordinates,
                           sigma,
                           channel_onehot,
                           channel_weights=None,
                           sparse=False):
    """Returns the heatmap targets from a set of point coordinates.
    This function maps a set of point coordinates to the output heatmap image
    applied using a Gaussian kernel. Note that this function be can used by both
    object detection and keypoint estimation tasks. For object detection, the
    "channel" refers to the object class. For keypoint estimation, the "channel"
    refers to the number of keypoint types.
    Args:
      y_grid: A 2D tensor with shape [height, width] which contains the grid
        y-coordinates given in the (output) image dimensions.
      x_grid: A 2D tensor with shape [height, width] which contains the grid
        x-coordinates given in the (output) image dimensions.
      y_coordinates: A 1D tensor with shape [num_instances] representing the
        y-coordinates of the instances in the output space coordinates.
      x_coordinates: A 1D tensor with shape [num_instances] representing the
        x-coordinates of the instances in the output space coordinates.
      sigma: A 1D tensor with shape [num_instances] representing the standard
        deviation of the Gaussian kernel to be applied to the point.
      channel_onehot: A 2D tensor with shape [num_instances, num_channels]
        representing the one-hot encoded channel labels for each point.
      channel_weights: A 1D tensor with shape [num_instances] corresponding to the
        weight of each instance.
      sparse: bool, indicating whether or not to use the sparse implementation
        of the function. The sparse version scales better with number of channels,
        but in some cases is known to cause OOM error. See (b/170989061).
    Returns:
      heatmap: A tensor of size [height, width, num_channels] representing the
        heatmap. Output (height, width) match the dimensions of the input grids.
    """

    # if sparse:
    #   return _coordinates_to_heatmap_sparse(
    #       y_grid, x_grid, y_coordinates, x_coordinates, sigma, channel_onehot,
    #       channel_weights)
    # else:
    return _coordinates_to_heatmap_dense(
        y_grid, x_grid, y_coordinates, x_coordinates, sigma, channel_onehot,
        channel_weights)


  def _coordinates_to_heatmap_dense(y_grid, x_grid, y_coordinates, x_coordinates,
                                  sigma, channel_onehot, channel_weights=None):
    """Dense version of coordinates to heatmap that uses an outer product."""
    num_instances, num_channels = (
        shape_utils.combined_static_and_dynamic_shape(channel_onehot))

    x_grid = tf.expand_dims(x_grid, 2)
    y_grid = tf.expand_dims(y_grid, 2)
    # The raw center coordinates in the output space.
    x_diff = x_grid - tf.math.floor(x_coordinates)
    y_diff = y_grid - tf.math.floor(y_coordinates)
    squared_distance = x_diff**2 + y_diff**2

    gaussian_map = tf.exp(-squared_distance / (2 * sigma * sigma))

    reshaped_gaussian_map = tf.expand_dims(gaussian_map, axis=-1)
    reshaped_channel_onehot = tf.reshape(channel_onehot,
                                        (1, 1, num_instances, num_channels))
    gaussian_per_box_per_class_map = (
        reshaped_gaussian_map * reshaped_channel_onehot)

    if channel_weights is not None:
      reshaped_weights = tf.reshape(channel_weights, (1, 1, num_instances, 1))
      gaussian_per_box_per_class_map *= reshaped_weights

    # Take maximum along the "instance" dimension so that all per-instance
    # heatmaps of the same class are merged together.
    heatmap = tf.reduce_max(gaussian_per_box_per_class_map, axis=2)

    # Maximum of an empty tensor is -inf, the following is to avoid that.
    heatmap = tf.maximum(heatmap, 0)

    return tf.stop_gradient(heatmap)

if __name__ == '__main__':
  tf.test.main()

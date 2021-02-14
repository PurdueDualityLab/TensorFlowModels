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


if __name__ == '__main__':
  tf.test.main()

import numpy as np
import tensorflow as tf

import centernet.losses.penalty_reduced_logistic_focal_loss as losses

LOG_2 = np.log(2)
LOG_3 = np.log(3)

class PenaltyReducedLogisticFocalLossTest(tf.test.TestCase):
  """Testing loss function from Equation (1) in [1].
  [1]: https://arxiv.org/abs/1904.07850
  """

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._prediction = np.array([
        # First batch
        [[1 / 2, 1 / 4, 3 / 4],
         [3 / 4, 1 / 3, 1 / 3]],
        # Second Batch
        [[0.0, 1.0, 1 / 2],
         [3 / 4, 2 / 3, 1 / 3]]], np.float32)
    self._prediction = np.log(self._prediction/(1 - self._prediction))

    self._target = np.array([
        # First batch
        [[1.0, 0.91, 1.0],
         [0.36, 0.84, 1.0]],
        # Second Batch
        [[0.01, 1.0, 0.75],
         [0.96, 1.0, 1.0]]], np.float32)

  def test_returns_correct_loss(self):
    def graph_fn(prediction, target):
      weights = tf.constant([
          [[1.0], [1.0]],
          [[1.0], [1.0]],
      ])
      loss = losses.PenaltyReducedLogisticFocalLoss(
          alpha=2.0, beta=0.5, reduction=tf.keras.losses.Reduction.NONE)
      computed_value = loss(target, prediction, weights)
      return computed_value
    computed_value = graph_fn(self._prediction, self._target)
    expected_value = np.array([
        # First batch
        [[1 / 4 * LOG_2,
          0.3 * 0.0625 * (2 * LOG_2 - LOG_3),
          1 / 16 * (2 * LOG_2 - LOG_3)],
         [0.8 * 9 / 16 * 2 * LOG_2,
          0.4 * 1 / 9 * (LOG_3 - LOG_2),
          4 / 9 * LOG_3]],
        # Second Batch
        [[0.0,
          0.0,
          1 / 2 * 1 / 4 * LOG_2],
         [0.2 * 9 / 16 * 2 * LOG_2,
          1 / 9 * (LOG_3 - LOG_2),
          4 / 9 * LOG_3]]])
    self.assertAllClose(expected_value, computed_value, rtol=1e-3, atol=1e-3)

  def test_returns_correct_loss_weighted(self):
    def graph_fn(prediction, target):
      weights = tf.constant([
          [[1.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
          [[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]],
      ])

      loss = losses.PenaltyReducedLogisticFocalLoss(
          alpha=2.0, beta=0.5, reduction=tf.keras.losses.Reduction.NONE)

      computed_value = loss(target, prediction, weights)
      return computed_value
    computed_value = graph_fn(self._prediction, self._target)
    expected_value = np.array([
        # First batch
        [[1 / 4 * LOG_2,
          0.0,
          1 / 16 * (2 * LOG_2 - LOG_3)],
         [0.0,
          0.0,
          4 / 9 * LOG_3]],
        # Second Batch
        [[0.0,
          0.0,
          1 / 2 * 1 / 4 * LOG_2],
         [0.0,
          0.0,
          0.0]]])

    self.assertAllClose(expected_value, computed_value, rtol=1e-3, atol=1e-3)

if __name__ == '__main__':
  tf.test.main()

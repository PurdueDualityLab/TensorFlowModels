import tensorflow as tf
import numpy as np

from centernet import losses

class L1LocalizationLossTest(tf.test.TestCase):

  def test_returns_correct_loss(self):
    def graph_fn():
      loss = losses.L1LocalizationLoss(reduction=tf.keras.losses.Reduction.NONE)
      pred = [[0.1, 0.2], [0.7, 0.5]]
      target = [[0.9, 1.0], [0.1, 0.4]]

      weights = [[1.0, 0.0], [1.0, 1.0]]
      return loss(pred, target, weights)
    computed_value = graph_fn()
    self.assertAllClose(computed_value, [[0.8, 0.0], [0.6, 0.1]], rtol=1e-6)

if __name__ == '__main__':
  tf.test.main()

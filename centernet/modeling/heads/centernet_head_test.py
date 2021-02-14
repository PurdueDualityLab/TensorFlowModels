from absl.testing import parameterized
import tensorflow as tf
import numpy as np

from centernet.modeling.heads import centernet_head


class CenterNetHeadTest(tf.test.TestCase, parameterized.TestCase):

  def test_head(self):
    head = centernet_head.CenterNetHead(classes=91, task="2D")

    outputs = head(np.zeros((2, 128, 128, 256), dtype=np.float32))
    self.assertEqual(len(outputs), 3)
    self.assertEqual(outputs['heatmaps'].shape, (2, 128, 128, 91))
    self.assertEqual(outputs['local_offset'].shape, (2, 128, 128, 2))
    self.assertEqual(outputs['object_size'].shape, (2, 128, 128, 2))

if __name__ == '__main__':
  tf.test.main()

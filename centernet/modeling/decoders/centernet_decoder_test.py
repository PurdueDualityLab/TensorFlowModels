from absl.testing import parameterized
import tensorflow as tf
import numpy as np

from centernet.modeling.decoders import centernet_decoder


class CenterNetDecoderTest(tf.test.TestCase, parameterized.TestCase):

  def test_head(self):
    task_outputs = {'heatmap': 91,
        'local_offset': 2, 
        'object_size': 2
    }

    head = centernet_decoder.CenterNetDecoder(task_outputs=task_outputs, 
      heatmap_bias=-2.19)

    # Output shape tests
    outputs = head(np.zeros((2, 128, 128, 256), dtype=np.float32))
    self.assertEqual(len(outputs), 3)
    self.assertEqual(outputs['heatmap'].shape, (2, 128, 128, 91))
    self.assertEqual(outputs['local_offset'].shape, (2, 128, 128, 2))
    self.assertEqual(outputs['object_size'].shape, (2, 128, 128, 2))

    # Weight initialization tests
    hm_bias_vector = np.asarray(head.layers['heatmap'].weights[-1])
    off_bias_vector = np.asarray(head.layers['local_offset'].weights[-1])
    size_bias_vector = np.asarray(head.layers['object_size'].weights[-1])

    self.assertArrayNear(hm_bias_vector, 
      np.repeat(-2.19, repeats=91), err=1.00e-6)
    self.assertArrayNear(off_bias_vector, 
      np.repeat(0, repeats=2), err=1.00e-6)
    self.assertArrayNear(size_bias_vector, 
      np.repeat(0, repeats=2), err=1.00e-6)

if __name__ == '__main__':
  tf.test.main()

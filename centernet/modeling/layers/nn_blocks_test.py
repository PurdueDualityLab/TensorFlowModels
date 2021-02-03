from absl.testing import parameterized
import tensorflow as tf

from centernet.modeling.layers import nn_blocks

class NNBlocksTest(parameterized.TestCase, tf.test.TestCase):
  def test_hourglass_block(self):
    dims    = [256, 256, 384, 384, 384, 512]
    modules = [2, 2, 2, 2, 2, 4]
    model = nn_blocks.HourglassBlock(dims, modules)
    test_input = tf.keras.Input((512, 512, 256))
    _ = model(test_input)

if __name__ == '__main__':
  tf.test.main()

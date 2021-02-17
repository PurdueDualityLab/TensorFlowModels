from absl.testing import parameterized
import tensorflow as tf
import numpy as np

from centernet.configs import backbones as cfg
from centernet.modeling.backbones import hourglass

import dataclasses
from official.modeling import hyperparams
from official.vision.beta.configs import backbones


@dataclasses.dataclass
class CenterNet(hyperparams.Config):
  backbone: backbones.Backbone = backbones.Backbone(type='hourglass')


class HourglassTest(tf.test.TestCase, parameterized.TestCase):

  def test_hourglass(self):
    model = hourglass.Hourglass(
        blocks_per_stage=[2, 3, 4, 5, 6],
        input_channel_dims=4,
        channel_dims_per_stage=[6, 8, 10, 12, 14],
        num_hourglasses=2)
    outputs = model(np.zeros((2, 64, 64, 3), dtype=np.float32))
    self.assertEqual(outputs[0].shape, (2, 16, 16, 6))
    self.assertEqual(outputs[1].shape, (2, 16, 16, 6))

    backbone = hourglass.build_hourglass(
        tf.keras.layers.InputSpec(shape=[None, 512, 512, 3]), CenterNet())
    input = np.zeros((2, 512, 512, 3), dtype=np.float32)
    outputs = backbone(input)
    self.assertEqual(outputs[0].shape, (2, 128, 128, 256))
    self.assertEqual(outputs[1].shape, (2, 128, 128, 256))


if __name__ == '__main__':
  tf.test.main()

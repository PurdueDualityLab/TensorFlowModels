from absl.testing import parameterized
import tensorflow as tf
import numpy as np
import dataclasses

from official.modeling import hyperparams
from official.vision.beta.configs import backbones

from centernet.modeling.CenterNet import build_centernet
from centernet.configs import centernet


class CenterNetTest(parameterized.TestCase, tf.test.TestCase):

  def testBuildCenterNet(self):
    input_specs = tf.keras.layers.InputSpec(shape=[None, 512, 512, 3])

    config = centernet.CenterNetTask()
    model, loss = build_centernet(input_specs=input_specs,
      task_config=config, l2_regularization=0)
    
    outputs = model(tf.zeros((5, 512, 512, 3)))
    self.assertEqual(len(outputs['raw_output']), 3)
    self.assertEqual(len(outputs['raw_output']['ct_heatmaps']), 2)
    self.assertEqual(len(outputs['raw_output']['ct_offset']), 2)
    self.assertEqual(len(outputs['raw_output']['ct_size']), 2)
    self.assertEqual(outputs['raw_output']['ct_heatmaps'][0].shape, (5, 128, 128, 90))
    self.assertEqual(outputs['raw_output']['ct_offset'][0].shape, (5, 128, 128, 2))
    self.assertEqual(outputs['raw_output']['ct_size'][0].shape, (5, 128, 128, 2))

    model.summary()
    


if __name__ == '__main__':
  tf.test.main()

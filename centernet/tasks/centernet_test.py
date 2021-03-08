from absl.testing import parameterized
import tensorflow as tf
import numpy as np
import dataclasses

from official.modeling import hyperparams
from official.vision.beta.configs import backbones

from centernet.tasks.centernet import CenterNetTask
from centernet.configs import centernet as exp_cfg


class CenterNetTaskTest(parameterized.TestCase, tf.test.TestCase):

  def testCenterNetTask(self):
    model_config = exp_cfg.CenterNet(input_size=[512, 512, 3])
    config = exp_cfg.CenterNetTask(model=model_config)
    task = CenterNetTask(config)

    model = task.build_model()
    outputs = model(tf.zeros((3, 512, 512, 3)))

    self.assertEqual(len(outputs['raw_output']), 3)
    self.assertEqual(outputs['raw_output']['ct_heatmaps'][0].shape, (3, 128, 128, 90))
    self.assertEqual(outputs['raw_output']['ct_offset'][0].shape, (3, 128, 128, 2))
    self.assertEqual(outputs['raw_output']['ct_size'][0].shape, (3, 128, 128, 2))

    model.summary()    


if __name__ == '__main__':
  tf.test.main()

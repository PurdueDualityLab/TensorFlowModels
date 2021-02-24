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
    
    # TODO: add some call tests
    


if __name__ == '__main__':
  tf.test.main()

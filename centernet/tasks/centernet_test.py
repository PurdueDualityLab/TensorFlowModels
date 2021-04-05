import dataclasses

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from absl.testing import parameterized

import orbit
from centernet.configs import centernet as exp_cfg
from centernet.dataloaders.centernet_input import CenterNetParser
from centernet.tasks.centernet import CenterNetTask
from centernet.utils.weight_utils.load_weights import (
    get_model_weights_as_dict, load_weights_model)
from official.modeling import hyperparams
from official.vision.beta.configs import backbones

CENTERNET_CKPT_PATH = 'D:\\weights\centernet_hg104_512x512_coco17_tpu-8\checkpoint'

class CenterNetTaskTest(parameterized.TestCase, tf.test.TestCase):

  # def testCenterNetTask(self):
  #   model_config = exp_cfg.CenterNet(input_size=[512, 512, 3])
  #   config = exp_cfg.CenterNetTask(model=model_config)
  #   task = CenterNetTask(config)

  #   model = task.build_model()
  #   outputs = model(tf.zeros((3, 512, 512, 3)))

  #   self.assertEqual(len(outputs['raw_output']), 3)
  #   self.assertEqual(outputs['raw_output']['ct_heatmaps'][0].shape, (3, 128, 128, 90))
  #   self.assertEqual(outputs['raw_output']['ct_offset'][0].shape, (3, 128, 128, 2))
  #   self.assertEqual(outputs['raw_output']['ct_size'][0].shape, (3, 128, 128, 2))

  #   model.summary()

  def testCenterNetValidation(self):
    model_config = exp_cfg.CenterNet(input_size=[512, 512, 3])
    config = exp_cfg.CenterNetTask(model=model_config)
    
    task = CenterNetTask(config)
    model = task.build_model()
    task.initialize(model)
    metrics = task.build_metrics(training=False)
    strategy = tf.distribute.get_strategy()
    
    dataset = orbit.utils.make_distributed_dataset(strategy, task.build_inputs,
                                                   config.validation_data)

    # tf.print("Dataset: ")
    # tf.print(dataset)
    # iterator = iter(dataset)
    # tf.print("Getting logs: ")
    # logs = task.validation_step(next(iterator), model, metrics=metrics)
    # tf.print("logs: ")
    # tf.print(logs)



if __name__ == '__main__':
  tf.test.main()

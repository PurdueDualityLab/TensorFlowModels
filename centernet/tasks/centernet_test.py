from absl.testing import parameterized
import tensorflow as tf
import numpy as np
import dataclasses

import orbit

from official.modeling import hyperparams
from official.vision.beta.configs import backbones

from centernet.tasks.centernet import CenterNetTask
from centernet.configs import centernet as exp_cfg
from centernet.utils.weight_utils.load_weights import get_model_weights_as_dict, load_weights_model

import tensorflow_datasets as tfds

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
    metrics = task.build_metrics(training=False)

    weights_dict, _ = get_model_weights_as_dict(CENTERNET_CKPT_PATH)
    load_weights_model(model, weights_dict, 'hourglass104_512', 'detection_2d')

    data_dir = 'D:\Datasets'
    ds, ds_info = tfds.load(name='coco/2017',
                            split='validation', 
                            shuffle_files=True, 
                            data_dir=data_dir, 
                            with_info=True, 
                            download=True)

    strategy = tf.distribute.get_strategy()
    dataset = orbit.utils.make_distributed_dataset(strategy, ds)

    iterator = iter(dataset)
    print(next(iterator))
    # logs = task.validation_step(next(iterator), model, metrics=metrics)
    # print(logs)

if __name__ == '__main__':
  tf.test.main()

  
  





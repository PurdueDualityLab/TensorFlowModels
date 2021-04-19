import tensorflow as tf
from absl.testing import parameterized

import orbit
from centernet.tasks import centernet
from official.core import exp_factory

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

  @parameterized.parameters(("centernet_tpu",))
  def testCenterNetValidation(self, config_name):
    config = exp_factory.get_exp_config(config_name)

    task = centernet.CenterNetTask(config.task)
    model = task.build_model()
    metrics = task.build_metrics(training=False)
    strategy = tf.distribute.get_strategy()

    dataset = orbit.utils.make_distributed_dataset(strategy, task.build_inputs,
                                                   config.task.train_data)

    iterator = iter(dataset)
    opt_factory = optimization.OptimizerFactory(config.trainer.optimizer_config)
    logs = task.validation_step(next(iterator), model, metrics=metrics)
    self.assertIn("loss", logs)


if __name__ == '__main__':
  tf.test.main()

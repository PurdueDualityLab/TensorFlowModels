from official.modeling.optimization import configs
from official.common import registry_imports
from yolo.common import registry_imports

from yolo.tasks import yolo
import orbit
from official.core import exp_factory
from official.modeling import optimization

import tensorflow as tf
from absl.testing import parameterized
from official.core import train_utils


class YoloTaskTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters(("scaled_yolo_3l_coco",))
  def test_task(self, config_name):
    config = exp_factory.get_exp_config(config_name)

    config.trainer.optimizer_config.ema = None
    config.task.train_data.global_batch_size = 1
    config.task.validation_data.global_batch_size = 1

    task = yolo.YoloTask(config.task)
    model = task.build_model()
    metrics = task.build_metrics(training=False)
    strategy = tf.distribute.get_strategy()

    train = orbit.utils.make_distributed_dataset(strategy, task.build_inputs,
                                                   config.task.train_data)
    test = orbit.utils.make_distributed_dataset(strategy, task.build_inputs,
                                                   config.task.validation_data)
    train = iter(train)
    test = iter(test)
    optimizer = task.create_optimizer(config.trainer.optimizer_config)
    logs = task.train_step(next(train), model, optimizer, metrics=metrics)
    self.assertIn("loss", logs)
    logs = task.validation_step(next(test), model, metrics=metrics)
    self.assertIn("loss", logs)


if __name__ == "__main__":
  tf.test.main()

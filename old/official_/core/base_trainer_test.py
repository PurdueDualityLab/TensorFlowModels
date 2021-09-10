# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for tensorflow_models.core.trainers.trainer."""
# pylint: disable=g-direct-tensorflow-import
import os
from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import strategy_combinations
from official.core import base_trainer as trainer_lib
from official.core import config_definitions as cfg
from official.core import train_lib
from official.utils.testing import mock_task


def all_strategy_combinations():
  return combinations.combine(
      distribution=[
          strategy_combinations.default_strategy,
          strategy_combinations.cloud_tpu_strategy,
          strategy_combinations.one_device_strategy_gpu,
      ],)


class TrainerTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._config = cfg.ExperimentConfig(
        trainer=cfg.TrainerConfig(
            optimizer_config=cfg.OptimizationConfig({
                'optimizer': {
                    'type': 'sgd'
                },
                'learning_rate': {
                    'type': 'constant'
                }
            })))

  def create_test_trainer(self, config, model_dir=None, task=None):
    task = task or mock_task.MockTask(config.task, logging_dir=model_dir)
    ckpt_exporter = train_lib.maybe_create_best_ckpt_exporter(config, model_dir)
    trainer = trainer_lib.Trainer(
        config,
        task,
        model=task.build_model(),
        optimizer=task.create_optimizer(config.trainer.optimizer_config,
                                        config.runtime),
        checkpoint_exporter=ckpt_exporter)
    return trainer

  @combinations.generate(all_strategy_combinations())
  def test_trainer_train(self, distribution):
    with distribution.scope():
      trainer = self.create_test_trainer(self._config)
      logs = trainer.train(tf.convert_to_tensor(5, dtype=tf.int32))
      self.assertIn('training_loss', logs)
      self.assertIn('learning_rate', logs)

  @combinations.generate(all_strategy_combinations())
  def test_trainer_validate(self, distribution):
    with distribution.scope():
      trainer = self.create_test_trainer(self._config)
      logs = trainer.evaluate(tf.convert_to_tensor(5, dtype=tf.int32))
      self.assertEqual(logs['counter'], 5. * distribution.num_replicas_in_sync)
      self.assertIn('validation_loss', logs)

  @combinations.generate(all_strategy_combinations())
  def test_trainer_validate_without_loss(self, distribution):

    class MockTaskWithoutValidationLoss(mock_task.MockTask):

      def validation_step(self, inputs, model, metrics=None):
        # Disable validation loss.
        logs = super().validation_step(inputs, model)
        del logs[self.loss]
        return logs

    with distribution.scope():
      task = MockTaskWithoutValidationLoss()
      trainer = self.create_test_trainer(self._config, task=task)
      logs = trainer.evaluate(tf.convert_to_tensor(5, dtype=tf.int32))
      self.assertEqual(logs['counter'], 5. * distribution.num_replicas_in_sync)
      self.assertNotIn('validation_loss', logs)

  @combinations.generate(
      combinations.combine(
          mixed_precision_dtype=['float32', 'bfloat16', 'float16'],
          loss_scale=[None, 'dynamic', 128, 256],
      ))
  def test_configure_optimizer(self, mixed_precision_dtype, loss_scale):
    config = cfg.ExperimentConfig(
        runtime=cfg.RuntimeConfig(
            mixed_precision_dtype=mixed_precision_dtype, loss_scale=loss_scale),
        trainer=cfg.TrainerConfig(
            optimizer_config=cfg.OptimizationConfig({
                'optimizer': {
                    'type': 'sgd'
                },
                'learning_rate': {
                    'type': 'constant'
                },
                'use_experimental_api': {
                    'type': False
                },
            })))
    trainer = self.create_test_trainer(config)
    if mixed_precision_dtype != 'float16':
      self.assertIsInstance(trainer.optimizer, tf.keras.optimizers.SGD)
    elif mixed_precision_dtype == 'float16' and loss_scale is None:
      self.assertIsInstance(trainer.optimizer, tf.keras.optimizers.SGD)
    else:
      self.assertIsInstance(trainer.optimizer,
                            tf.keras.mixed_precision.LossScaleOptimizer)

    metrics = trainer.train(tf.convert_to_tensor(5, dtype=tf.int32))
    self.assertIn('training_loss', metrics)

  def test_export_best_ckpt(self):
    config = cfg.ExperimentConfig(
        trainer=cfg.TrainerConfig(
            best_checkpoint_export_subdir='best_ckpt',
            best_checkpoint_eval_metric='acc',
            optimizer_config=cfg.OptimizationConfig({
                'optimizer': {
                    'type': 'sgd'
                },
                'learning_rate': {
                    'type': 'constant'
                }
            })))
    model_dir = self.get_temp_dir()
    trainer = self.create_test_trainer(config, model_dir=model_dir)
    trainer.train(tf.convert_to_tensor(1, dtype=tf.int32))
    trainer.evaluate(tf.convert_to_tensor(1, dtype=tf.int32))
    self.assertTrue(
        tf.io.gfile.exists(os.path.join(model_dir, 'best_ckpt', 'info.json')))

  def test_recovery(self):
    config = cfg.ExperimentConfig(
        trainer=cfg.TrainerConfig(
            loss_upper_bound=0.5,
            recovery_max_trials=2,
            optimizer_config=cfg.OptimizationConfig({
                'optimizer': {
                    'type': 'sgd'
                },
                'learning_rate': {
                    'type': 'constant'
                }
            })))
    model_dir = self.get_temp_dir()
    trainer = self.create_test_trainer(config, model_dir=model_dir)
    checkpoint_manager = tf.train.CheckpointManager(
        trainer.checkpoint, self.get_temp_dir(), max_to_keep=2)
    checkpoint_manager.save()
    trainer.add_recovery(config.trainer, checkpoint_manager=checkpoint_manager)
    before_weights = trainer.model.get_weights()
    _ = trainer.train(tf.convert_to_tensor(1, dtype=tf.int32))
    # The training loss is 1.0 and upper_bound is 0.5, so the recover happens.
    after_weights = trainer.model.get_weights()
    for left, right in zip(before_weights, after_weights):
      self.assertAllEqual(left, right)

    # Let's the loss be NaN and max_trials = 0 to see RuntimeError.
    config = cfg.ExperimentConfig(
        trainer=cfg.TrainerConfig(
            recovery_max_trials=0,
            optimizer_config=cfg.OptimizationConfig({
                'optimizer': {
                    'type': 'sgd'
                },
                'learning_rate': {
                    'type': 'constant'
                }
            })))
    task = mock_task.MockTask(config.task, logging_dir=model_dir)

    def build_losses(labels, model_outputs, aux_losses=None):
      del labels, model_outputs
      return tf.constant([np.nan], tf.float32) + aux_losses

    task.build_losses = build_losses
    trainer = trainer_lib.Trainer(
        config,
        task,
        model=task.build_model(),
        optimizer=task.create_optimizer(config.trainer.optimizer_config,
                                        config.runtime))
    trainer.add_recovery(config.trainer, checkpoint_manager=checkpoint_manager)
    with self.assertRaises(RuntimeError):
      _ = trainer.train(tf.convert_to_tensor(2, dtype=tf.int32))


if __name__ == '__main__':
  tf.test.main()

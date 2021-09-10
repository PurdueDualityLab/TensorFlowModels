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

"""Multitask training driver library."""
# pytype: disable=attribute-error
import os
from absl import logging
import orbit
import tensorflow as tf
from official.core import base_task
from official.core import base_trainer as core_lib
from official.core import train_utils
from official.modeling.multitask import configs
from official.modeling.multitask import evaluator as evaluator_lib
from official.modeling.multitask import multitask


def run_experiment_with_multitask_eval(
    *,
    distribution_strategy: tf.distribute.Strategy,
    train_task: base_task.Task,
    eval_tasks: multitask.MultiTask,
    mode: str,
    params: configs.MultiEvalExperimentConfig,
    model_dir: str,
    run_post_eval: bool = False,
    save_summary: bool = True) -> tf.keras.Model:
  """Runs train/eval configured by the experiment params.

  Args:
    distribution_strategy: A distribution distribution_strategy.
    train_task: A base_task.Task instance.
    eval_tasks: A multitask.MultiTask with evaluation tasks.
    mode: A 'str', specifying the mode. Can be 'train', 'eval', 'train_and_eval'
      or 'continuous_eval'.
    params: MultiEvalExperimentConfig instance.
    model_dir: A 'str', a path to store model checkpoints and summaries.
    run_post_eval: Whether to run post eval once after training, metrics logs
      are returned.
    save_summary: Whether to save train and validation summary.

  Returns:
      model: `tf.keras.Model` instance.
  """

  is_training = 'train' in mode
  is_eval = 'eval' in mode
  with distribution_strategy.scope():
    optimizer = train_task.create_optimizer(params.trainer.optimizer_config,
                                            params.runtime)
    model = train_task.build_model()
    if is_training:
      trainer = core_lib.Trainer(
          config=params,
          task=train_task,
          model=model,
          optimizer=optimizer,
          train=True,
          evaluate=False)
    else:
      trainer = None
    if is_eval:
      evaluator = evaluator_lib.MultiTaskEvaluator(
          task=eval_tasks,
          model=model,
          global_step=trainer.global_step if is_training else None,
          checkpoint_exporter=train_utils.maybe_create_best_ckpt_exporter(
              params, model_dir))
    else:
      evaluator = None

  if trainer:
    checkpoint = trainer.checkpoint
    global_step = trainer.global_step
  else:
    checkpoint = evaluator.checkpoint
    global_step = evaluator.global_step

  checkpoint_manager = tf.train.CheckpointManager(
      checkpoint,
      directory=model_dir,
      max_to_keep=params.trainer.max_to_keep,
      step_counter=global_step,
      checkpoint_interval=params.trainer.checkpoint_interval,
      init_fn=trainer.initialize if trainer else None)

  controller = orbit.Controller(
      strategy=distribution_strategy,
      trainer=trainer,
      evaluator=evaluator,
      global_step=global_step,
      steps_per_loop=params.trainer.steps_per_loop,
      checkpoint_manager=checkpoint_manager,
      summary_dir=os.path.join(model_dir, 'train') if save_summary else None,
      eval_summary_dir=os.path.join(model_dir, 'validation') if
      (save_summary) else None,
      summary_interval=params.trainer.summary_interval if
      (save_summary) else None)

  logging.info('Starts to execute mode: %s', mode)
  with distribution_strategy.scope():
    if mode == 'train':
      controller.train(steps=params.trainer.train_steps)
    elif mode == 'train_and_eval':
      controller.train_and_evaluate(
          train_steps=params.trainer.train_steps,
          eval_steps=params.trainer.validation_steps,
          eval_interval=params.trainer.validation_interval)
    elif mode == 'eval':
      controller.evaluate(steps=params.trainer.validation_steps)
    elif mode == 'continuous_eval':

      def timeout_fn():
        if evaluator.global_step.numpy() >= params.trainer.train_steps:
          return True
        return False

      controller.evaluate_continuously(
          steps=params.trainer.validation_steps,
          timeout=params.trainer.continuous_eval_timeout,
          timeout_fn=timeout_fn)
    else:
      raise NotImplementedError('The mode is not implemented: %s' % mode)

    if run_post_eval:
      return model, evaluator.evaluate(
          tf.convert_to_tensor(params.trainer.validation_steps))
    else:
      return model, {}

# ==============================================================================
"""TensorFlow Model Garden Vision training driver."""
from yolo.utils.run_utils import prep_gpu
try:
  prep_gpu()
except BaseException:
  print('GPUs ready')

from absl import app
from absl import flags
import gin
import sys

from official.core import train_utils
# pylint: disable=unused-import
from yolo.common import registry_imports
# pylint: enable=unused-import
from official.common import distribute_utils
from official.common import flags as tfm_flags
from official.core import task_factory
from official.core import train_lib
from official.modeling import performance

import copy
import json
import os
from typing import Any, Mapping, Tuple

# Import libraries
from absl import logging
import orbit
import tensorflow as tf

from official.core import train_utils
from official.core import base_task
from official.core import config_definitions

FLAGS = flags.FLAGS
"""
python3 -m yolo.train --mode=train_and_eval --experiment=darknet_classification --model_dir=training_dir --config_file=yolo/configs/experiments/darknet53.yaml
"""
"""
python3 -m yolo.train --mode=train_and_eval --experiment=yolo_v4_coco --model_dir=training_dir --config_file=yolo/configs/experiments/yolov4.yaml
"""


def main(_):
  gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_params)
  print(FLAGS.experiment)
  params = train_utils.parse_configuration(FLAGS)

  model_dir = FLAGS.model_dir
  if 'train' in FLAGS.mode:
    # Pure eval modes do not output yaml files. Otherwise continuous eval job
    # may race against the train job for writing the same file.
    train_utils.serialize_config(params, model_dir)

  # Sets mixed_precision policy. Using 'mixed_float16' or 'mixed_bfloat16'
  # can have significant impact on model speeds by utilizing float16 in case of
  # GPUs, and bfloat16 in the case of TPUs. loss_scale takes effect only when
  # dtype is float16
  if params.runtime.mixed_precision_dtype:
    performance.set_mixed_precision_policy(params.runtime.mixed_precision_dtype,
                                           params.runtime.loss_scale)
  distribution_strategy = distribute_utils.get_distribution_strategy(
      distribution_strategy=params.runtime.distribution_strategy,
      all_reduce_alg=params.runtime.all_reduce_alg,
      num_gpus=params.runtime.num_gpus,
      tpu_address=params.runtime.tpu)
  with distribution_strategy.scope():
    task = task_factory.get_task(params.task, logging_dir=model_dir)

  print(task)
  model = get_model(
      distribution_strategy=distribution_strategy,
      task=task,
      mode=FLAGS.mode,
      params=params,
      model_dir=model_dir)

  print(model)


if __name__ == '__main__':
  import datetime

  a = datetime.datetime.now()
  tfm_flags.define_flags()
  app.run(main)
  b = datetime.datetime.now()

  print('\n\n\n\n\n\n\n {b - a}')

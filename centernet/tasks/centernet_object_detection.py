import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as mixed_precision

from absl import logging
from official.core import base_task
from official.core import input_reader
from official.core import task_factory

from official.vision.beta.evaluation import coco_evaluator


# @task_factory.register_task_cls(exp_cfg.YoloTask)
class CenterNetObjectDetectionTask(base_task.Task):

  def __init__(self, params, logging_dir: str = None):
    super().__init__(params, logging_dir)

  def build_losses(self, outputs, labels, aux_losses=None):
    loss = 0.0
    metric_dict = dict()

    # TODO: Calculate loss

    return loss, metric_dict

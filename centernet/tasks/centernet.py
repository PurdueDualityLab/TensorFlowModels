import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as mixed_precision

from absl import logging
from official.core import base_task
from official.core import input_reader
from official.core import task_factory
from centernet.configs import centernet as exp_cfg

from official.vision.beta.evaluation import coco_evaluator


@task_factory.register_task_cls(exp_cfg.CenterNetTask)
class CenterNetTask(base_task.Task):
  """A single-replica view of training procedure.
  RetinaNet task provides artifacts for training/evalution procedures, including
  loading/iterating over Datasets, initializing the model, calculating the loss,
  post-processing, and customized metrics with reduction.
  """

  def __init__(self, params, logging_dir: str = None):
    super().__init__(params, logging_dir)
    self._loss_dict = None

    self.coco_metric = None
    self._metric_names = []
    self._metrics = []
    return

  def build_model(self):
    """get an instance of CenterNet"""
    from centernet.modeling.CenterNet import build_centernet
    params = self.task_config.train_data
    model_base_cfg = self.task_config.model
    l2_weight_decay = self.task_config.weight_decay / 2.0

    input_specs = tf.keras.layers.InputSpec(shape=[None] +
                                            model_base_cfg.input_size)
    l2_regularizer = (
      tf.keras.regularizers.l2(l2_weight_decay) if l2_weight_decay else None)

    model, losses = build_centernet(input_specs, self.task_config, l2_regularizer)
    self._loss_dict = losses
    return model

  # Everything below was from YOLO
  def build_inputs(self, params, input_context=None):
    pass

  def build_losses(self, outputs, labels, aux_losses=None):
    pass

  def build_metrics(self, training=True):
    pass

  def train_step(self, inputs, model, optimizer, metrics=None):
    pass

  def aggregate_logs(self, state=None, step_outputs=None):
    pass

  def reduce_aggregated_logs(self, aggregated_logs):
    pass

  def _get_masks(self,
                 xy_exponential=True,
                 exp_base=2,
                 xy_scale_base='default_value'):
    pass

  def initialize(self, model: tf.keras.Model):
    pass
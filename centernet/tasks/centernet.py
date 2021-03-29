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

  def build_inputs(self, params, input_context=None):
    pass

  def build_losses(self, outputs, labels, aux_losses=None):
    total_loss = 0.0
    total_scale_loss = 0.0
    total_offset_loss = 0.0
    loss = 0.0
    scale_loss = 0.0
    offset_loss = 0.0

    metric_dict = dict()

    # TODO: Calculate loss
    flattened_ct_heatmaps = utils._flatten_spatial_dimensions(labels['ct_heatmaps'])
    num_boxes = utils._to_float32(utils.get_num_instances_from_weights(labels['tag_masks']))   #gt_weights_list here shouldn't be tag_masks here

    object_center_loss = penalty_reduced_logistic_focal_loss.PenaltyReducedLogisticFocalLoss()
    # Loop through each feature output head.
    for pred in outputs['ct_heatmaps']:
      pred = utils._flatten_spatial_dimensions(pred)
      total_loss += object_center_loss(
          flattened_ct_heatmaps, pred)  #removed weight parameter (weight = per_pixel_weight)
    center_loss = tf.reduce_sum(total_loss) / (
        float(len(outputs['ct_heatmaps'])) * num_boxes)
    loss += center_loss
    metric_dict['ct_loss'] = center_loss

    #localization loss for offset and scale loss
    localization_loss_fn = l1_localization_loss.L1LocalizationLoss()
    for scale_pred, offset_pred in zip(outputs['ct_size'], outputs['ct_offset']):
      # Compute the scale loss.
      scale_pred = utils.get_batch_predictions_from_indices(
          scale_pred, labels['tag_locs'])
      total_scale_loss += localization_loss_fn(
          labels['ct_size'], scale_pred)                #removed  weights=batch_weights
      # Compute the offset loss.
      offset_pred = utils.get_batch_predictions_from_indices(
          offset_pred, labels['tag_locs'])
      total_offset_loss += localization_loss_fn(
          labels['ct_offset'], offset_pred)             #removed weights=batch_weights
    scale_loss += tf.reduce_sum(total_scale_loss) / (
        float(len(outputs['ct_size'])) * num_boxes)
    offset_loss += tf.reduce_sum(total_offset_loss) / (
        float(len(outputs['ct_size'])) * num_boxes)
    metric_dict['ct_scale_loss'] = scale_loss
    metric_dict['ct_offset_loss'] = offset_loss

    return loss, metric_dict

  def build_metrics(self, training=True):
    metrics = []
    metric_names = self._metric_names

    for name in metric_names:
      metrics.append(tf.keras.metrics.Mean(name, dtype=tf.float32))

    self._metrics = metrics
    if not training:
      self.coco_metric = coco_evaluator.COCOEvaluator(
          annotation_file=self.task_config.annotation_file,
          include_mask=False,
          need_rescale_bboxes=False,
          per_category_metrics=self._task_config.per_category_metrics)
    return metrics

  def train_step(self, inputs, model, optimizer, metrics=None):
    pass
  
  def validation_step(self, inputs, model, metrics=None):
    # get the data point
    image, label = inputs

    # computer detivative and apply gradients
    y_pred = model(image, training=False)
    y_pred = tf.nest.map_structure(lambda x: tf.cast(x, tf.float32), y_pred['raw_output'])
    loss_metrics = self.build_losses(y_pred, label)
    logs = {self.loss: loss_metrics['total_loss']}

    coco_model_outputs = {
      'detection_boxes':
          box_ops.denormalize_boxes(
              tf.cast(y_pred['bbox'], tf.float32), image_shape),
      'detection_scores':
          y_pred['confidence'],
      'detection_classes':
          y_pred['classes'],
      'num_detections':
          tf.shape(y_pred['bbox'])[:-1],
    }

    logs.update({self.coco_metric.name: (label, coco_model_outputs)})

    if metrics:
      for m in metrics:
        m.update_state(loss_metrics[m.name])
        logs.update({m.name: m.result()})

    return logs

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
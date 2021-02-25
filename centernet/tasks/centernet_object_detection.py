import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as mixed_precision

from absl import logging
from official.core import base_task
from official.core import input_reader
from official.core import task_factory

from official.vision.beta.evaluation import coco_evaluator
from centernet.configs import centernet as cfg
import centernet.ops.loss_utils as utils
from centernet.losses import penalty_reduced_logistic_focal_loss
from centernet.losses import l1_localization_loss

@task_factory.register_task_cls(cfg.CenterNetTask)
class CenterNetTask(base_task.Task):

  def __init__(self, params, logging_dir: str = None):
    super().__init__(params, logging_dir)

  def build_inputs(self, params, input_context=None):
    pass

  def build_model(self):
    task_cfg = self.task_config

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
    pass

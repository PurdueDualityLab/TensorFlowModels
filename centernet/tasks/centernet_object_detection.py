import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as mixed_precision

from absl import logging
from official.core import base_task
from official.core import input_reader
from official.core import task_factory

from official.vision.beta.evaluation import coco_evaluator
from centernet.configs import centernet as cfg
import centernet.ops.loss_ops as utils
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
    num_boxes = utils._to_float32(utils.get_num_instances_from_weights(labels['tag_masks']))  

    object_center_loss = penalty_reduced_logistic_focal_loss.PenaltyReducedLogisticFocalLoss(reduction=tf.keras.losses.Reduction.NONE)
    
    outputs['ct_heatmaps'] = utils._flatten_spatial_dimensions(outputs['ct_heatmaps'])
    total_loss += object_center_loss(
        flattened_ct_heatmaps, outputs['ct_heatmaps'])  #removed weight parameter (weight = per_pixel_weight)
    center_loss = tf.reduce_sum(total_loss) / (
        float(len(outputs['ct_heatmaps'])) * num_boxes)
    loss += center_loss
    metric_dict['ct_loss'] = center_loss

    localization_loss_fn = l1_localization_loss.L1LocalizationLoss(reduction=tf.keras.losses.Reduction.NONE)
    # Compute the scale loss.
    scale_pred = outputs['ct_size']
    offset_pred = outputs['ct_offset']
    total_scale_loss += localization_loss_fn(
        labels['ct_size'], scale_pred)                #removed  weights=batch_weights
    # Compute the offset loss.
    total_offset_loss += localization_loss_fn(
        labels['ct_offset'], offset_pred)             #removed weights=batch_weights
    scale_loss += tf.reduce_sum(total_scale_loss) / (
        float(len(outputs['ct_size'])) * num_boxes)
    offset_loss += tf.reduce_sum(total_offset_loss) / (
        float(len(outputs['ct_size'])) * num_boxes)

    metric_dict['ct_scale_loss'] = scale_loss
    metric_dict['ct_offset_loss'] = offset_loss

    print(metric_dict)
    return loss, metric_dict

  def build_metrics(self, training=True):
    pass

import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as mixed_precision

from absl import logging
from official.core import base_task
from official.core import input_reader
from official.core import task_factory

from official.vision.beta.evaluation import coco_evaluator
import loss_utils as utils

# @task_factory.register_task_cls(exp_cfg.YoloTask)
class CenterNetObjectDetectionTask(base_task.Task):

  def __init__(self, params, logging_dir: str = None):
    super().__init__(params, logging_dir)

  def build_losses(self, outputs, labels, aux_losses=None):
    loss = 0.0
    metric_dict = dict()

    # TODO: Calculate loss
    flattened_ct_heatmaps = utils._flatten_spatial_dimensions(labels['ct_heatmaps'])
    num_boxes = utils._to_float32(utils.get_num_instances_from_weights(labels['tag_masks']))

    object_center_loss = self._center_params.classification_loss
    # Loop through each feature output head.
    for pred in outputs['ct_heatmaps']:
      pred = utils._flatten_spatial_dimensions(pred)
      total_loss = object_center_loss(
          pred, flattened_ct_heatmaps)  #removed weight parameter (weight = per_pixel_weight)
      loss += tf.reduce_sum(total_loss) / (
        float(len(total_loss)) * num_boxes)

    return loss, metric_dict

 
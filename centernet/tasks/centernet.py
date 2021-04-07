import tensorflow as tf
from absl import logging
from tensorflow.keras.mixed_precision import experimental as mixed_precision

from centernet.configs import centernet as exp_cfg
from centernet.dataloaders import centernet_input
from centernet.losses import (l1_localization_loss,
                              penalty_reduced_logistic_focal_loss)
from centernet.ops import loss_ops
from official.core import base_task, input_reader, task_factory
from official.vision.beta.dataloaders import (tf_example_decoder,
                                              tf_example_label_map_decoder,
                                              tfds_detection_decoders)
from official.vision.beta.evaluation import coco_evaluator


@task_factory.register_task_cls(exp_cfg.CenterNetTask)
class CenterNetTask(base_task.Task):

  def __init__(self, params, logging_dir: str = None):
    super().__init__(params, logging_dir)
    self._loss_dict = None

    self.coco_metric = None
    self._metric_names = []
    self._metrics = []

  def build_inputs(self, params, input_context=None):
    """Build input dataset."""
    decoder = self.get_decoder(params)
    model = self.task_config.model

    parser = centernet_input.CenterNetParser(
        num_classes=model.num_classes,
        max_num_instances=model.max_num_instances,
        gaussian_iou=model.gaussian_iou,
    )

    reader = input_reader.InputReader(
        params,
        dataset_fn=tf.data.TFRecordDataset,
        decoder_fn=decoder.decode,
        parser_fn=parser.parse_fn(params.is_training),
        postprocess_fn=parser.postprocess_fn(params.is_training))
    dataset = reader.read(input_context=input_context)
    return dataset

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

  def initialize(self, model: tf.keras.Model):
    """Initializes CenterNet model by loading pretrained weights """
    if self.task_config.load_odapi_weights and self.task_config.load_extremenet_weights:
      raise ValueError('Only 1 of odapi or extremenet weights should be loaded')
    
    if self.task_config.load_odapi_weights or self.task_config.load_extremenet_weights:
      from centernet.utils.weight_utils.tf_to_dict import get_model_weights_as_dict
      from centernet.utils.weight_utils.load_weights import load_weights_model
      from centernet.utils.weight_utils.load_weights import load_weights_backbone

      # Load entire model weights from the ODAPI checkpoint
      if self.task_config.load_odapi_weights:
        weights_file = self.task_config.model.base.odapi_weights
        weights_dict, n_weights = get_model_weights_as_dict(weights_file)
        load_weights_model(model, weights_dict, 
          backbone_name=self.task_config.model.base.backbone_name,
          decoder_name=self.task_config.model.base.decoder_name)

      # Load backbone weights from ExtremeNet
      else:
        weights_file = self.task_config.model.base.extremenet_weights
        weights_dict, n_weights = get_model_weights_as_dict(weights_file)
        load_weights_backbone(model.backbone, weights_dict['feature_extractor'], 
          backbone_name=self.task_config.model.base.backbone_name)

  def get_decoder(self, params):
    if params.tfds_name:
      if params.tfds_name in tfds_detection_decoders.TFDS_ID_TO_DECODER_MAP:
        decoder = tfds_detection_decoders.TFDS_ID_TO_DECODER_MAP[
            params.tfds_name]()
      else:
        raise ValueError('TFDS {} is not supported'.format(params.tfds_name))
    else:
      decoder_cfg = params.decoder.get()
      if params.decoder.type == 'simple_decoder':
        decoder = tf_example_decoder.TfExampleDecoder(
            regenerate_source_id=decoder_cfg.regenerate_source_id)
      elif params.decoder.type == 'label_map_decoder':
        decoder = tf_example_label_map_decoder.TfExampleDecoderLabelMap(
            label_map=decoder_cfg.label_map,
            regenerate_source_id=decoder_cfg.regenerate_source_id)
      else:
        raise ValueError('Unknown decoder type: {}!'.format(
            params.decoder.type)) 
    return decoder


  def build_losses(self, outputs, labels, aux_losses=None):
    total_loss = 0.0
    total_scale_loss = 0.0
    total_offset_loss = 0.0
    loss = 0.0
    scale_loss = 0.0
    offset_loss = 0.0

    metric_dict = dict()

    # TODO: Calculate loss
    flattened_ct_heatmaps = loss_ops._flatten_spatial_dimensions(labels['ct_heatmaps'])
    num_boxes = loss_ops._to_float32(loss_ops.get_num_instances_from_weights(labels['tag_masks']))

    object_center_loss = penalty_reduced_logistic_focal_loss.PenaltyReducedLogisticFocalLoss(reduction=tf.keras.losses.Reduction.NONE)

    output_ct_heatmaps = loss_ops._flatten_spatial_dimensions(outputs['ct_heatmaps'][-1])
    total_loss += object_center_loss(
        flattened_ct_heatmaps, output_ct_heatmaps)  #removed weight parameter (weight = per_pixel_weight)
    center_loss = tf.reduce_sum(total_loss) / (
        float(len(output_ct_heatmaps)) * num_boxes)
    loss += center_loss
    metric_dict['ct_loss'] = center_loss

    localization_loss_fn = l1_localization_loss.L1LocalizationLoss(reduction=tf.keras.losses.Reduction.NONE)
    # Compute the scale loss.
    scale_pred = outputs['ct_size'][-1]
    offset_pred = outputs['ct_offset'][-1]
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

    return loss, metric_dict

  def build_metrics(self, training=True):
    metrics = []

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
    y_pred = tf.nest.map_structure(lambda x: tf.cast(x, tf.float32), y_pred)
    loss_metrics = self.build_losses(y_pred['raw_output'], label)
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

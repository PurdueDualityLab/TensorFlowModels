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
from official.vision.beta.ops import box_ops


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
      image_w=params.parser.image_w,
      image_h=params.parser.image_h,
      num_classes=model.num_classes,
      max_num_instances=params.parser.max_num_instances,
      gaussian_iou=params.parser.gaussian_iou,
      output_dims=params.parser.output_dims,
      dtype=params.parser.dtype
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


  def build_losses(self, outputs, labels, num_replicas=1, scale_replicas=1, aux_losses=None):
    total_loss = 0.0
    loss = 0.0

    metric_dict = dict()

    # Create loss functions
    object_center_loss_fn = penalty_reduced_logistic_focal_loss.PenaltyReducedLogisticFocalLoss(reduction=tf.keras.losses.Reduction.NONE)
    localization_loss_fn = l1_localization_loss.L1LocalizationLoss(reduction=tf.keras.losses.Reduction.NONE)

    # Set up box indices so that they have a batch element as well
    box_indices = loss_ops.add_batch_to_indices(labels['box_indices'])

    box_mask = tf.cast(labels['box_mask'], dtype=tf.float32)
    num_boxes = loss_ops._to_float32(loss_ops.get_num_instances_from_weights(labels['box_mask']))

    # Calculate center heatmap loss
    pred_ct_heatmap_list = outputs['ct_heatmaps']
    true_flattened_ct_heatmap = loss_ops._flatten_spatial_dimensions(
      labels['ct_heatmaps'])
    
    true_flattened_ct_heatmap = tf.cast(true_flattened_ct_heatmap, tf.float32)
    total_center_loss = 0.0
    for ct_heatmap in pred_ct_heatmap_list:
      pred_flattened_ct_heatmap = loss_ops._flatten_spatial_dimensions(
        ct_heatmap)
      pred_flattened_ct_heatmap = tf.cast(pred_flattened_ct_heatmap, tf.float32)
      total_center_loss += object_center_loss_fn(
        pred_flattened_ct_heatmap, true_flattened_ct_heatmap)

    center_loss = tf.reduce_sum(total_center_loss) / float(len(pred_ct_heatmap_list) * num_boxes)
    metric_dict['ct_loss'] = center_loss

    # Calculate scale loss
    pred_scale_list = outputs['ct_size']
    true_scale = labels['size']
    true_scale = tf.cast(true_scale, tf.float32)

    total_scale_loss = 0.0
    for scale_map in pred_scale_list:
      pred_scale = loss_ops.get_batch_predictions_from_indices(scale_map, box_indices)
      pred_scale = tf.cast(pred_scale, tf.float32)
      # Only apply loss for boxes that appear in the ground truth
      total_scale_loss += tf.reduce_sum(localization_loss_fn(pred_scale, true_scale), axis=-1) * box_mask
    
    scale_loss = tf.reduce_sum(total_scale_loss) / float(len(pred_scale_list) * num_boxes)
    metric_dict['scale_loss'] = scale_loss

    # Calculate offset loss
    pred_offset_list = outputs['ct_offset']
    true_offset = labels['ct_offset']
    true_offset = tf.cast(true_offset, tf.float32)

    total_offset_loss = 0.0
    for offset_map in pred_offset_list:
      pred_offset = loss_ops.get_batch_predictions_from_indices(offset_map, box_indices)
      pred_offset = tf.cast(pred_offset, tf.float32)
      # Only apply loss for boxes that appear in the ground truth
      total_offset_loss += tf.reduce_sum(localization_loss_fn(pred_offset, true_offset), axis=-1) * box_mask
    
    offset_loss = tf.reduce_sum(total_offset_loss) / float(len(pred_offset_list) * num_boxes)
    metric_dict['ct_offset_loss'] = offset_loss
    
    # Aggregate and finalize loss
    loss_weights = self.task_config.losses.detection
    total_loss = (center_loss + 
                 loss_weights.scale_weight * scale_loss +
                 loss_weights.offset_weight * offset_loss)

    metric_dict['total_loss'] = total_loss
    return total_loss, metric_dict

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
    # get the data point
    image, label = inputs

    with tf.GradientTape() as tape:
      # compute a prediction
      # cast to float32
      y_pred = model(image, training=True)
      y_pred = tf.nest.map_structure(lambda x: tf.cast(x, tf.float32), y_pred['raw_output'])
      loss_metrics = self.build_losses(y_pred, label)

    #scale the loss for numerical stability
    if isinstance(optimizer, mixed_precision.LossScaleOptimizer):
      total_loss = optimizer.get_scaled_loss(loss_metrics['total_loss'])

    #compute the gradient
    train_vars = model.trainable_variables
    gradients = tape.gradient(total_loss, train_vars)

    #get unscaled loss if the scaled loss was used
    if isinstance(optimizer, mixed_precision.LossScaleOptimizer):
      gradients = optimizer.get_unscaled_gradients(gradients)

    if self.task_config.gradient_clip_norm > 0.0:
      gradients, _ = tf.clip_by_global_norm(gradients,
                                            self.task_config.gradient_clip_norm)

    optimizer.apply_gradients(zip(gradients, train_vars))

    logs = {self.loss: loss_metrics['total_loss']}
    if metrics:
      for m in metrics:
        m.update_state(loss_metrics[m.name])
        logs.update({m.name: m.result()})

    tf.print(logs, end='\n')
    ret = '\033[F' * (len(logs.keys()) + 1)
    tf.print(ret, end='\n')
    return logs
  
  def validation_step(self, inputs, model, metrics=None):
    # get the data point
    print("(centernet_task) - validation step")
    image, label = inputs

    scale_replicas = tf.distribute.get_strategy().num_replicas_in_sync
    if self._task_config.model.filter.use_reduction_sum:
      num_replicas = 1
    else:
      num_replicas = scale_replicas
    
    y_pred = model(image, training=False)
    print(y_pred)
    y_pred = tf.nest.map_structure(lambda x: tf.cast(x, tf.float32), y_pred)
    loss, loss_metrics = self.build_losses(
        y_pred['raw_output'],
        label,
        num_replicas=num_replicas,
        scale_replicas=1)
    logs = {self.loss: loss_metrics['total_loss']}

    image_shape = tf.shape(image)[1:-1]

    label['boxes'] = box_ops.denormalize_boxes(
        tf.cast(label['bbox'], tf.float32), image_shape)
    del label['bbox']

    coco_model_outputs = {
        'detection_boxes':
            box_ops.denormalize_boxes(
                tf.cast(y_pred['bbox'], tf.float32), image_shape),
        'detection_scores':
            y_pred['confidence'],
        'detection_classes':
            y_pred['classes'],
        'num_detections':
            y_pred['num_dets'],
        'source_id':
            label['source_id'],
    }

    logs.update({self.coco_metric.name: (label, coco_model_outputs)})

    if metrics:
      for m in metrics:
        m.update_state(loss_metrics[m.name])
        logs.update({m.name: m.result()})
    print("(centernet_task) - validation_step, logs: ", logs)
    return logs

  def aggregate_logs(self, state=None, step_outputs=None):
    print("(centernet_task) - aggregate_logs")
    if not state:
      self.coco_metric.reset_states()
      state = self.coco_metric
    self.coco_metric.update_state(step_outputs[self.coco_metric.name][0],
                                  step_outputs[self.coco_metric.name][1])
    return state

  def reduce_aggregated_logs(self, aggregated_logs):
    return self.coco_metric.result()

import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as mixed_precision

from absl import logging
from official.core import base_task
from official.core import input_reader
from official.core import task_factory
from official.vision.beta.evaluation import coco_evaluator

# TODO: Already added to official codebase, change to official version later
from yolo.dataloaders.decoders import tfds_coco_decoder

from centernet.configs import centernet as cfg
from centernet.dataloaders import centernet_input
from centernet.modeling.CenterNet import build_centernet
from centernet.ops import loss_ops
from centernet.losses import penalty_reduced_logistic_focal_loss
from centernet.losses import l1_localization_loss


@task_factory.register_task_cls(cfg.CenterNetTask)
class CenterNetTask(base_task.Task):

  def __init__(self, params, logging_dir: str = None):
    super().__init__(params, logging_dir)

  def build_inputs(self, params, input_context=None):
    """Build input dataset."""
    decoder = tfds_coco_decoder.MSCOCODecoder()
    """
    decoder_cfg = params.decoder.get()
    if params.decoder.type == 'simple_decoder':
        decoder = tf_example_decoder.TfExampleDecoder(
            regenerate_source_id=decoder_cfg.regenerate_source_id)
    elif params.decoder.type == 'label_map_decoder':
        decoder = tf_example_label_map_decoder.TfExampleDecoderLabelMap(
            label_map=decoder_cfg.label_map,
            regenerate_source_id=decoder_cfg.regenerate_source_id)
    else:
        raise ValueError('Unknown decoder type: {}!'.format(params.decoder.type))
    """

    model = self.task_config.model

    masks, path_scales, xy_scales = self._get_masks()
    anchors = self._get_boxes(gen_boxes=params.is_training)

    print(masks, path_scales, xy_scales)
    parser = centernet_input.CenterNetParser(
        num_classes=model.num_classes,
        gaussian_iou=model.gaussian_iou
    )

    if params.is_training:
      post_process_fn = parser.postprocess_fn()
    else:
      post_process_fn = None

    reader = input_reader.InputReader(
        params,
        dataset_fn=tf.data.TFRecordDataset,
        decoder_fn=decoder.decode,
        parser_fn=parser.parse_fn(params.is_training),
        postprocess_fn=post_process_fn)
    dataset = reader.read(input_context=input_context)
    return dataset

  def build_model(self):
    """get an instance of CenterNet"""
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

    outputs['ct_heatmaps'] = loss_ops._flatten_spatial_dimensions(outputs['ct_heatmaps'])
    total_loss += object_center_loss(
        flattened_ct_heatmaps, outputs['ct_heatmaps'])  #removed weight parameter (weight = per_pixel_weight)
    center_loss = tf.reduce_sum(total_loss) / (
        float(len(outputs['ct_heatmaps'])) * num_boxes)
    loss += center_loss

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

    metric_dict['ct_loss'] = center_loss
    metric_dict['ct_scale_loss'] = scale_loss
    metric_dict['ct_offset_loss'] = offset_loss

    for key in metric_dict.keys():
      total_loss += metric_dict[key]

    metric_dict['total_loss'] = total_loss
    return metric_dict

  def build_metrics(self, training=True):
    pass

  def train_step(self, inputs, model, optimizer, metrics=None):
    # get the data point
    image, label = inputs

    with tf.GradientTape() as tape:
      # compute a prediction
      # cast to float32
      y_pred = model(image, training=True)
      y_pred = tf.nest.map_structure(lambda x: tf.cast(x, tf.float32), y_pred)
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


    





  
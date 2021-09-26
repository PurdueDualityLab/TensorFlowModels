import tensorflow as tf
from typing import Optional
from collections import defaultdict

from absl import logging
from official.core import base_task
from official.core import input_reader
from official.core import task_factory
from official.core import config_definitions
from official.modeling import performance
from official.vision.beta.ops import box_ops
from official.vision.beta.evaluation import coco_evaluator
from yolo.dataloaders import tf_example_decoder
from official.vision.beta.dataloaders import tfds_detection_decoders
from official.vision.beta.dataloaders import tf_example_label_map_decoder

from yolo import optimization
from yolo.ops import mosaic
from yolo.ops import preprocessing_ops
from yolo.dataloaders import yolo_input
from yolo.configs import yolo as exp_cfg

from tensorflow.keras import mixed_precision

OptimizationConfig = optimization.OptimizationConfig
RuntimeConfig = config_definitions.RuntimeConfig


@task_factory.register_task_cls(exp_cfg.YoloTask)
class YoloTask(base_task.Task):
  """A single-replica view of training procedure.
  YOLO task provides artifacts for training/evalution procedures, including
  loading/iterating over Datasets, initializing the model, calculating the loss,
  post-processing, and customized metrics with reduction.
  """

  def __init__(self, params, logging_dir: str = None):
    super().__init__(params, logging_dir)
    self.coco_metric = None
    self._loss_fn = None
    self._model = None
    self._metrics = []

    preprocessing_ops.set_random_seeds(seed=params.train_data.seed)
    return

  def build_model(self):
    """Build an instance of Yolo"""
    from yolo.modeling.factory import build_yolo

    model_base_cfg = self.task_config.model
    l2_weight_decay = self.task_config.weight_decay / 2.0

    input_size = model_base_cfg.input_size.copy()
    input_specs = tf.keras.layers.InputSpec(shape=[None] + input_size)
    l2_regularizer = (
        tf.keras.regularizers.l2(l2_weight_decay) if l2_weight_decay else None)
    model, losses = build_yolo(input_specs, model_base_cfg, 
                               l2_regularizer)

    self._loss_fn = losses
    self._model = model
    return model

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
            coco91_to_80=decoder_cfg.coco91_to_80,
            regenerate_source_id=decoder_cfg.regenerate_source_id)
      elif params.decoder.type == 'label_map_decoder':
        decoder = tf_example_label_map_decoder.TfExampleDecoderLabelMap(
            label_map=decoder_cfg.label_map,
            regenerate_source_id=decoder_cfg.regenerate_source_id)
      else:
        raise ValueError('Unknown decoder type: {}!'.format(
            params.decoder.type))
    return decoder

  def build_inputs(self, params, input_context=None):
    """Build input dataset."""
    model = self.task_config.model

    backbone = model.backbone.get()
    anchor_dict, level_limits = model.anchor_boxes.get(backbone.min_level,
                                                       backbone.max_level)

    base_config = dict(
        letter_box=params.parser.letter_box,
        aug_rand_translate=params.parser.aug_rand_translate,
        aug_rand_angle=params.parser.aug_rand_angle,
        aug_rand_perspective=params.parser.aug_rand_perspective,
        area_thresh=params.parser.area_thresh,
        random_flip=params.parser.random_flip,
        seed=params.seed,
    )

    decoder = self.get_decoder(params)
    sample_fn = mosaic.Mosaic(
        output_size=model.input_size,
        mosaic_frequency=params.parser.mosaic.mosaic_frequency,
        mixup_frequency=params.parser.mosaic.mixup_frequency,
        jitter=params.parser.mosaic.jitter,
        mosaic_center=params.parser.mosaic.mosaic_center,
        mosaic_crop_mode=params.parser.mosaic.mosaic_crop_mode,
        aug_scale_min=params.parser.mosaic.aug_scale_min,
        aug_scale_max=params.parser.mosaic.aug_scale_max,
        **base_config)

    parser = yolo_input.Parser(
        output_size=model.input_size,
        anchors=anchor_dict,
        use_tie_breaker=params.parser.use_tie_breaker,
        jitter=params.parser.jitter,
        aug_scale_min=params.parser.aug_scale_min,
        aug_scale_max=params.parser.aug_scale_max,
        aug_rand_hue=params.parser.aug_rand_hue,
        aug_rand_saturation=params.parser.aug_rand_saturation,
        aug_rand_brightness=params.parser.aug_rand_brightness,
        max_num_instances=params.parser.max_num_instances,
        scale_xy=model.detection_generator.scale_xy.get(),
        expanded_strides=model.detection_generator.path_scales.get(),
        darknet=model.darknet_based_model,
        best_match_only=params.parser.best_match_only,
        anchor_t=params.parser.anchor_thresh,
        random_pad=params.parser.random_pad,
        level_limits=level_limits,
        dtype=params.dtype,
        **base_config)

    reader = input_reader.InputReader(
        params,
        dataset_fn=tf.data.TFRecordDataset,
        decoder_fn=decoder.decode,
        sample_fn=sample_fn.mosaic_fn(is_training=params.is_training),
        parser_fn=parser.parse_fn(params.is_training))
    dataset = reader.read(input_context=input_context)
    return dataset

  def build_metrics(self, training=True):
    metrics = []

    backbone = self.task_config.model.backbone.get()
    metric_names = defaultdict(list)
    for key in range(backbone.min_level, backbone.max_level + 1):
      key = str(key)
      metric_names[key].append('loss')
      metric_names[key].append("avg_iou")
      metric_names[key].append("avg_obj")

    metric_names['net'].append('box')
    metric_names['net'].append('class')
    metric_names['net'].append('conf')

    for i, key in enumerate(metric_names.keys()):
      metrics.append(ListMetrics(metric_names[key], name=key))

    self._metrics = metrics
    if not training:
      self.coco_metric = coco_evaluator.COCOEvaluator(
          annotation_file=self.task_config.annotation_file,
          include_mask=False,
          need_rescale_bboxes=False,
          per_category_metrics=self._task_config.per_category_metrics)

    return metrics

  def build_losses(self, outputs, labels, aux_losses=None):
    return self._loss_fn(labels, outputs)

  ## training ##
  def train_step(self, inputs, model, optimizer, metrics=None):
    image, label = inputs

    with tf.GradientTape(persistent=False) as tape:
      # Compute a prediction
      y_pred = model(image, training=True)

      # Cast to float32 for gradietn computation
      y_pred = tf.nest.map_structure(lambda x: tf.cast(x, tf.float32), y_pred)

      # Get the total loss
      (scaled_loss, metric_loss,
       loss_metrics) = self.build_losses(y_pred['raw_output'], label)

      # Scale the loss for numerical stability
      if isinstance(optimizer, mixed_precision.LossScaleOptimizer):
        scaled_loss = optimizer.get_scaled_loss(scaled_loss)

    # Compute the gradient
    train_vars = model.trainable_variables
    gradients = tape.gradient(scaled_loss, train_vars)

    # Get unscaled loss if we are using the loss scale optimizer on fp16
    if isinstance(optimizer, mixed_precision.LossScaleOptimizer):
      gradients = optimizer.get_unscaled_gradients(gradients)

    # Clip the gradients
    if self.task_config.gradient_clip_norm > 0.0:
      gradients, _ = tf.clip_by_global_norm(gradients,
                                            self.task_config.gradient_clip_norm)

    # Apply gradients to the model
    optimizer.apply_gradients(zip(gradients, train_vars))
    logs = {self.loss: metric_loss}

    # Compute all metrics
    if metrics:
      for m in metrics:
        m.update_state(loss_metrics[m.name])
        logs.update({m.name: m.result()})
    return logs

  ## evaluation ##
  def _reorg_boxes(self, boxes, num_detections, image):
    """This function is used to reorganize and clip the predicitions to remove
    all padding and only take predicitions within the image"""

    # Build a prediciton mask to take only the number of detections
    mask = tf.sequence_mask(num_detections, maxlen=tf.shape(boxes)[1])
    mask = tf.cast(tf.expand_dims(mask, axis=-1), boxes.dtype)

    # Denormalize the boxes by the shape of the image
    inshape = tf.cast(preprocessing_ops.get_image_shape(image), boxes.dtype)
    boxes = box_ops.denormalize_boxes(boxes, inshape)

    # Mask the boxes for usage
    boxes *= mask
    boxes += (mask - 1)
    return boxes

  def validation_step(self, inputs, model, metrics=None):
    image, label = inputs

    # Step the model once
    y_pred = model(image, training=False)
    y_pred = tf.nest.map_structure(lambda x: tf.cast(x, tf.float32), y_pred)
    (_, metric_loss, loss_metrics) = self.build_losses(y_pred['raw_output'],
                                                       label)
    logs = {self.loss: metric_loss}

    # Reorganize and rescale the boxes
    boxes = self._reorg_boxes(y_pred['bbox'], y_pred['num_detections'], image)
    label['groundtruths']["boxes"] = self._reorg_boxes(
        label['groundtruths']["boxes"], label['groundtruths']["num_detections"],
        image)

    # Build the input for the coc evaluation metric
    coco_model_outputs = {
        'detection_boxes': boxes,
        'detection_scores': y_pred['confidence'],
        'detection_classes': y_pred['classes'],
        'num_detections': y_pred['num_detections'],
        'source_id': label['groundtruths']['source_id'],
        'image_info': label['groundtruths']['image_info']
    }

    # Compute all metrics
    if metrics:
      logs.update(
          {self.coco_metric.name: (label['groundtruths'], coco_model_outputs)})
      for m in metrics:
        m.update_state(loss_metrics[m.name])
        logs.update({m.name: m.result()})
    return logs

  def aggregate_logs(self, state=None, step_outputs=None):
    if not state:
      self.coco_metric.reset_states()
      state = self.coco_metric
    self.coco_metric.update_state(step_outputs[self.coco_metric.name][0],
                                  step_outputs[self.coco_metric.name][1])
    return state

  def reduce_aggregated_logs(self, aggregated_logs, global_step=None):
    res = self.coco_metric.result()
    return res

  def initialize(self, model: tf.keras.Model):
    """Loading pretrained checkpoint."""
    if not self.task_config.init_checkpoint:
      logging.info("Training from Scratch.")
      return

    ckpt_dir_or_file = self.task_config.init_checkpoint
    if tf.io.gfile.isdir(ckpt_dir_or_file):
      ckpt_dir_or_file = tf.train.latest_checkpoint(ckpt_dir_or_file)

    # Restoring checkpoint.
    if self.task_config.init_checkpoint_modules == 'all':
      ckpt = tf.train.Checkpoint(model=model)
      status = ckpt.restore(ckpt_dir_or_file)
      status.expect_partial().assert_existing_objects_matched()
    elif self.task_config.init_checkpoint_modules == 'backbone':
      ckpt = tf.train.Checkpoint(backbone=model.backbone)
      status = ckpt.restore(ckpt_dir_or_file)
      status.expect_partial().assert_existing_objects_matched()
    elif self.task_config.init_checkpoint_modules == 'decoder':
      ckpt = tf.train.Checkpoint(backbone=model.backbone, decoder=model.decoder)
      status = ckpt.restore(ckpt_dir_or_file)
      status.expect_partial()  #.assert_existing_objects_matched()
    else:
      assert "Only 'all' or 'backbone' can be used to initialize the model."

    logging.info('Finished loading pretrained checkpoint from %s',
                 ckpt_dir_or_file)

  def _wrap_optimizer(self, optimizer, runtime_config):
    if runtime_config and runtime_config.loss_scale:
      use_float16 = runtime_config.mixed_precision_dtype == "float16"
      optimizer = performance.configure_optimizer(
          optimizer,
          use_graph_rewrite=False,
          use_float16=use_float16,
          loss_scale=runtime_config.loss_scale)
    return optimizer

  def create_optimizer(self,
                       optimizer_config: OptimizationConfig,
                       runtime_config: Optional[RuntimeConfig] = None):
    """Creates an TF optimizer from configurations.

    Args:
      optimizer_config: the parameters of the Optimization settings.
      runtime_config: the parameters of the runtime.

    Returns:
      A tf.optimizers.Optimizer object.
    """
    opt_factory = optimization.YoloOptimizerFactory(optimizer_config)
    ema = opt_factory._use_ema
    opt_factory._use_ema = False
    if (self._task_config.model.head.smart_bias):
      optimizer = opt_factory.build_optimizer(opt_factory.build_learning_rate())
      optimizer.set_bias_lr(
          opt_factory.get_bias_lr_schedule(self._task_config.smart_bias_lr))

      weights, biases, others = self._model.get_weight_groups(
          self._model.trainable_variables)
      optimizer.set_params(weights, biases, others)
    else:
      optimizer = opt_factory.build_optimizer(opt_factory.build_learning_rate())
    opt_factory._use_ema = ema
    optimizer = opt_factory.add_ema(optimizer)
    optimizer = self._wrap_optimizer(optimizer, runtime_config)
    return optimizer


class ListMetrics(object):

  def __init__(self, metric_names, name="ListMetrics", **kwargs):
    self.name = name
    self._metric_names = metric_names
    self._metrics = self.build_metric()
    return

  def build_metric(self):
    metric_names = self._metric_names
    metrics = []
    for name in metric_names:
      metrics.append(tf.keras.metrics.Mean(name, dtype=tf.float32))
    return metrics

  def update_state(self, loss_metrics):
    metrics = self._metrics
    for m in metrics:
      m.update_state(loss_metrics[m.name])
    return

  def result(self):
    logs = dict()
    metrics = self._metrics
    for m in metrics:
      logs.update({m.name: m.result()})
    return logs

  def reset_states(self):
    metrics = self._metrics
    for m in metrics:
      m.reset_states()
    return

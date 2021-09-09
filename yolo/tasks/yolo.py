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
from official.vision.beta.dataloaders import tf_example_decoder
from official.vision.beta.dataloaders import tfds_detection_decoders
from official.vision.beta.dataloaders import tf_example_label_map_decoder
from yolo.configs import yolo as exp_cfg

from yolo.ops import mosaic
from yolo.ops.kmeans_anchors import BoxGenInputReader
from yolo.dataloaders import yolo_input
from yolo import optimization
from yolo.ops import preprocessing_ops

from tensorflow.keras.mixed_precision import experimental as mixed_precision

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
    self._loss_dict = None
    self._num_boxes = None
    self._anchors_built = False

    self._model = None
    self._masks = None
    self._path_scales = None
    self._x_y_scales = None
    self.coco_metric = None

    self._metrics = []
    self._use_reduced_logs = self.task_config.reduced_logs
    return

  def build_model(self):
    """get an instance of Yolo v3 or v4"""
    from yolo.modeling.factory import build_yolo
    params = self.task_config.train_data
    model_base_cfg = self.task_config.model
    l2_weight_decay = self.task_config.weight_decay / 2.0

    masks, path_scales, xy_scales = self._get_masks()

    _, anchor_free = self._get_boxes(gen_boxes=params.is_training)

    input_size = model_base_cfg.input_size.copy()
    if model_base_cfg.dynamic_conv:
      print("WARNING: dynamic convolution is only supported on GPU and may \
             require significantly more memory. Validation will only work at \
             a batchsize of 1. The model will be trained at the input \
             resolution and evaluated at a dynamic resolution")
      input_size[0], input_size[1] = None, None
    input_specs = tf.keras.layers.InputSpec(shape=[None] + input_size)
    l2_regularizer = (
        tf.keras.regularizers.l2(l2_weight_decay) if l2_weight_decay else None)

    model, losses = build_yolo(input_specs, model_base_cfg, l2_regularizer,
                               masks, xy_scales, path_scales)

    model.summary()
    if anchor_free is not None:
      print("INFO: The model is operating under anchor free conditions")

    self._loss_dict = losses
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
            regenerate_source_id=decoder_cfg.regenerate_source_id)
      elif params.decoder.type == 'label_map_decoder':
        decoder = tf_example_label_map_decoder.TfExampleDecoderLabelMap(
            label_map=decoder_cfg.label_map,
            regenerate_source_id=decoder_cfg.regenerate_source_id)
      else:
        raise ValueError('Unknown decoder type: {}!'.format(
            params.decoder.type))
    return decoder

  def _get_masks(self):

    def _build(values):
      if "all" in values and values["all"] is not None:
        for key in values:
          if key != 'all':
            values[key] = values["all"]
      return values

    start = 0
    masks = {}
    params = self.task_config.model

    if self._masks is None or self._path_scales is None or self._x_y_scales is None:
      for i in range(params.min_level, params.max_level + 1):
        masks[str(i)] = list(range(start, params.boxes_per_scale + start))
        start += params.boxes_per_scale

      self._masks = masks
      self._path_scales = _build(params.filter.path_scales.as_dict())
      self._x_y_scales = _build(params.filter.scale_xy.as_dict())

    return self._masks, self._path_scales, self._x_y_scales

  def build_inputs(self, params, input_context=None):
    """Build input dataset."""

    decoder = self.get_decoder(params)
    model = self.task_config.model

    masks, _, xy_scales = self._get_masks()
    anchors, anchor_free_limits = self._get_boxes(gen_boxes=params.is_training)

    rsize = params.parser.mosaic.resize
    if rsize is None:
      rsize = params.parser.resize

    rcrop = params.parser.mosaic.jitter
    if rcrop is None:
      rcrop = params.parser.jitter

    osize = params.parser.mosaic.output_resolution
    if osize is None:
      osize = model.input_size

    sample_fn = mosaic.Mosaic(
        output_size=osize,
        max_resolution=params.parser.mosaic.max_resolution,
        mosaic_frequency=params.parser.mosaic.mosaic_frequency,
        mixup_frequency=params.parser.mosaic.mixup_frequency,
        crop_area=params.parser.mosaic.crop_area,
        crop_area_mosaic=params.parser.mosaic.crop_area_mosaic,
        mosaic_crop_mode=params.parser.mosaic.mosaic_crop_mode,
        aspect_ratio_mode=params.parser.mosaic.aspect_ratio_mode,
        random_crop=rcrop,
        random_pad=params.parser.mosaic.random_pad,
        translate=params.parser.aug_rand_translate,
        resize=rsize,
        seed=params.seed,
        deterministic=params.seed != None,
        area_thresh=params.parser.area_thresh)

    parser = yolo_input.Parser(
        output_size=model.input_size,
        min_level=model.min_level,
        max_level=model.max_level,
        masks=masks,
        anchors=anchors,
        letter_box=params.parser.letter_box,
        use_tie_breaker=params.parser.use_tie_breaker,
        random_flip=params.parser.random_flip,
        jitter=params.parser.jitter,
        resize=params.parser.resize,
        jitter_mosaic=params.parser.jitter_mosaic,
        resize_mosaic=params.parser.resize_mosaic,
        aug_rand_transalate=params.parser.aug_rand_translate,
        aug_rand_saturation=params.parser.aug_rand_saturation,
        aug_rand_brightness=params.parser.aug_rand_brightness,
        aug_scale_min=params.parser.aug_scale_min,
        aug_scale_max=params.parser.aug_scale_max,
        mosaic_min=params.parser.mosaic_scale_min,
        mosaic_max=params.parser.mosaic_scale_max,
        mosaic_translate=params.parser.mosaic_translate,
        random_pad=params.parser.random_pad,
        aug_rand_hue=params.parser.aug_rand_hue,
        aug_rand_angle=params.parser.aug_rand_angle,
        max_num_instances=params.parser.max_num_instances,
        dynamic_conv=model.dynamic_conv,
        scale_xy=xy_scales,
        stride=params.parser.stride,
        area_thresh=params.parser.area_thresh,
        use_scale_xy=params.parser.use_scale_xy,
        best_match_only=params.parser.best_match_only,
        anchor_t=params.parser.anchor_thresh,
        coco91to80=self.task_config.coco91to80,
        anchor_free_limits=anchor_free_limits,
        seed=params.seed,
        dtype=params.dtype)

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

    self._get_masks()
    metric_names = defaultdict(list)
    for key in self._masks.keys():
      metric_names[key].append('loss')
      metric_names[key].append("avg_iou")
      metric_names[key].append("avg_obj")

      if not self._use_reduced_logs:
        metric_names[key].append('box_loss')
        metric_names[key].append('class_loss')
        metric_names[key].append('conf_loss')

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
    metric_dict = defaultdict(dict)
    metric_dict['net']['box'] = 0
    metric_dict['net']['class'] = 0
    metric_dict['net']['conf'] = 0
    loss_val = 0
    metric_loss = 0

    for key in outputs.keys():
      (_loss, _loss_box, _loss_conf, _loss_class, _mean_loss, _avg_iou,
       _avg_obj) = self._loss_dict[key](labels['true_conf'][key], 
                                        labels['inds'][key], 
                                        labels['upds'][key],
                                        labels['bbox'], 
                                        labels['classes'],
                                        outputs[key])
      loss_val += _loss

      # detach all the below gradients: none of them should make a
      # contribution to the gradient form this point forwards
      metric_loss += tf.stop_gradient(_mean_loss)
      metric_dict[key]['loss'] = tf.stop_gradient(_mean_loss)
      metric_dict[key]['avg_iou'] = tf.stop_gradient(_avg_iou)
      metric_dict[key]["avg_obj"] = tf.stop_gradient(_avg_obj)

      metric_dict['net']['box'] += tf.stop_gradient(_loss_box)
      metric_dict['net']['class'] += tf.stop_gradient(_loss_class)
      metric_dict['net']['conf'] += tf.stop_gradient(_loss_conf)

      if not self._use_reduced_logs:
        metric_dict[key]['conf_loss'] = tf.stop_gradient(_loss_conf)
        metric_dict[key]['box_loss'] = tf.stop_gradient(_loss_box)
        metric_dict[key]['class_loss'] = tf.stop_gradient(_loss_class)

    # Account for model distribution across devices
    num_replicas = tf.distribute.get_strategy().num_replicas_in_sync
    scale = 1
    if self._task_config.model.filter.use_scaled_loss:
      num_replicas = 1
      scale = 3 / len(list(outputs.keys()))

    scale = tf.cast(scale, tf.float32)
    loss_val = loss_val * scale/num_replicas
    return loss_val, metric_loss, metric_dict

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
  def _reorg_boxes(self, boxes, num_detections, info, image):
    """This function is used to reorganize and clip the predicitions to remove
    all padding and only take predicitions within the image"""

    # Build a prediciton mask to take only the number of detections
    mask = tf.sequence_mask(num_detections, maxlen=tf.shape(boxes)[1])
    mask = tf.cast(tf.expand_dims(mask, axis=-1), boxes.dtype)

    # Denormalize the boxes by the shape of the image
    if self.task_config.model.dynamic_conv:
      # Split all infos
      inshape = tf.expand_dims(info[:, 1, :], axis=1)
      ogshape = tf.expand_dims(info[:, 0, :], axis=1)
      scale = tf.expand_dims(info[:, 2, :], axis=1)
      offset = tf.expand_dims(info[:, 3, :], axis=1)

      # Clip the boxes to remove all padding
      boxes = box_ops.denormalize_boxes(boxes, inshape)
      boxes /= tf.tile(scale, [1, 1, 2])
      boxes += tf.tile(offset, [1, 1, 2])
      boxes = box_ops.clip_boxes(boxes, ogshape)
    else:
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
    boxes = self._reorg_boxes(
        y_pred['bbox'], y_pred['num_detections'],
        tf.cast(label['groundtruths']['image_info'], tf.float32), image)
    label['groundtruths']["boxes"] = self._reorg_boxes(
        label['groundtruths']["boxes"], label['groundtruths']["num_detections"],
        tf.cast(label['groundtruths']['image_info'], tf.float32), image)

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
    ret_dict = dict()
    if self._use_reduced_logs:
      ret_dict["AP"] = res["AP"]
      ret_dict["AP50"] = res["AP50"]
      ret_dict["AP75"] = res["AP75"]
      ret_dict["APs"] = res["APs"]
      ret_dict["APm"] = res["APm"]
      ret_dict["APl"] = res["APl"]
      ret_dict = {"AP": ret_dict}
    else:
      ret_dict.update(res)
    return ret_dict

  def _get_boxes(self, gen_boxes=True):
    """Checks for boxes or calls kmeans to auto generate a set of boxes"""
    if gen_boxes and self.task_config.model._boxes is None and not self._anchors_built:
      params = self.task_config.train_data
      decoder = self.get_decoder(params)
      model_base_cfg = self.task_config.model
      self._num_boxes = (model_base_cfg.max_level - model_base_cfg.min_level +
                         1) * model_base_cfg.boxes_per_scale
      reader = BoxGenInputReader(
          params,
          decoder_fn=decoder.decode,
          transform_and_batch_fn=lambda x, y: x,
          parser_fn=None)
      anchors = reader.read(
          k=self._num_boxes,
          image_width=self._task_config.model.input_size[0],
          input_context=None)
      self.task_config.model.set_boxes(anchors)
      self._anchors_built = True
      del reader
    return (self.task_config.model._boxes,
            self.task_config.model.anchor_free_limits)



  def initialize(self, model: tf.keras.Model):
    """initialize the weights of the model"""
    if self.task_config.load_darknet_weights:
      from yolo.utils import DarkNetConverter
      from yolo.utils._darknet2tf.load_weights import split_converter
      from yolo.utils._darknet2tf.load_weights2 import load_weights_backbone
      from yolo.utils._darknet2tf.load_weights2 import load_weights_decoder
      from yolo.utils._darknet2tf.load_weights2 import load_weights_prediction_layers
      from yolo.utils.downloads.file_manager import download

      weights_file = self.task_config.model.darknet_weights_file
      config_file = self.task_config.model.darknet_weights_cfg

      if ('cache' not in weights_file and 'cache' not in config_file):
        list_encdec = DarkNetConverter.read(config_file, weights_file)
      else:
        import os
        path = os.path.abspath('cache')
        if (not os.path.isdir(path)):
          os.mkdir(path)

        cfg = f"{path}/cfg/{config_file.split('/')[-1]}"
        if not os.path.isfile(cfg):
          download(config_file.split('/')[-1])

        wgt = f"{path}/weights/{weights_file.split('/')[-1]}"
        if not os.path.isfile(wgt):
          download(weights_file.split('/')[-1])

        list_encdec = DarkNetConverter.read(cfg, wgt)

      splits = model.backbone._splits
      if 'neck_split' in splits.keys():
        encoder, neck, decoder = split_converter(list_encdec,
                                                 splits['backbone_split'],
                                                 splits['neck_split'])
      else:
        encoder, decoder = split_converter(list_encdec,
                                           splits['backbone_split'])
        neck = None

      load_weights_backbone(model.backbone, encoder)

      if self.task_config.darknet_load_decoder:
        cfgheads = load_weights_decoder(
            model.decoder, [neck, decoder],
            csp=self._task_config.model.base.decoder.type == 'csp')
        load_weights_prediction_layers(cfgheads, model.head)

    else:
      """Loading pretrained checkpoint."""
      if not self.task_config.init_checkpoint:
        print("loaded nothing")
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
        ckpt = tf.train.Checkpoint(
            backbone=model.backbone, decoder=model.decoder)
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
    if (self._task_config.smart_bias_lr > 0.0):
      optimizer = opt_factory.build_optimizer(opt_factory.build_learning_rate())
      optimizer.set_bias_lr(
          opt_factory.get_bias_lr_schedule(self._task_config.smart_bias_lr))

      weights, biases, others = self._model.get_weight_groups(
          self._model.trainable_variables)
      optimizer.set_params(weights, biases, others)
    else:
      optimizer = opt_factory.build_optimizer(opt_factory.build_learning_rate())

    print(optimizer)

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

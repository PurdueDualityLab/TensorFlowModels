from numpy import blackman
from tensorflow.keras import optimizers
from tensorflow.python.keras.backend import int_shape
from tensorflow.python.ops.clip_ops import clip_by_value
from tensorflow.python.ops.gen_array_ops import shape
from tensorflow.python.training import optimizer
from yolo.ops.preprocessing_ops import apply_infos
import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as mixed_precision

from absl import logging
from official.core import base_task
from official.core import input_reader
from official.core import task_factory
from yolo.configs import yolo as exp_cfg

from official.vision.beta.evaluation import coco_evaluator
from official.vision.beta.dataloaders import tf_example_decoder
from official.vision.beta.dataloaders import tfds_detection_decoders
from official.vision.beta.dataloaders import tf_example_label_map_decoder

from yolo.dataloaders import yolo_input
from yolo.ops import mosaic
from yolo.ops.kmeans_anchors import BoxGenInputReader
from yolo.ops.box_ops import xcycwh_to_yxyx

from official.vision.beta.ops import box_ops, preprocess_ops
from yolo.modeling.layers import detection_generator
from collections import defaultdict

from typing import Optional
from official.core import config_definitions
from yolo import optimization 
from official.modeling import performance


from yolo.optimization.CompositeOptimizer import CompositeOptimizer

OptimizationConfig = optimization.OptimizationConfig
RuntimeConfig = config_definitions.RuntimeConfig

class AssignMetric(tf.keras.metrics.Metric):

  def __init__(self, name, dtype, **kwargs):
    super().__init__(name=name, dtype=dtype, **kwargs)
    self.value = self.add_weight('value')

  def update_state(self, value):
    self.value.assign(value)
    return 

  def result(self):
    return self.value

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
      #if name != "iterations" and name != "bias_LR":
      metrics.append(tf.keras.metrics.Mean(name, dtype=tf.float32))
      # else:
      #   metrics.append(AssignMetric(name, dtype=tf.float32))
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
    # super().reset_states()
    metrics = self._metrics
    for m in metrics:
      m.reset_states()
    return


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
    self._metric_names = []
    self._metrics = []
    
    # self._test_var = tf.Variable(0, trainable=False)
    # self._var_names = []
    return

  def build_model(self):
    """get an instance of Yolo v3 or v4"""
    # from yolo.modeling.yolo_model import build_yolo
    from yolo.modeling.factory import build_yolo
    params = self.task_config.train_data
    model_base_cfg = self.task_config.model
    l2_weight_decay = self.task_config.weight_decay / 2.0

    masks, path_scales, xy_scales = self._get_masks()

    anchors = self._get_boxes(gen_boxes=params.is_training)

    input_size = model_base_cfg.input_size.copy()
    if model_base_cfg.dynamic_conv:
      print("WARNING: dynamic convolution is only supported on GPU and may \
             require significantly more memory. Validation will only work at \
             a batchsize of 1. The model will be trained at the input \
             resolution and evaluated at a dynamic resolution")
      input_size[0] = None
      input_size[1] = None
    input_specs = tf.keras.layers.InputSpec(shape=[None] + input_size)
    l2_regularizer = (
        tf.keras.regularizers.l2(l2_weight_decay) if l2_weight_decay else None)

    model, losses = build_yolo(input_specs, model_base_cfg, l2_regularizer,
                               masks, xy_scales, path_scales)

    model.print()

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

  def build_inputs(self, params, input_context=None):
    """Build input dataset."""

    decoder = self.get_decoder(params)
    model = self.task_config.model

    masks, path_scales, xy_scales = self._get_masks()
    anchors = self._get_boxes(gen_boxes=params.is_training)

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
        random_pad=params.parser.random_pad,
        translate=params.parser.aug_rand_translate,
        resize=rsize,
        seed=params.seed, 
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
        sheer=params.parser.sheer,
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
        seed=params.seed, 
        dtype=params.dtype)

    reader = input_reader.InputReader(
        params,
        dataset_fn=tf.data.TFRecordDataset,
        decoder_fn=decoder.decode,
        sample_fn=sample_fn.mosaic_fn(is_training=params.is_training),
        parser_fn=parser.parse_fn(params.is_training))
    dataset = reader.read(input_context=input_context)

    print(dataset)
    return dataset

  def build_losses(self, outputs, labels, aux_losses=None):
    metric_dict = defaultdict(dict)
    loss_val = 0
    metric_dict['global']['total_loss'] = 0
    metric_dict['global']['total_box'] = 0
    metric_dict['global']['total_class'] = 0
    metric_dict['global']['total_conf'] = 0

    grid = labels['true_conf']
    inds = labels['inds']
    upds = labels['upds']

    scale = tf.cast(3 / len(list(outputs.keys())), tf.float32)
    for key in outputs.keys():
      (_loss, _loss_box, _loss_conf, _loss_class, _mean_loss, _avg_iou,
       _avg_obj, _recall50,
       _precision50) = self._loss_dict[key](grid[key], inds[key], upds[key],
                                            labels['bbox'], labels['classes'],
                                            outputs[key])
      loss_val += _loss

      # detach all the below gradients: none of them should make a contribution to the
      # gradient form this point forwards
      metric_dict['global']['total_loss'] += tf.stop_gradient(_mean_loss)
      metric_dict['global']['total_box'] += tf.stop_gradient(_loss_box)
      metric_dict['global']['total_class'] += tf.stop_gradient(_loss_class)
      metric_dict['global']['total_conf'] += tf.stop_gradient(_loss_conf)
      metric_dict[key]['conf_loss'] = tf.stop_gradient(_loss_conf)
      metric_dict[key]['box_loss'] = tf.stop_gradient(_loss_box)
      metric_dict[key]['class_loss'] = tf.stop_gradient(_loss_class)
      metric_dict[key]["recall50"] = tf.stop_gradient(_recall50)
      metric_dict[key]["precision50"] = tf.stop_gradient(_precision50)
      metric_dict[key]["avg_iou"] = tf.stop_gradient(_avg_iou)
      metric_dict[key]["avg_obj"] = tf.stop_gradient(_avg_obj)

    return loss_val * scale, metric_dict

  def build_metrics(self, training=True):
    metrics = []
    metric_names = self._metric_names

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

  ## training ##
  def train_step(self, inputs, model, optimizer, metrics=None):
    # get the data point
    image, label = inputs

    num_replicas = tf.distribute.get_strategy().num_replicas_in_sync
    # if self._task_config.model.filter.use_scaled_loss:
    #   num_replicas = 1
      
    with tf.GradientTape(persistent=False) as tape:
      # compute a prediction
      y_pred = model(image, training=True)

      # cast to float 32
      y_pred = tf.nest.map_structure(lambda x: tf.cast(x, tf.float32), y_pred)

      # get the total loss
      loss, loss_metrics = self.build_losses(y_pred['raw_output'], label)

      # TF will aggregate gradients via sum, so we need to divide by the world
      # size when computing the mean of loss over batches. For scaled loss
      # we want the sum over all batches, so we instead use num replicas equal
      # to 1 in order to aggregate the sum of the gradients
      scaled_loss = loss / num_replicas

      # scale the loss for numerical stability
      if isinstance(optimizer, mixed_precision.LossScaleOptimizer):
        scaled_loss = optimizer.get_scaled_loss(scaled_loss)

    # compute the gradient
    train_vars = model.trainable_variables
    gradients = tape.gradient(scaled_loss, train_vars)

    if self._task_config.model.filter.use_scaled_loss:
      gradients = [gradient * num_replicas for gradient in gradients]

    # get unscaled loss if the scaled_loss was used
    if isinstance(optimizer, mixed_precision.LossScaleOptimizer):
      gradients = optimizer.get_unscaled_gradients(gradients)

    if self.task_config.gradient_clip_norm > 0.0:
      gradients, _ = tf.clip_by_global_norm(gradients,
                                            self.task_config.gradient_clip_norm)

    # tf.print(loss, scaled_loss, loss_metrics['global']['total_loss'], tf.reduce_sum(gradients[-2]))

    optimizer.apply_gradients(zip(gradients, train_vars))
    logs = {self.loss: loss}


    if metrics:
      for m in metrics:
        m.update_state(loss_metrics[m.name])
        logs.update({m.name: m.result()})
    return logs


  ## evaluation ##
  def _reorg_boxes(self, boxes, num_detections, info):
    mask = tf.sequence_mask(num_detections, maxlen=tf.shape(boxes)[1])
    mask = tf.cast(tf.expand_dims(mask, axis=-1), boxes.dtype)

    # split all infos
    inshape = tf.expand_dims(info[:, 1, :], axis=1)
    ogshape = tf.expand_dims(info[:, 0, :], axis=1)
    scale = tf.expand_dims(info[:, 2, :], axis=1)
    offset = tf.expand_dims(info[:, 3, :], axis=1)

    # reorg to image shape
    boxes = box_ops.denormalize_boxes(boxes, inshape)

    if self.task_config.model.dynamic_conv:
      boxes /= tf.tile(scale, [1, 1, 2])
      boxes += tf.tile(offset, [1, 1, 2])
      boxes = box_ops.clip_boxes(boxes, ogshape)

    # mask the boxes for usage
    boxes *= mask
    boxes += (mask - 1)
    return boxes

  def validation_step(self, inputs, model, metrics=None):
    # get the data point
    image, label = inputs

    y_pred = model(image, training=False)
    y_pred = tf.nest.map_structure(lambda x: tf.cast(x, tf.float32), y_pred)
    loss, loss_metrics = self.build_losses(y_pred['raw_output'], label)
    logs = {self.loss: loss_metrics['global']['total_loss']}

    boxes = self._reorg_boxes(
        y_pred['bbox'], y_pred['num_detections'],
        tf.cast(label['groundtruths']['image_info'], tf.float32))
    label['groundtruths']["boxes"] = self._reorg_boxes(
        label['groundtruths']["boxes"], label['groundtruths']["num_detections"],
        tf.cast(label['groundtruths']['image_info'], tf.float32))

    coco_model_outputs = {
        'detection_boxes': boxes,
        'detection_scores': y_pred['confidence'],
        'detection_classes': y_pred['classes'],
        'num_detections': y_pred['num_detections'],
        'source_id': label['groundtruths']['source_id'],
        'image_info': label['groundtruths']['image_info']
    }


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
    return self.coco_metric.result()

  @property
  def anchors(self):
    return self.task_config.model.boxes

  def _get_boxes(self, gen_boxes=True):

    if gen_boxes and self.task_config.model._boxes is None and not self._anchors_built:
      # must save the boxes!
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
    return self.task_config.model._boxes

  def _get_masks(self,
                 xy_exponential=True,
                 exp_base=2,
                 xy_scale_base='default_value'):

    def _build(values):
      if "all" in values and values["all"] is not None:
        for key in values:
          if key != 'all':
            values[key] = values["all"]
      print(values)
      return values

    start = 0
    boxes = {}
    path_scales = {}
    scale_x_y = {}

    if xy_scale_base == 'default_base':
      xy_scale_base = 0.05
      xy_scale_base = xy_scale_base / (
          self._boxes_per_level * (self._max_level - self._min_level + 1) - 1)
    elif xy_scale_base == 'default_value':
      xy_scale_base = 0.00625

    params = self.task_config.model

    if self._masks is None or self._path_scales is None or self._x_y_scales is None:
      for i in range(params.min_level, params.max_level + 1):
        boxes[str(i)] = list(range(start, params.boxes_per_scale + start))
        start += params.boxes_per_scale

      self._masks = boxes
      self._path_scales = _build(params.filter.path_scales.as_dict())
      self._x_y_scales = _build(params.filter.scale_xy.as_dict())

    metric_names = defaultdict(list)
    for key in self._masks.keys():
      metric_names[key].append('box_loss')
      metric_names[key].append('class_loss')
      metric_names[key].append('conf_loss')
      metric_names[key].append("recall50")
      metric_names[key].append("precision50")
      metric_names[key].append("avg_iou")
      metric_names[key].append("avg_obj")

    metric_names['global'].append('total_loss')
    metric_names['global'].append('total_box')
    metric_names['global'].append('total_class')
    metric_names['global'].append('total_conf')
    # metric_names['global'].append('iterations')
    # if self.task_config.model.smart_bias:
    #   metric_names['global'].append('bias_LR')

    print(metric_names)

    # metric_names.append('darknet_loss')
    self._metric_names = metric_names

    return self._masks, self._path_scales, self._x_y_scales

  def initialize(self, model: tf.keras.Model):

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

  # @classmethod
  # def create_optimizer(cls, optimizer_config: OptimizationConfig,
  #                      runtime_config: Optional[RuntimeConfig] = None):

  def _wrap_optimizer(self, optimizer, runtime_config):
    if runtime_config and runtime_config.loss_scale:
      use_float16 = runtime_config.mixed_precision_dtype == "float16"
      optimizer = performance.configure_optimizer(
          optimizer,
          use_graph_rewrite=False,
          use_float16=use_float16,
          loss_scale=runtime_config.loss_scale)
    return optimizer

  def create_optimizer(self, optimizer_config: OptimizationConfig,
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
      optimizer_weights = opt_factory.build_optimizer(opt_factory.build_learning_rate())
      optimizer_others = opt_factory.build_optimizer(opt_factory.build_learning_rate())
      optimizer_biases = opt_factory.build_optimizer(opt_factory.get_bias_lr_schedule(self._task_config.smart_bias_lr))

      optimizer_weights.name = "weights_lr"
      optimizer_others.name = "others_lr"
      optimizer_biases.name = "bias_lr"
      weights, bias, other = self._model.get_weight_groups(self._model.trainable_variables)
      optimizer = CompositeOptimizer(
        [(optimizer_weights, lambda:weights),
        (optimizer_biases, lambda:bias),
        (optimizer_others, lambda:other)]
      )

    else:
      optimizer = opt_factory.build_optimizer(opt_factory.build_learning_rate())

    print(optimizer)
    # optimizer = self._wrap_optimizer(optimizer, runtime_config)
    opt_factory._use_ema = ema
    optimizer = opt_factory.add_ema(optimizer)
    return optimizer
# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Contains classes used to train Yolo."""

from absl import logging
import collections

from official.core import base_task
from official.core import input_reader
from official.core import task_factory
from official.core import config_definitions
from official.modeling import performance
from official.vision.beta.ops import box_ops
from official.vision.beta.evaluation import coco_evaluator
from official.vision.beta.dataloaders import tfds_factory
from official.vision.beta.dataloaders import tf_example_label_map_decoder

from yolo import optimization
from yolo.ops import mosaic
from yolo.ops import preprocessing_ops
from yolo.ops import kmeans_anchors
from yolo.dataloaders import yolo_input
from yolo.dataloaders import tf_example_decoder
from yolo.configs import yolo as exp_cfg

import tensorflow as tf
from typing import Optional

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
    self._coco_91_to_80 = False
    self._annotation_file = self.task_config.annotation_file
    self._metrics = []

    # globally set the random seed
    preprocessing_ops.set_random_seeds(seed=params.train_data.seed)
    return

  def build_model(self, batch_size = None):
    """Build an instance of Yolo."""
    from yolo.modeling.factory import build_yolo

    model_base_cfg = self.task_config.model
    l2_weight_decay = self.task_config.weight_decay / 2.0

    input_size = model_base_cfg.input_size.copy()
    input_specs = tf.keras.layers.InputSpec(shape=[batch_size] + input_size)
    l2_regularizer = (
        tf.keras.regularizers.l2(l2_weight_decay) if l2_weight_decay else None)
    model, losses = build_yolo(input_specs, model_base_cfg, 
                               l2_regularizer)

    # save for later usage within the task.
    self._loss_fn = losses
    self._model = model
    return model

  def _get_data_decoder(self, params):
    """Get a decoder object to decode the dataset."""
    if params.tfds_name:
      decoder = tfds_factory.get_detection_decoder(params.tfds_name)
    else:
      decoder_cfg = params.decoder.get()
      if params.decoder.type == 'simple_decoder':
        self._coco_91_to_80 = decoder_cfg.coco91_to_80
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

  def generate_anchors(self, num_anchors = None, input_context = None):
    input_size = self.task_config.model.input_size
    boxes = self.task_config.model.anchor_boxes
    backbone =  self.task_config.model.backbone.get()

    dataset = self.task_config.validation_data
    decoder = self._get_data_decoder(dataset)

    if num_anchors is None:
      num_anchors = backbone.max_level - backbone.min_level + 1
      num_anchors *= boxes.anchors_per_scale

    dataset.global_batch_size = 1
    box_reader = kmeans_anchors.BoxGenInputReader(
        dataset,
        dataset_fn=tf.data.TFRecordDataset,
        decoder_fn=decoder.decode)

    boxes = box_reader.read(
      k = num_anchors, 
      anchors_per_scale = boxes.anchors_per_scale,
      anchors = None if boxes.boxes is None else [box.box for box in boxes.boxes],
      image_resolution = input_size,
      input_context = input_context
    )
    return boxes



  def build_inputs(self, params, input_context=None):
    """Build input dataset."""
    model = self.task_config.model

    # get anchor boxes dict based on models min and max level
    backbone = model.backbone.get()
    anchor_dict, level_limits = model.anchor_boxes.get(backbone.min_level,
                                                       backbone.max_level)

    # set shared patamters between mosaic and yolo_input
    base_config = dict(
        letter_box=params.parser.letter_box,
        aug_rand_translate=params.parser.aug_rand_translate,
        aug_rand_angle=params.parser.aug_rand_angle,
        aug_rand_perspective=params.parser.aug_rand_perspective,
        area_thresh=params.parser.area_thresh,
        random_flip=params.parser.random_flip,
        seed=params.seed,
    )

    # get the decoder
    decoder = self._get_data_decoder(params)

    # init Mosaic
    sample_fn = mosaic.Mosaic(
        output_size=model.input_size,
        mosaic_frequency=params.parser.mosaic.mosaic_frequency,
        mixup_frequency=params.parser.mosaic.mixup_frequency,
        jitter=params.parser.mosaic.jitter,
        mosaic_center=params.parser.mosaic.mosaic_center,
        mosaic_crop_mode=params.parser.mosaic.mosaic_crop_mode,
        aug_scale_min=params.parser.mosaic.aug_scale_min,
        aug_scale_max=params.parser.mosaic.aug_scale_max,
        random_pad=params.parser.mosaic.random_pad,
        **base_config)

    # init Parser
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

    # init the dataset reader
    reader = input_reader.InputReader(
        params,
        dataset_fn=tf.data.TFRecordDataset,
        decoder_fn=decoder.decode,
        sample_fn=sample_fn.mosaic_fn(is_training=params.is_training),
        parser_fn=parser.parse_fn(params.is_training))
    dataset = reader.read(input_context=input_context)
    return dataset

  def build_metrics(self, training=True):
    """Build detection metrics."""
    metrics = []

    backbone = self.task_config.model.backbone.get()
    metric_names = collections.defaultdict(list)
    for key in range(backbone.min_level, backbone.max_level + 1):
      key = str(key)
      metric_names[key].append('loss')
      metric_names[key].append("avg_iou")
      metric_names[key].append("avg_obj")

    metric_names['net'].append('box')
    metric_names['net'].append('class')
    metric_names['net'].append('conf')

    for i, key in enumerate(metric_names.keys()):
      metrics.append(_ListMetrics(metric_names[key], name=key))

    self._metrics = metrics
    if not training:
      annotation_file = self.task_config.annotation_file
      # if self._coco_91_to_80:
      #   # annotation_file = None
        # self._annotation_file = annotation_file
      self.coco_metric = coco_evaluator.COCOEvaluator(
          annotation_file=annotation_file,
          include_mask=False,
          need_rescale_bboxes=False,
          per_category_metrics=self._task_config.per_category_metrics)

    return metrics

  def build_losses(self, outputs, labels, aux_losses=None):
    """Build YOLO losses."""
    return self._loss_fn(labels, outputs)

  def train_step(self, inputs, model, optimizer, metrics=None):
    """Train Step. Forward step and backwards propagate the model.
 
    Args:
      inputs: a dictionary of input tensors.
      model: the model, forward pass definition.
      optimizer: the optimizer for this training step.
      metrics: a nested structure of metrics objects.
 
    Returns:
      A dictionary of logs.
    """
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
      if isinstance(optimizer, tf.keras.mixed_precision.LossScaleOptimizer):
        scaled_loss = optimizer.get_scaled_loss(scaled_loss)

    # Compute the gradient
    train_vars = model.trainable_variables
    gradients = tape.gradient(scaled_loss, train_vars)

    # Get unscaled loss if we are using the loss scale optimizer on fp16
    if isinstance(optimizer, tf.keras.mixed_precision.LossScaleOptimizer):
      gradients = optimizer.get_unscaled_gradients(gradients)

    # Apply gradients to the model
    optimizer.apply_gradients(zip(gradients, train_vars))
    logs = {self.loss: metric_loss}

    # Compute all metrics
    if metrics:
      for m in metrics:
        m.update_state(loss_metrics[m.name])
        logs.update({m.name: m.result()})
    return logs

  def _reorg_boxes(self, boxes, info, num_detections):
    """Scale and Clean boxes prior to Evaluation."""
    mask = tf.sequence_mask(num_detections, maxlen=tf.shape(boxes)[1])
    mask = tf.cast(tf.expand_dims(mask, axis = -1), boxes.dtype)

    # Denormalize the boxes by the shape of the image
    inshape = tf.expand_dims(info[:, 1, :], axis = 1)
    ogshape = tf.expand_dims(info[:, 0, :], axis = 1)
    scale = tf.expand_dims(info[:, 2, :], axis = 1)
    offset = tf.expand_dims(info[:, 3, :], axis = 1)

    boxes = box_ops.denormalize_boxes(boxes, inshape)
    boxes = box_ops.clip_boxes(boxes, inshape)
    boxes += tf.tile(offset, [1, 1, 2])
    boxes /= tf.tile(scale, [1, 1, 2])
    boxes = box_ops.clip_boxes(boxes, ogshape)

    # Mask the boxes for usage
    boxes *= mask
    boxes += (mask - 1)
    return boxes

  def validation_step(self, inputs, model, metrics=None):
    """Validatation step.
 
    Args:
      inputs: a dictionary of input tensors.
      model: the keras.Model.
      metrics: a nested structure of metrics objects.
      
    Returns:
      A dictionary of logs.
    """
    image, label = inputs

    # Step the model once
    y_pred = model(image, training=False)
    y_pred = tf.nest.map_structure(lambda x: tf.cast(x, tf.float32), y_pred)
    (_, metric_loss, loss_metrics) = self.build_losses(y_pred['raw_output'],
                                                       label)
    logs = {self.loss: metric_loss}

    # Reorganize and rescale the boxes
    info = label['groundtruths']['image_info']

    if self._coco_91_to_80 and self._annotation_file is not None:
      # undo the coco80 to 91
      class_ids = [
          1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
          23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
          44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62,
          63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85,
          86, 87, 88, 89, 90
      ]
      new_classes = tf.expand_dims(tf.cast(class_ids, y_pred['classes'].dtype), axis=0)
      new_classes = tf.expand_dims(new_classes, axis=0)

      cmask = tf.range(0, 80)
      cmask = tf.expand_dims(tf.cast(cmask, y_pred['classes'].dtype), axis=0)
      cmask = tf.expand_dims(cmask, axis=0)

      preds = tf.expand_dims(y_pred['classes'], axis=-1)
      mask = tf.cast(preds == cmask, cmask.dtype)
      y_pred['classes'] = tf.reduce_max(mask * new_classes + (mask - 1), axis = -1)

    boxes = self._reorg_boxes(y_pred['bbox'], info, y_pred["num_detections"])

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
    """Get Metric Results."""
    if not state:
      self.coco_metric.reset_states()
      state = self.coco_metric
    self.coco_metric.update_state(step_outputs[self.coco_metric.name][0],
                                  step_outputs[self.coco_metric.name][1])
    return state

  def reduce_aggregated_logs(self, aggregated_logs, global_step=None):
    """Reduce logs and remove unneeded items. Update with COCO results."""
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
      status.expect_partial() 
    else:
      assert "Only 'all' or 'backbone' can be used to initialize the model."

    logging.info('Finished loading pretrained checkpoint from %s',
                 ckpt_dir_or_file)

  # def initialize(self, model: tf.keras.Model):
  #     from yolo.utils import DarkNetConverter
  #     from yolo.utils._darknet2tf.load_weights import split_converter
  #     from yolo.utils._darknet2tf.load_weights2 import load_weights_backbone
  #     from yolo.utils._darknet2tf.load_weights2 import load_weights_decoder
  #     from yolo.utils._darknet2tf.load_weights2 import load_weights_prediction_layers
  #     from yolo.utils.downloads.file_manager import download

  #     weights_file = "/home/vbanna/Research/TensorFlowModels/cache/weights/yolov4-p5.weights"#self.task_config.model.darknet_weights_file
  #     config_file = "/home/vbanna/Research/TensorFlowModels/cache/cfg/yolov4-p5.cfg" #self.task_config.model.darknet_weights_cfg

  #     #if ('cache' not in weights_file and 'cache' not in config_file):
  #     list_encdec = DarkNetConverter.read(config_file, weights_file)
  #     # else:
  #     #   import os
  #     #   path = os.path.abspath('cache')
  #     #   if (not os.path.isdir(path)):
  #     #     os.mkdir(path)

  #     #   cfg = f"{path}/cfg/{config_file.split('/')[-1]}"
  #     #   if not os.path.isfile(cfg):
  #     #     download(config_file.split('/')[-1])

  #     #   wgt = f"{path}/weights/{weights_file.split('/')[-1]}"
  #     #   if not os.path.isfile(wgt):
  #     #     download(weights_file.split('/')[-1])

  #     #   list_encdec = DarkNetConverter.read(cfg, wgt)

  #     splits = model.backbone._splits
  #     if 'neck_split' in splits.keys():
  #       encoder, neck, decoder = split_converter(list_encdec,
  #                                                splits['backbone_split'],
  #                                                splits['neck_split'])
  #     else:
  #       encoder, decoder = split_converter(list_encdec,
  #                                          splits['backbone_split'])
  #       neck = None

  #     load_weights_backbone(model.backbone, encoder)
  #     #model.backbone.trainable = False

  #     # if self.task_config.darknet_load_decoder:
  #     cfgheads = load_weights_decoder(
  #         model.decoder, [decoder, []],
  #         csp= ("csp" in self._task_config.model.decoder.get().type))
  #     load_weights_prediction_layers(cfgheads, model.head)


  def _wrap_optimizer(self, optimizer, runtime_config):
    """Wraps the optimizer object with the loss scale optimizer."""
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

    opt_type = opt_factory._optimizer_type
    if (opt_type == 'sgd_torch'):
      optimizer = opt_factory.build_optimizer(opt_factory.build_learning_rate())
      optimizer.set_bias_lr(
          opt_factory.get_bias_lr_schedule(self._task_config.smart_bias_lr))

      weights, biases, others = self._model.get_weight_groups(
          self._model.trainable_variables)
      optimizer.set_params(weights, biases, others)
    else:
      optimizer = opt_factory.build_optimizer(opt_factory.build_learning_rate())
    opt_factory._use_ema = ema

    if ema:
      logging.info("EMA is enabled.")
    optimizer = opt_factory.add_ema(optimizer)
    # if ema and self.task_config.model.backbone.get().type == "swin":
    #   optimizer.shadow_copy(self._model.decoder)
    optimizer = self._wrap_optimizer(optimizer, runtime_config)
    return optimizer


class _ListMetrics:
  """Private class used to cleanly place the matric values for each level."""

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
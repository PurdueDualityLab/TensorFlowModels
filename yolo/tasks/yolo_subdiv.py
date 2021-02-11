import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as mixed_precision

from absl import logging
from official.core import base_task
from official.core import input_reader
from official.core import task_factory
from yolo.configs import yolo as exp_cfg

from official.vision.beta.evaluation import coco_evaluator

from yolo.dataloaders import yolo_input
from yolo.dataloaders.decoders import tfds_coco_decoder
from yolo.ops.kmeans_anchors import BoxGenInputReader
from yolo.ops.box_ops import xcycwh_to_yxyx

from official.vision.beta.ops import box_ops, preprocess_ops
from yolo.modeling.layers.detection_generator import YoloGTFilter
from yolo.tasks import yolo


@task_factory.register_task_cls(exp_cfg.YoloSubDivTask)
class YoloSubDivTask(yolo.YoloTask):
  """A single-replica view of training procedure.
  RetinaNet task provides artifacts for training/evalution procedures, including
  loading/iterating over Datasets, initializing the model, calculating the loss,
  post-processing, and customized metrics with reduction.
  """

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

    if params.is_training:
      params.global_batch_size = params.global_batch_size // self.task_config.subdivisions
      if params.global_batch_size == 0:
        raise RuntimeError('batchsize must be divisible by the subdivisions')

    parser = yolo_input.Parser(
        image_w=params.parser.image_w,
        image_h=params.parser.image_h,
        num_classes=model.num_classes,
        min_level=model.min_level,
        max_level=model.max_level,
        fixed_size=params.parser.fixed_size,
        jitter_im=params.parser.jitter_im,
        jitter_boxes=params.parser.jitter_boxes,
        masks=masks,
        use_tie_breaker=params.parser.use_tie_breaker,
        min_process_size=params.parser.min_process_size,
        max_process_size=params.parser.max_process_size,
        max_num_instances=params.parser.max_num_instances,
        random_flip=params.parser.random_flip,
        pct_rand=params.parser.pct_rand,
        seed=params.parser.seed,
        aug_rand_saturation=params.parser.aug_rand_saturation,
        aug_rand_brightness=params.parser.aug_rand_brightness,
        aug_rand_zoom=params.parser.aug_rand_zoom,
        aug_rand_hue=params.parser.aug_rand_hue,
        anchors=anchors,
        dtype=params.dtype)

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

    if params.is_training:
      dataset = dataset.batch(self.task_config.subdivisions)
    return dataset

  def build_losses(self, outputs, labels, div=None, aux_losses=None):
    loss = 0.0
    loss_box = 0.0
    loss_conf = 0.0
    loss_class = 0.0
    metric_dict = dict()

    grid = labels['grid_form']
    for key in outputs.keys():
      if div is not None:
        _loss, _loss_box, _loss_conf, _loss_class, _avg_iou, _recall50 = self._loss_dict[
            key](grid[key][div], outputs[key])
      else:
        _loss, _loss_box, _loss_conf, _loss_class, _avg_iou, _recall50 = self._loss_dict[
            key](grid[key], outputs[key])
      loss += _loss
      loss_box += _loss_box
      loss_conf += _loss_conf
      loss_class += _loss_class
      metric_dict[f"recall50_{key}"] = tf.stop_gradient(_recall50)
      metric_dict[f"avg_iou_{key}"] = tf.stop_gradient(_avg_iou)

    metric_dict['box_loss'] = loss_box
    metric_dict['conf_loss'] = loss_conf
    metric_dict['class_loss'] = loss_class
    metric_dict['total_loss'] = loss
    return loss, metric_dict

  def train_step(self, inputs, model, optimizer, metrics=None):
    # get the data point
    image, label = inputs
    num_replicas = tf.distribute.get_strategy().num_replicas_in_sync
    logs = {}
    net_loss = 0

    with tf.GradientTape() as tape:
      # compute a prediction
      # cast to float32
      for i in range(self.task_config.subdivisions):
        y_pred = model(image[i], training=True)
        loss, loss_metrics = self.build_losses(
            y_pred['raw_output'], label, div=i)
        net_loss += loss

        # if metrics:
        #   for m in metrics:
        #     m.update_state(loss_metrics[m.name])
        #     logs.update({m.name: tf.stop_gradient(m.result())})

      scaled_loss = net_loss / num_replicas

      # scale the loss for numerical stability
      if isinstance(optimizer, mixed_precision.LossScaleOptimizer):
        scaled_loss = optimizer.get_scaled_loss(scaled_loss)

    # compute the gradient
    train_vars = model.trainable_variables
    gradients = tape.gradient(scaled_loss, train_vars)
    # get unscaled loss if the scaled_loss was used
    if isinstance(optimizer, mixed_precision.LossScaleOptimizer):
      gradients = optimizer.get_unscaled_gradients(gradients)
    if self.task_config.gradient_clip_norm > 0.0:
      gradients, _ = tf.clip_by_global_norm(gradients,
                                            self.task_config.gradient_clip_norm)
    optimizer.apply_gradients(zip(gradients, train_vars))

    # custom metrics
    logs['loss'] = net_loss

    tf.print(logs, end='\n')

    ret = '\033[F' * (len(logs.keys()) + 1)
    tf.print(ret, end='\n')
    return logs

  def validation_step(self, inputs, model, metrics=None):
    # get the data point
    image, label = inputs

    # computer detivative and apply gradients
    y_pred = model(image, training=False)
    loss, loss_metrics = self.build_losses(y_pred['raw_output'], label)

    # #custom metrics
    logs = {'loss': loss}
    # loss_metrics.update(metrics)
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
            tf.shape(y_pred['bbox'])[:-1],
        'source_id':
            label['source_id'],
    }

    logs.update({self.coco_metric.name: (label, coco_model_outputs)})

    if metrics:
      for m in metrics:
        m.update_state(loss_metrics[m.name])
        logs.update({m.name: m.result()})
    return logs


if __name__ == '__main__':
  import matplotlib.pyplot as plt
  from yolo.utils.run_utils import prep_gpu
  prep_gpu()

  config = exp_cfg.YoloSubDivTask(subdivisions=4)
  task = YoloSubDivTask(config)
  model = task.build_model()
  model.summary()
  task.initialize(model)

  train_data = task.build_inputs(config.train_data)
  test_data = task.build_inputs(config.validation_data)
  metrics = task.build_metrics(training=False)
  # test_data = task.build_inputs(config.task.validation_data)

  optimizer = tf.keras.optimizers.Adam()
  print(test_data)
  for l, (i, j) in enumerate(test_data):
    preds = task.validation_step((i, j), model, metrics=metrics)
    print(preds)
    # boxes = xcycwh_to_yxyx(j['bbox'])

    # i = tf.image.draw_bounding_boxes(i, boxes, [[1.0, 0.0, 0.0]])

    # i = tf.image.draw_bounding_boxes(i, preds['bbox'], [[0.0, 1.0, 0.0]])
    # plt.imshow(i[0].numpy())
    # plt.show()

    if l > 2:
      break

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


@task_factory.register_task_cls(exp_cfg.YoloTask)
class YoloTask(base_task.Task):
  """A single-replica view of training procedure.
    RetinaNet task provides artifacts for training/evalution procedures, including
    loading/iterating over Datasets, initializing the model, calculating the loss,
    post-processing, and customized metrics with reduction.
    """

        model, losses = build_yolo(input_specs, model_base_cfg, l2_regularizer, masks, xy_scales, path_scales)
        self._loss_dict = losses
        return model


    def build_inputs(self, params, input_context=None):
        """Build input dataset."""
        decoder = tfds_coco_decoder.MSCOCODecoder()
        '''
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
        '''


        model = self.task_config.model

        masks, path_scales, xy_scales = self._get_masks()
        anchors = self._get_boxes(gen_boxes=params.is_training)

        print(masks, path_scales, xy_scales)
        parser = yolo_input.Parser(
                    image_w=params.parser.image_w,
                    image_h=params.parser.image_h,
                    num_classes=model.num_classes,
                    min_level=model.min_level,
                    max_level=model.max_level,
                    fixed_size=params.parser.fixed_size,
                    jitter_im=params.parser.jitter_im,
                    jitter_boxes=params.parser.jitter_boxes,
                    masks = masks,
                    use_tie_breaker = params.parser.use_tie_breaker,
                    min_process_size=params.parser.min_process_size,
                    max_process_size=params.parser.max_process_size,
                    max_num_instances = params.parser.max_num_instances,
                    random_flip = params.parser.random_flip,
                    pct_rand=params.parser.pct_rand,
                    seed = params.parser.seed,
                    aug_rand_saturation=params.parser.aug_rand_saturation,
                    aug_rand_brightness=params.parser.aug_rand_brightness,
                    aug_rand_zoom=params.parser.aug_rand_zoom,
                    aug_rand_hue=params.parser.aug_rand_hue,
                    anchors = anchors)

        if params.is_training:
            post_process_fn = parser.postprocess_fn()
        else:
          scale_x_y[str(i)] = 1.0
        start += params.boxes_per_scale

      self._masks = boxes
      self._path_scales = path_scales
      self._x_y_scales = scale_x_y

    return self._masks, self._path_scales, self._x_y_scales

  def initialize(self, model: tf.keras.Model):
    if self.task_config.load_darknet_weights:
      from yolo.utils import DarkNetConverter
      from yolo.utils._darknet2tf.load_weights import split_converter
      from yolo.utils._darknet2tf.load_weights2 import load_weights_backbone
      from yolo.utils._darknet2tf.load_weights2 import load_head
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
      model.backbone.trainable = False

      if self.task_config.darknet_load_decoder:
        if neck is not None:
          load_weights_backbone(model.decoder.neck, neck)
          model.decoder.neck.trainable = False
        cfgheads = load_head(model.decoder.head, decoder)
        model.decoder.head.trainable = False
        load_weights_prediction_layers(cfgheads, model.head)
        model.head.trainable = False
    else:
      """Loading pretrained checkpoint."""
      if not self.task_config.init_checkpoint:
        return

      ckpt_dir_or_file = self.task_config.init_checkpoint
      if tf.io.gfile.isdir(ckpt_dir_or_file):
        ckpt_dir_or_file = tf.train.latest_checkpoint(ckpt_dir_or_file)

      # Restoring checkpoint.
      if self.task_config.init_checkpoint_modules == 'all':
        ckpt = tf.train.Checkpoint(**model.checkpoint_items)
        status = ckpt.restore(ckpt_dir_or_file)
        status.assert_consumed()
      elif self.task_config.init_checkpoint_modules == 'backbone':
        ckpt = tf.train.Checkpoint(backbone=model.backbone)
        status = ckpt.restore(ckpt_dir_or_file)
        status.expect_partial().assert_existing_objects_matched()
      else:
        assert "Only 'all' or 'backbone' can be used to initialize the model."

      logging.info('Finished loading pretrained checkpoint from %s',
                   ckpt_dir_or_file)


if __name__ == '__main__':
  import matplotlib.pyplot as plt
  from yolo.utils.run_utils import prep_gpu
  prep_gpu()

  config = exp_cfg.YoloTask(model=exp_cfg.Yolo(base='v3'))
  task = YoloTask(config)
  model = base_task.build_model()
  model.summary()
  base_task.initialize(model)

  train_data = base_task.build_inputs(config.train_data)
  # test_data = base_task.build_inputs(config.task.validation_data)

  for l, (i, j) in enumerate(train_data):
    preds = model(i, training=False)
    boxes = xcycwh_to_yxyx(j['bbox'])

    i = tf.image.draw_bounding_boxes(i, boxes, [[1.0, 0.0, 0.0]])

    i = tf.image.draw_bounding_boxes(i, preds['bbox'], [[0.0, 1.0, 0.0]])
    plt.imshow(i[0].numpy())
    plt.show()

    if l > 2:
      break

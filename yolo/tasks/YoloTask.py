import tensorflow as tf 

import official.core.base_task as task
import official.core.input_reader as dataset


from absl import logging
import tensorflow as tf
from official.core import base_task
from official.core import input_reader
from official.core import task_factory
from official.vision import keras_cv
from official.vision.beta.configs import retinanet as exp_cfg
from official.vision.beta.dataloaders import retinanet_input
from official.vision.beta.dataloaders import tf_example_decoder
from official.vision.beta.dataloaders import tf_example_label_map_decoder
from official.vision.beta.evaluation import coco_evaluator
from official.vision.beta.modeling import factory


@task_factory.register_task_cls(exp_cfg.YoloTask)
class YoloTask(base_task.Task):
  """A single-replica view of training procedure.
  RetinaNet task provides artifacts for training/evalution procedures, including
  loading/iterating over Datasets, initializing the model, calculating the loss,
  post-processing, and customized metrics with reduction.
  """

    def build_model(self):
        """get an instance of Yolo v3 or v4"""
        cfg = self.task_config
        if cfg.version == "v3":
            from yolo.modeling.Yolov3 import Yolov3 as run_model 
        elif cfg.version == "v4":
            from yolo.modeling.Yolov4 import Yolov4 as run_model 
        else:
            raise Exception("unsupported model in build model")
        
        model = run_model(input_shape = cfg.input_shape,
                          model = cfg.type,
                          classes = cfg.classes,
                          backbone = cfg.backbone,
                          head = cfg.head,
                          head_filter = cfg.head_filter,
                          masks = cfg.masks,
                          boxes = cfg.boxes,
                          path_scales = cfg.path_scales,
                          x_y_scales = cfg.x_y_scales,
                          thresh = cfg.thresh,
                          weight_decay = cfg.weight_decay, 
                          class_thresh = dfg.class_thresh,
                          use_nms = cfg.use_nms,
                          using_rt = cfg.using_rt,
                          max_boxes = cfg.max_boxes,
                          scale_boxes = cfg.scale_boxes,
                          scale_mult = cfg.scale_mult,
                          use_tie_breaker = cfg.use_tie_breaker,
                          clip_grads_norm = cfg.clip_grads_norm, 
                          policy=cfg.policy)
        return model

    def initialize(self, model: tf.keras.Model):
        if self.task_config.load_original_weights:
            backbone_weights = self.task_config.backbone_from_darknet
            head_weights = self.task_config.head_from_darknet
            weights_file = self.task_config.backbone_from_darknet

            model.load_weights_from_dn(dn2tf_backbone = backbone_weights, 
                                       dn2tf_head = head_weights, 
                                       weights_file = weights_file)
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

            logging.info('Finished loading pretrained checkpoint from %s', ckpt_dir_or_file)





    def build_inputs(self, params, input_context=None):


    def build_losses(self, outputs, labels, aux_losses=None):


    def build_metrics(self, training=True):


    def train_step(self, inputs, model, optimizer, metrics=None):


    def validation_step(self, inputs, model, metrics=None):


    def aggregate_logs(self, state=None, step_outputs=None):


    def reduce_aggregated_logs(self, aggregated_logs):


class YoloTask(task.Task):
    def __init__(self):
        super().__init__()
        return 

    def initialize(self):
        return 

    def build_model(self, cfg):

        return model
    
    def compile_model(self):
        return 
    
    def build_losses(self):
        return 
        
    def build_metrics(self):
        return 

    def train_step(self):
        return 
    
    def validation_step(self):
        return
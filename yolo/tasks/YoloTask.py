import tensorflow as tf 

import official.core.base_task as task
import official.core.input_reader as dataset


from absl import logging
import tensorflow as tf
from official.core import base_task
from official.core import input_reader
from official.core import task_factory
from official.vision import keras_cv
from yolo.configs import yolo as exp_cfg
# from official.vision.beta.dataloaders import retinanet_input
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
    def __init__(self, params, logging_dir: str = None):
        super().__init__(params, logging_dir)
        self._model_pointer = None
        return


    def build_model(self):
        """get an instance of Yolo v3 or v4"""
        cfg = self.task_config
        if "v3" in cfg.model.type:
            from yolo.modeling.Yolov3 import Yolov3 as run_model 
        elif "v4" in cfg.model.type:
            from yolo.modeling.Yolov4 import Yolov4 as run_model 
        else:
            raise Exception("unsupported model in build model")
        
        task_cfg = cfg.get_build_model_dict()
        model = run_model(**task_cfg)
        model.build(model._input_shape)
        self._model_pointer = model
        return model

    def initialize(self, model: tf.keras.Model):
        if self.task_config.load_original_weights:
            backbone_weights = self.task_config.backbone_from_darknet
            head_weights = self.task_config.head_from_darknet
            weights_file = self.task_config.weights_file

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
        return

    def build_losses(self, outputs, labels, aux_losses=None):

        return

    def build_metrics(self, training=True):
        return

    def train_step(self, inputs, model, optimizer, metrics=None):
        return

    def validation_step(self, inputs, model, metrics=None):
        return

    def aggregate_logs(self, state=None, step_outputs=None):
        return

    def reduce_aggregated_logs(self, aggregated_logs):
        return


if __name__ == "__main__":
    cfg = exp_cfg.YoloTask()

    print(cfg.as_dict())
    # task = YoloTask(exp_cfg.YoloTask())
    # model = task.build_model()
    # model.summary()



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

from yolo.dataloaders import YOLO_Detection_Input
from yolo.dataloaders.decoders import tfds_coco_decoder


@task_factory.register_task_cls(exp_cfg.YoloTask)
class YoloTask(base_task.Task):
    """A single-replica view of training procedure.
    RetinaNet task provides artifacts for training/evalution procedures, including
    loading/iterating over Datasets, initializing the model, calculating the loss,
    post-processing, and customized metrics with reduction.
    """
    def __init__(self, params, logging_dir: str = None):
        super().__init__(params, logging_dir)
        self._loss_dict = None
        return

    def build_model(self):
        """get an instance of Yolo v3 or v4"""
        from yolo.modeling.Yolo import build_yolo

        model_base_cfg = self.task_config.model
        l2_weight_decay = self.task_config.weight_decay

        input_specs = tf.keras.layers.InputSpec(shape=[None] +
                                                model_base_cfg.input_size)
        l2_regularizer = (tf.keras.regularizers.l2(l2_weight_decay)
                          if l2_weight_decay else None)

        model, losses = build_yolo(input_specs, model_base_cfg, l2_regularizer)
        self._loss_dict = losses
        return model

    def initialize(self, model: tf.keras.Model):
        if self.task_config.load_original_weights:
            backbone_weights = self.task_config.backbone_from_darknet
            head_weights = self.task_config.head_from_darknet
            weights_file = self.task_config.weights_file

            model.load_weights_from_dn(dn2tf_backbone=backbone_weights,
                                       dn2tf_head=head_weights,
                                       weights_file=weights_file)
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

        ANCHOR = self.task_config.model.anchors.get()
        anchors = [[float(f) for f in a.split(',')] for a in ANCHOR._boxes]
        parser = YOLO_Detection_Input.Parser(
                    image_w=params.parser.image_w,
                    image_h=params.parser.image_h,
                    num_classes=self.task_config.model.num_classes,
                    fixed_size=params.parser.fixed_size,
                    jitter_im=params.parser.jitter_im,
                    jitter_boxes=params.parser.jitter_boxes,
                    net_down_scale=params.parser.net_down_scale,
                    min_process_size=params.parser.min_process_size,
                    max_process_size=params.parser.max_process_size,
                    max_num_instances = params.parser.max_num_instances,
                    random_flip = params.parser.random_flip,
                    pct_rand=params.parser.pct_rand,
                    seed = params.parser.seed,
                    anchors = anchors)
        reader = input_reader.InputReader(
            params,
            dataset_fn=tf.data.TFRecordDataset,
            decoder_fn=decoder.decode,
            parser_fn=parser.parse_fn(params.is_training))
        dataset = reader.read(input_context=input_context)
        return dataset

    def build_losses(self, outputs, labels, aux_losses=None):
        loss = 0.0
        loss_box = 0.0
        loss_conf = 0.0
        loss_class = 0.0
        metric_dict = dict()

        for key in output.keys():
            _loss, _loss_box, _loss_conf, _loss_class, _avg_iou, _recall50 = self._loss_dict[
                key](labels, outputs[key])
            loss += _loss
            loss_box += _loss_box
            loss_conf += _loss_conf
            loss_class += _loss_class
            metric_dict[f"recall50_{key}"] = tf.stop_gradient(_recall50)
            metric_dict[f"avg_iou_{key}"] = tf.stop_gradient(_avg_iou)

        metric_dict["box_loss"] = loss_box
        metric_dict["conf_loss"] = loss_conf
        metric_dict["class_loss"] = loss_class
        return loss, metric_dict

    def build_metrics(self, training=True):
        return super().build_metrics(training=training)

    def train_step(self, inputs, model, optimizer, metrics=None):
        #get the data point
        image, label = inputs

        num_replicas = tf.distribute.get_strategy().num_replicas_in_sync
        with tf.GradientTape() as tape:
            # compute a prediction
            y_pred = model(image, training=True)
            loss, metrics = self.build_losses(y_pred["raw_output"], label)
            scaled_loss = loss / num_replicas

            # scale the loss for numerical stability
            if isinstance(self.optimizer, mixed_precision.LossScaleOptimizer):
                scaled_loss = self.optimizer.get_scaled_loss(scaled_loss)

        # compute the gradient
        train_vars = model.trainable_variables
        gradients = tape.gradient(scaled_loss, train_vars)

        # get unscaled loss if the scaled_loss was used
        if isinstance(optimizer, mixed_precision.LossScaleOptimizer):
            gradients = optimizer.get_unscaled_gradients(gradients)

        if self.task_config.gradient_clip_norm > 0.0:
            gradients, _ = tf.clip_by_global_norm(
                gradients, self.task_config.gradient_clip_norm)
        optimizer.apply_gradients(zip(gradients, train_vars))

        #custom metrics
        logs = {"loss": loss}
        logs.update(metrics)
        return logs

    def validation_step(self, inputs, model, metrics=None):
        #get the data point
        image, label = inputs

        # computer detivative and apply gradients
        y_pred = model(image, training=False)
        loss, metrics = self.build_losses(y_pred["raw_output"], label)

        #custom metrics
        loss_metrics = {"loss": loss}
        loss_metrics.update(metrics)
        return loss_metrics

    def aggregate_logs(self, state=None, step_outputs=None):
        return super().aggregate_logs(state=state, step_outputs=step_outputs)

    def reduce_aggregated_logs(self, aggregated_logs):
        return super().reduce_aggregated_logsI(aggregated_logs)


if __name__ == "__main__":
    from yolo.configs import yolo as exp_cfg
    import matplotlib.pyplot as plt
    task = YoloTask(exp_cfg.YoloTask())
    ds = task.build_inputs(exp_cfg.YoloTask().train_data)
    print(ds)
    for i, el in enumerate(ds):
        print(el[0][0])
        print(el[0][0].shape)
        plt.imshow(el[0][0].numpy())
        plt.show()
        if i == 10:
            break
import os
import tensorflow as tf
from abc import ABC
from abc import abstractmethod

from tensorflow.keras.mixed_precision import experimental as mixed_precision

class Yolo(tf.keras.Model, ABC):
    @abstractmethod
    def get_default_attributes():
        ...

    @abstractmethod
    def get_summary():
        ...

    @abstractmethod
    def load_weights_from_dn(self,
                             dn2tf_backbone=True,
                             dn2tf_head=False,
                             config_file=None,
                             weights_file=None):
        ...

    def process_datasets(self,
                         train,
                         test,
                         batch_size=1,
                         image_w=416,
                         image_h=416,
                         fixed_size=False,
                         jitter_im=0.1,
                         jitter_boxes=0.005,
                         _eval_is_training = False):

        from yolo.dataloaders.YoloParser import YoloParser
        from yolo.dataloaders.YoloParser import YoloPostProcessing

        parser = YoloParser(image_w=image_w,
                            image_h=image_h,
                            fixed_size=fixed_size,
                            jitter_im=jitter_im,
                            jitter_boxes=jitter_boxes,
                            max_num_instances=self._max_boxes,
                            masks=self._masks,
                            anchors=self._boxes)
        post = YoloPostProcessing(image_w=image_w,image_h=image_h)
        train_parser = parser.parse_fn(is_training= True)
        test_parser = parser.parse_fn(is_training=_eval_is_training)

        train = train.map(train_parser, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        test = test.map(train_parser, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        train = train.batch(batch_size)
        test = test.batch(batch_size)
        
        train = train.map(post.postprocess_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        train = train.prefetch(tf.data.experimental.AUTOTUNE)
        test = test.prefetch(tf.data.experimental.AUTOTUNE)
        return train, test

    def compile(self, optimizer='rmsprop', loss=None, metrics=None, loss_weights=None, weighted_metrics=None, run_eagerly=None, **kwargs):
        super().compile(optimizer=optimizer, loss=loss, metrics=metrics, loss_weights=loss_weights, weighted_metrics=weighted_metrics, run_eagerly=run_eagerly, **kwargs)
        self._loss_fn = loss
        self._loss_weights = loss_weights
        return 
        
    def apply_loss_fn(self, label, y_pred):
        loss = 0.0
        loss_box = 0.0
        loss_conf = 0.0
        loss_class = 0.0
        metric_dict = dict()

        for key in y_pred.keys():
            _loss, _loss_box, _loss_conf, _loss_class, _avg_iou, _recall50 = self._loss_fn[key](label, y_pred[key])
            loss += _loss
            loss_box += _loss_box
            loss_conf += _loss_conf
            loss_class += _loss_class
            metric_dict[f"recall50_{key}"] = tf.stop_gradient(_recall50)
            metric_dict[f"avg_iou_{key}"] =  tf.stop_gradient(_avg_iou)
        
        metric_dict["box_loss"] = loss_box
        metric_dict["conf_loss"] = loss_conf
        metric_dict["class_loss"] = loss_class
        return loss, metric_dict

    def train_step(self, data):
        '''
        for float16 training
        opt = tf.keras.optimizers.SGD(0.25)
        opt = tf.keras.mixed_precision.experimental.LossScaleOptimizer(opt, "dynamic")
        '''
        #get the data point
        image, label = data

        # computer detivative and apply gradients
        num_replicas = tf.distribute.get_strategy().num_replicas_in_sync
        with tf.GradientTape() as tape:
            # compute a prediction
            y_pred = self(image, training=True)
            loss, metrics = self.apply_loss_fn(label, y_pred["raw_output"])
            scaled_loss = loss/num_replicas

            # scale the loss for numerical stability
            if isinstance(self.optimizer, mixed_precision.LossScaleOptimizer):
                scaled_loss = self.optimizer.get_scaled_loss(scaled_loss)

        # compute the gradient
        train_vars = self.trainable_variables
        gradients = tape.gradient(scaled_loss, train_vars)

        # get unscaled loss if the scaled_loss was used
        if isinstance(self.optimizer, mixed_precision.LossScaleOptimizer):
            gradients = self.optimizer.get_unscaled_gradients(gradients)

        gradients, _ = tf.clip_by_global_norm(gradients, 5.0)

        self.optimizer.apply_gradients(zip(gradients, train_vars))
        
        #custom metrics
        loss_metrics = {"loss":loss}
        loss_metrics.update(metrics)
        return loss_metrics

    def test_step(self, data):
        #get the data point
        image, label = data

        # computer detivative and apply gradients
        y_pred = self(image, training=False)
        loss, metrics = self.apply_loss_fn(label, y_pred["raw_output"])

        #custom metrics
        loss_metrics = {"loss":loss}
        loss_metrics.update(metrics)
        return loss_metrics

    def generate_loss(self,
                      ignore_thresh: float = 0.7, 
                      truth_thresh: float = 1.0,
                      loss_type="ciou") -> "Dict[Yolo_Loss]":
        """
        Create loss function instances for each of the detection heads.

        Args:
            scale: the amount by which to scale the anchor boxes that were
                   provided in __init__
        """
        from yolo.modeling.functions.yolo_loss import Yolo_Loss
        loss_dict = {}
        for key in self._masks.keys():
            loss_dict[key] = Yolo_Loss(classes = self._classes,
                                       anchors=self._boxes,
                                       ignore_thresh=ignore_thresh,
                                       truth_thresh=truth_thresh,
                                       loss_type=loss_type,
                                       path_key=key,
                                       mask=self._masks[key],
                                       scale_anchors=self._path_scales[key],
                                       scale_x_y=self._x_y_scales[key],
                                       use_tie_breaker=self._use_tie_breaker)
        self._loss_fn = loss_dict
        return loss_dict

    def match_optimizer_to_policy(self, optimizer, scaling = "dynamic"):
        if self._policy != "float32":
            return tf.keras.mixed_precision.experimental.LossScaleOptimizer(optimizer, scaling)
        return optimizer

    def set_policy(self,
                   policy='mixed_float16',
                   save_weights_temp_name="abn7lyjptnzuj918"):
        print(f"setting policy: {policy}")
        if self._policy == policy:
            return
        else:
            self._policy = policy
        policy = mixed_precision.Policy(self._policy)
        mixed_precision.set_policy(policy)
        dtype = policy.compute_dtype

        # save weights and and rebuild model, then load the weights if the model is built
        if self._built:
            self.save_weights(save_weights_temp_name)
            self.build(input_shape=self._input_shape)
            self.load_weights(save_weights_temp_name)
            os.system(f"rm {save_weights_temp_name}.*")
        return

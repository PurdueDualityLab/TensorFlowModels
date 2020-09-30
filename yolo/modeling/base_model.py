import os
import tensorflow as tf
from abc import ABC
from abc import abstractmethod

from tensorflow.keras.mixed_precision import experimental as mixed_precision

class Yolo(tf.keras.Model, ABC):
    @abstractmethod
    def get_models():
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

    # @abstractmethod
    # def train_step(self, data):
    #     ...

    # @abstractmethod
    # def test_step(self):
    #     ...

    # @abstractmethod
    # def process_datasets(self):
    #     ...

    def process_datasets(self,
                         train,
                         test,
                         batch_size=1,
                         image_w=416,
                         image_h=416,
                         fixed_size=False,
                         jitter_im=0.1,
                         jitter_boxes=0.005):

        from yolo.dataloaders.YoloParser import YoloParser
        parser = YoloParser(image_w=image_w,
                            image_h=image_h,
                            fixed_size=fixed_size,
                            jitter_im=jitter_im,
                            jitter_boxes=jitter_boxes,
                            masks=self._masks,
                            anchors=self._boxes)

        preprocess_train = parser.unbatched_process_fn(is_training=True)
        postprocess_train = parser.batched_process_fn(is_training=True)

        preprocess_test = parser.unbatched_process_fn(is_training=False)
        postprocess_test = parser.batched_process_fn(is_training=False)

        train = train.map(preprocess_train).padded_batch(batch_size)
        train = train.map(postprocess_train)
        test = test.map(preprocess_test).padded_batch(batch_size)
        test = test.map(postprocess_test)

        train_size = tf.data.experimental.cardinality(train)
        test_size = tf.data.experimental.cardinality(test)
        print(train_size, test_size)

        return train, test

    def train_step(self, data):
        '''
        for float16 training
        opt = tf.keras.optimizers.SGD(0.25)
        opt = tf.keras.mixed_precision.experimental.LossScaleOptimizer(opt, "dynamic")
        '''
        #get the data point
        image = data["image"]
        label = data["label"]

        # computer detivative and apply gradients
        num_replicas = tf.distribute.get_strategy().num_replicas_in_sync
        with tf.GradientTape() as tape:
            # compute a prediction
            y_pred = self(image, training=True)
            loss = self.compiled_loss(label, y_pred["raw_output"])
            scaled_loss = loss/num_replicas

            # scale the loss for numerical stability
            if isinstance(self.optimizer, mixed_precision.LossScaleOptimizer):
                scaled_loss = self.optimizer.get_scaled_loss(scaled_loss)

        # compute the gradient
        train_vars = self.trainable_variables
        gradients = tape.gradient(scaled_loss, train_vars)

        # get unscaled loss if the scaled_loss was used
        if isinstance(self.optimizer, mixed_precision.LossScaleOptimizer):
            gradients = self.optimizer.get_unscaled_gradients(grads)

        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        #custom metrics
        loss_metrics = dict()
        for loss in self.compiled_loss._losses:
            loss_metrics[f"{loss._path_key}_boxes"] = loss.get_box_loss()
            loss_metrics[
                f"{loss._path_key}_classes"] = loss.get_classification_loss()
            loss_metrics[f"{loss._path_key}_avg_iou"] = loss.get_avg_iou()
            loss_metrics[
                f"{loss._path_key}_confidence"] = loss.get_confidence_loss()

        #compiled metrics
        self.compiled_metrics.update_state(label, y_pred["raw_output"])
        metrics_dict = {m.name: m.result() for m in self.metrics}
        metrics_dict.update(loss_metrics)
        return metrics_dict

    def test_step(self, data):
        #get the data point
        image = data["image"]
        label = data["label"]

        # computer detivative and apply gradients
        y_pred = self(image, training=False)
        loss = self.compiled_loss(label, y_pred["raw_output"])

        #custom metrics
        loss_metrics = dict()
        for loss in self.compiled_loss._losses:
            loss_metrics[f"{loss._path_key}_boxes"] = loss.get_box_loss()
            loss_metrics[
                f"{loss._path_key}_classes"] = loss.get_classification_loss()
            loss_metrics[f"{loss._path_key}_avg_iou"] = loss.get_avg_iou()
            loss_metrics[
                f"{loss._path_key}_confidence"] = loss.get_confidence_loss()

        #compiled metrics
        self.compiled_metrics.update_state(label, y_pred["raw_output"])
        metrics_dict = {m.name: m.result() for m in self.metrics}
        metrics_dict.update(loss_metrics)
        return metrics_dict

    def generate_loss(self,
                      scale: float = 1.0,
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
            loss_dict[key] = Yolo_Loss(mask=self._masks[key],
                                       anchors=self._boxes,
                                       scale_anchors=self._path_scales[key],
                                       ignore_thresh=0.7,
                                       truth_thresh=1,
                                       loss_type=loss_type,
                                       path_key=key,
                                       scale_x_y=self._x_y_scales[key],
                                       use_tie_breaker=self._use_tie_breaker)
        return loss_dict

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

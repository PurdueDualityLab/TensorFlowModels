import os
import tensorflow as tf
from abc import ABC, abstractmethod


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

    @abstractmethod
    def train_step(self, data):
        ...

    @abstractmethod
    def test_step(self):
        ...

    @abstractmethod
    def process_datasets(self):
        ...

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
        from tensorflow.keras.mixed_precision import experimental as mixed_precision
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

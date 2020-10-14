import tensorflow as tf
import tensorflow.keras as ks
from typing import *

from yolo.modeling.backbones.Darknet import Darknet
from yolo.modeling.model_heads._Yolov4Head import Yolov4Head
from yolo.modeling.model_heads._Yolov3Head import Yolov3Head
from yolo.modeling.model_heads._Yolov4Neck import Yolov4Neck
from yolo.modeling.building_blocks import YoloLayer


from yolo.utils import DarkNetConverter
from yolo.utils.file_manager import download
from yolo.utils._darknet2tf.load_weights import split_converter
from yolo.utils._darknet2tf.load_weights2 import load_weights_backbone
from yolo.utils._darknet2tf.load_weights2 import load_weights_v4head
from yolo.utils._darknet2tf.load_weights import load_weights_dnBackbone
from yolo.utils._darknet2tf.load_weights import load_weights_dnHead

class Yolo(ks.Model):
    def __init__(
            self,
            classes=80,
            backbone = None,
            neck = None,
            head = None,
            decoder = None,
            weight_decay = 0.005,
            clip_grads_norm = 0.0,
            **kwargs):
        super().__init__(**kwargs)

        #required inputs
        self._input_shape = [None] + input_shape
        self._classes = classes
        self._backbone_neck_split = None
        self._neck_head_split = None

        #init base params
        self._weight_decay = weight_decay
        self._clip_grads_norm = clip_grads_norm
        self._loss_fn = None

        #model components
        self._backbone = backbone
        self._neck = neck
        self._head = head
        self._decoder = decoder
        return

    @property
    def backbone(self):
        return self._backbone
    
    @property
    def neck(self):
        return self._neck
    
    @property
    def head(self):
        return self._head
    
    @property
    def decoder(self):
        return self._decoder

    def set_loss_fn_dict(self, loss_fn):
        self._loss_fn = loss_fn
        return 

    def build(self, input_shape):
        self._backbone.build(input_shape)
        nshape = self._backbone.output_shape
        if self._neck != None:
            self.neck.build(nshape)
            nshape = self._neck.output_shape
        self._head.build(nshape)
        super().build(input_shape)
        return 

    def call(self, inputs, training=False):
        maps = self._backbone(inputs)

        if self._neck != None:
            maps = self._neck(maps)

        raw_head = self._head(maps)
        if training:
            return {"raw_output": raw_head}
        else:
            predictions = self._head_filter(raw_head)
            predictions.update({"raw_output": raw_head})
            return predictions

    def load_weights_from_dn(self,
                             dn2tf_backbone=True,
                             dn2tf_head=True,
                             config_file=None,
                             weights_file=None):
        """
        load the entire Yolov3 Model for tensorflow

        example:
            load yolo with darknet wieghts for backbone
            model = Yolov3()
            model.build(input_shape = (1, 416, 416, 3))
            model.load_weights_from_dn(dn2tf_backbone = True, dn2tf_head = True)

        to be implemented
        example:
            load custom back bone weigths

        example:
            load custom head weigths

        example:
            load back bone weigths from tensorflow (our training)

        example:
            load head weigths from tensorflow (our training)

        Args:
            dn2tf_backbone: bool, if true it will load backbone weights for yolo v3 from darknet .weights file
            dn2tf_head: bool, if true it will load head weights for yolo v3 from darknet .weights file
            config_file: str path for the location of the configuration file to use when decoding darknet weights
            weights_file: str path with the file containing the dark net weights
        """
        if not self.built:
            self.build(self._input_shape)

        if dn2tf_backbone or dn2tf_head:
            if config_file is None:
                config_file = download(self._model_name + '.cfg')
            if weights_file is None:
                weights_file = download(self._model_name + '.weights')
            list_encdec = DarkNetConverter.read(config_file, weights_file)
            encoder, neck, decoder = split_converter(
                list_encdec, self._encoder_decoder_split_location, 138)

        if dn2tf_backbone:
            #load_weights_dnBackbone(self._backbone, encoder, mtype = self._backbone_name)
            load_weights_backbone(self._backbone, encoder)
            self._backbone.trainable = False

        if dn2tf_head:
            load_weights_backbone(self._neck, neck)
            self._neck.trainable = False
            load_weights_v4head(self._head, decoder)
            self._head.trainable = False
        return

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

        if self._clip_grads_norm > 0.0: 
            gradients, _ = tf.clip_by_global_norm(gradients, self._clip_grads_norm)

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

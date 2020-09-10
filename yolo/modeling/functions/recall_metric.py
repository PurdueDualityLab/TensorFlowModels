import tensorflow as tf
import tensorflow.keras as ks
from tensorflow.keras import backend as K
import tensorflow_datasets as tfds
from yolo.modeling.yolo_v3 import Yolov3
from yolo.modeling.functions.iou import *


class YoloMAP_recall(ks.metrics.Metric):
    def __init__(self, threshold = 0.5, num = 3, name = "recall", **kwargs):
        super().__init__(name=f"{name}", **kwargs)
        self._thresh = threshold
        
        self._num = num
        self._value = self.add_weight(name = "total_recall", initializer='zeros')
        self._count = self.add_weight(name = "total_samples", initializer='zeros')
        return

    def update_state(self, y_true, y_pred, sample_weight = None):
        y_pred = tf.reshape(y_pred, [tf.shape(y_pred)[0], tf.shape(y_pred)[1], tf.shape(y_pred)[2], self._num , -1])
        pred_conf = tf.expand_dims(tf.math.sigmoid(y_pred[..., 4]), axis = -1)
        true_conf = tf.expand_dims(y_true[..., 4], axis = -1)

        value = tf.reduce_sum(tf.cast(pred_conf > self._thresh, dtype = self.dtype) * true_conf, axis = -1)/tf.reduce_sum(true_conf + 1e-16)
        self._value.assign_add(tf.reduce_sum(value)/tf.cast(tf.shape(y_pred)[0], dtype = tf.float32))
        self._count.assign_add(tf.cast(1.0, dtype = tf.float32))
        return
    
    def result(self):
        return self._value/self._count

class YoloMAP(ks.metrics.Metric):
    def __init__(self, threshold = 0.5, num = 3, name = "recall", **kwargs):
        super().__init__(name=f"{name}", **kwargs)
        self._thresh = threshold
        
        self._num = num
        self._value = self.add_weight(name = "total_recall", initializer='zeros')
        return

    def update_state(self, y_true, y_pred, sample_weight = None):
        y_pred = tf.reshape(y_pred, [tf.shape(y_pred)[0], tf.shape(y_pred)[1], tf.shape(y_pred)[2], self._num , -1])
        pred_conf = tf.expand_dims(tf.math.sigmoid(y_pred[..., 4]), axis = -1)
        true_conf = tf.expand_dims(y_true[..., 4], axis = -1)

        value = tf.reduce_sum(tf.cast(pred_conf > self._thresh, dtype = self.dtype) * true_conf, axis = -1)/tf.reduce_sum(true_conf + 1e-16)
        value = tf.reshape(value, [tf.shape(y_pred)[0], -1])
        value = tf.reduce_sum(value, axis = -1)
        self._value.assign(tf.reduce_mean(value))
        return
    
    def result(self):
        return self._value

class YoloMAP_recall75(ks.metrics.Metric):
    def __init__(self, threshold = 0.75, num = 3, name = "recall", **kwargs):
        super().__init__(name=f"{name}", **kwargs)
        self._thresh = threshold
        
        self._num = num
        self._value = self.add_weight(name = "total_recall", initializer='zeros')
        self._count = self.add_weight(name = "total_samples", initializer='zeros')
        return

    def update_state(self, y_true, y_pred, sample_weight = None):
        y_pred = tf.reshape(y_pred, [tf.shape(y_pred)[0], tf.shape(y_pred)[1], tf.shape(y_pred)[2], self._num , -1])
        pred_conf = tf.expand_dims(tf.math.sigmoid(y_pred[..., 4]), axis = -1)
        true_conf = tf.expand_dims(y_true[..., 4], axis = -1)

        value = tf.reduce_sum(tf.cast(pred_conf > self._thresh, dtype = self.dtype) * true_conf, axis = -1)/tf.reduce_sum(true_conf + 1e-16)
        self._value.assign_add(tf.reduce_sum(value))
        self._count.assign_add(tf.cast(tf.shape(y_pred)[0], dtype = tf.float32))
        return
    
    def result(self):
        return self._value/self._count

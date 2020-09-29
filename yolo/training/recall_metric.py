import tensorflow as tf
import tensorflow.keras as ks
from tensorflow.keras import backend as K
from yolo.utils.iou_utils import *


class YoloMAP_recall(ks.metrics.Metric):
    def __init__(self, threshold=0.45, num=3, name="recall", **kwargs):
        super().__init__(name=f"{name}", **kwargs)
        self._thresh = threshold

        self._num = num
        self._value = self.add_weight(name="total_recall", initializer='zeros')
        self._count = self.add_weight(name="total_samples",
                                      initializer='zeros')
        return

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.reshape(y_pred, [
            tf.shape(y_pred)[0],
            tf.shape(y_pred)[1],
            tf.shape(y_pred)[2], self._num, -1
        ])
        pred_conf = tf.expand_dims(tf.math.sigmoid(y_pred[..., 4]), axis=-1)
        true_conf = tf.expand_dims(y_true[..., 4], axis=-1)

        value = tf.reduce_sum(
            tf.cast(pred_conf > self._thresh, dtype=self.dtype) * true_conf,
            axis=(1, 2,
                  3)) / (tf.reduce_sum(true_conf, axis=(1, 2, 3)) + 1e-16)
        self._value.assign_add(tf.reduce_mean(value))
        self._count.assign_add(tf.cast(1.0, dtype=tf.float32))
        return

    def result(self):
        return self._value / self._count


class YoloMAP(ks.metrics.Metric):
    def __init__(self, threshold=0.5, num=3, name="recall", **kwargs):
        super().__init__(name=f"{name}", **kwargs)
        self._thresh = threshold
        self._num = num
        self._value = self.add_weight(name="total_recall", initializer='zeros')
        return

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.reshape(y_pred, [
            tf.shape(y_pred)[0],
            tf.shape(y_pred)[1],
            tf.shape(y_pred)[2], self._num, -1
        ])
        pred_conf = tf.expand_dims(tf.math.sigmoid(y_pred[..., 4]), axis=-1)
        true_conf = tf.expand_dims(y_true[..., 4], axis=-1)

        value = tf.reduce_sum(
            tf.cast(pred_conf > self._thresh, dtype=self.dtype) * true_conf,
            axis=(1, 2,
                  3)) / (tf.reduce_sum(true_conf, axis=(1, 2, 3)) + 1e-16)
        self._value.assign(tf.reduce_mean(value))
        return

    def result(self):
        return self._value


class YoloMAP_recall75(ks.metrics.Metric):
    def __init__(self, threshold=0.75, num=3, name="recall", **kwargs):
        super().__init__(name=f"{name}", **kwargs)
        self._thresh = threshold

        self._num = num
        self._value = self.add_weight(name="total_recall", initializer='zeros')
        self._count = self.add_weight(name="total_samples",
                                      initializer='zeros')
        return

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.reshape(y_pred, [
            tf.shape(y_pred)[0],
            tf.shape(y_pred)[1],
            tf.shape(y_pred)[2], self._num, -1
        ])
        pred_conf = tf.expand_dims(tf.math.sigmoid(y_pred[..., 4]), axis=-1)
        true_conf = tf.expand_dims(y_true[..., 4], axis=-1)

        value = tf.reduce_sum(
            tf.cast(pred_conf > self._thresh, dtype=self.dtype) * true_conf,
            axis=-1) / tf.reduce_sum(true_conf + 1e-16)
        self._value.assign_add(tf.reduce_sum(value))
        self._count.assign_add(tf.cast(tf.shape(y_pred)[0], dtype=tf.float32))
        return

    def result(self):
        return self._value / self._count


class YoloIOU_recall(ks.metrics.Metric):
    def __init__(self,
                 threshold=0.75,
                 num=3,
                 anchors=[(10, 13), (16, 30), (33, 23)],
                 name="recall",
                 **kwargs):
        super().__init__(name=f"{name}", **kwargs)
        self._thresh = threshold

        self._anchors = anchors
        self._num = num
        self._value = self.add_weight(name="total_recall", initializer='zeros')
        self._count = self.add_weight(name="total_samples",
                                      initializer='zeros')
        return

    @tf.function
    def _get_anchor_grid(self, width, height, batch_size):
        """ get the transformed anchor boxes for each dimention """
        anchors = tf.cast(self._anchors, dtype=self.dtype) / tf.cast(
            416, dtype=tf.float32)
        anchors = tf.reshape(anchors, [1, -1])
        anchors = tf.repeat(anchors, width * height, axis=0)
        anchors = tf.reshape(anchors, [1, width, height, self._num, -1])
        anchors = tf.repeat(anchors, batch_size, axis=0)
        return anchors

    @tf.function
    def _get_centers(self, lwidth, lheight, batch_size):
        """ generate a grid that is used to detemine the relative centers of the bounding boxs """
        x_left, y_left = tf.meshgrid(tf.range(0, lwidth), tf.range(0, lheight))
        x_y = K.stack([x_left, y_left], axis=-1)
        x_y = tf.cast(x_y, dtype=self.dtype) / tf.cast(lwidth,
                                                       dtype=self.dtype)
        x_y = tf.repeat(tf.expand_dims(tf.repeat(tf.expand_dims(x_y, axis=-2),
                                                 self._num,
                                                 axis=-2),
                                       axis=0),
                        batch_size,
                        axis=0)
        return x_y

    def update_state(self, y_true, y_pred, sample_weight=None):
        batch_size = tf.cast(tf.shape(y_pred)[0], dtype=tf.int32)
        width = tf.cast(tf.shape(y_pred)[1], dtype=tf.int32)
        height = tf.cast(tf.shape(y_pred)[2], dtype=tf.int32)
        grid_points = self._get_centers(width, height, batch_size)
        anchor_grid = self._get_anchor_grid(width, height, batch_size)

        y_pred = tf.reshape(y_pred, [batch_size, width, height, self._num, -1])
        y_pred = tf.cast(y_pred, dtype=self.dtype)

        fwidth = tf.cast(width, tf.float32)
        fheight = tf.cast(height, tf.float32)

        pred_xy = tf.math.sigmoid(y_pred[..., 0:2])
        pred_wh = y_pred[..., 2:4]

        box_xy = pred_xy / fwidth + grid_points
        box_wh = tf.math.exp(pred_wh) * anchor_grid
        pred_box = K.concatenate([box_xy, box_wh], axis=-1)
        true_box = y_true[..., 0:4]

        iou = tf.nn.relu(box_iou(true_box, pred_box, dtype=self.dtype))
        value = tf.reduce_sum(iou) / tf.cast(tf.math.count_nonzero(iou),
                                             dtype=tf.float32)
        self._value.assign_add(value)
        self._count.assign_add(tf.cast(1, dtype=tf.float32))
        return

    def result(self):
        return self._value / self._count


class YoloClass_recall(ks.metrics.Metric):
    def __init__(self, num=3, name="recall", **kwargs):
        super().__init__(name=f"{name}", **kwargs)

        self._num = num
        self._value = self.add_weight(name="total_recall", initializer='zeros')
        self._count = self.add_weight(name="total_samples",
                                      initializer='zeros')
        return

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.reshape(y_pred, [
            tf.shape(y_pred)[0],
            tf.shape(y_pred)[1],
            tf.shape(y_pred)[2], self._num, -1
        ])
        true_conf = y_true[..., 4]
        pred_class = y_pred[..., 5:][true_conf > 0]
        true_class = y_true[..., 5:][true_conf > 0]
        value = ks.metrics.categorical_accuracy(pred_class, true_class)
        self._value.assign_add(tf.reduce_sum(value))
        self._count.assign_add(tf.cast(tf.shape(value)[0], dtype=self.dtype))

    def result(self):
        return self._value / self._count

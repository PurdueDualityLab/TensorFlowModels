"""Contains common building blocks for yolo neural networks."""
import tensorflow as tf
import tensorflow.keras as ks
import tensorflow.keras.backend as K

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys

from yolo.utils import loss_utils 
from yolo.utils import box_utils
from yolo.modeling.functions.yolo_loss import Yolo_Loss


@ks.utils.register_keras_serializable(package='yolo')
class YoloLayer(ks.Model):
    def __init__(self,
                 masks,
                 anchors,
                 classes, 
                 thresh = 0.5,
                 cls_thresh = 0.5,
                 ignore_thresh = 0.7, 
                 loss_type = "ciou", 
                 use_tie_breaker = True,
                 max_boxes = 200,
                 path_scale = None,
                 scale_xy = None,
                 use_nms = True,
                 **kwargs):
        super().__init__(**kwargs)
        self._masks = masks
        self._anchors = anchors
        self._thresh = thresh
        self._cls_thresh = cls_thresh
        self._ignore_thresh = ignore_thresh
        self._max_boxes = max_boxes
        self._classes = classes
        self._loss_type = loss_type
        self._use_tie_breaker = use_tie_breaker
        self._keys = list(masks.keys())
        self._len_keys = len(self._keys)
        self._path_scale = path_scale
        self._use_nms = use_nms
        self._scale_xy = scale_xy or {key: 1.0 for key, _ in masks.items()}
        self._generator = {}
        self._len_mask = {}
        for i, key in enumerate(self._keys):
            anchors = [self._anchors[mask] for mask in self._masks[key]]
            self._generator[key] = self.get_generators(anchors,
                                                       self._path_scale[key],
                                                       key)
            self._len_mask[key] = len(self._masks[key])
        return

    def get_generators(self, anchors, path_scale, path_key):
        anchor_generator = loss_utils.GridGenerator(anchors,
                                         scale_anchors=path_scale,
                                         low_memory=True,
                                         name=f"yolo_layer_{path_key}",
                                         reset=True)
        return anchor_generator

    def parse_prediction_path(self, generator, len_mask, scale_xy, inputs):
        shape = tf.shape(inputs)
        #reshape the yolo output to (batchsize, width, height, number_anchors, remaining_points)
        data = tf.reshape(inputs, [shape[0], shape[1], shape[2], len_mask, -1])
        centers, anchors = generator(shape[1],
                                     shape[2],
                                     shape[0],
                                     dtype=data.dtype)

        # compute the true box output values
        _, _, boxes = loss_utils.parse_yolo_box_predictions(data[..., 0:4],
                                                 tf.cast(shape[1], data.dtype),
                                                 tf.cast(shape[2], data.dtype),
                                                 anchors,
                                                 centers,
                                                 scale_x_y=scale_xy)
        box = box_utils.xcycwh_to_yxyx(boxes)

        # computer objectness and generate grid cell mask for where objects are located in the image
        objectness = tf.expand_dims(tf.math.sigmoid(data[..., 4]), axis=-1)
        scaled = tf.math.sigmoid(data[..., 5:]) * objectness

        #compute the mask of where objects have been located
        mask = tf.reduce_any(objectness > self._thresh, axis=-1)
        mask = tf.reduce_any(mask, axis=0)

        # reduce the dimentions of the box predictions to (batch size, max predictions, 4)
        box = tf.boolean_mask(box, mask, axis=1)[:, :200, :]
        # reduce the dimentions of the box predictions to (batch size, max predictions, classes)
        classifications = tf.boolean_mask(scaled, mask, axis=1)[:, :200, :]
        return box, classifications

    def call(self, inputs):
        boxes, classifs = self.parse_prediction_path(
            self._generator[self._keys[0]], self._len_mask[self._keys[0]],
            self._scale_xy[self._keys[0]], inputs[str(self._keys[0])])
        i = 1
        while i < self._len_keys:
            key = self._keys[i]
            b, c = self.parse_prediction_path(self._generator[key],
                                              self._len_mask[key],
                                              self._scale_xy[key], inputs[str(key)])
            boxes = K.concatenate([boxes, b], axis=1)
            classifs = K.concatenate([classifs, c], axis=1)
            i += 1

        if self._use_nms:
            boxes = tf.cast(boxes, dtype=tf.float32)
            classifs = tf.cast(classifs, dtype=tf.float32)
            nms = tf.image.combined_non_max_suppression(
                tf.expand_dims(boxes, axis=2), classifs, self._max_boxes,
                self._max_boxes, self._thresh, self._cls_thresh)
            return {
                "bbox": nms.nmsed_boxes,
                "classes": nms.nmsed_classes,
                "confidence": nms.nmsed_scores,
            }
        else:
            return {
                "bbox": boxes,
                "classes": tf.math.argmax(classifs, axis=-1),
                "confidence":
                classifs,  #tf.math.reduce_max(classifs, axis = -1),
            }

    @property
    def losses(self):
        loss_dict = {}
        for key in self._keys:
            loss_dict[key] = Yolo_Loss(
                classes=self._classes,
                anchors=self._anchors,
                ignore_thresh=self._ignore_thresh,
                loss_type=self._loss_type,
                path_key=key,
                mask=self._masks[key],
                scale_anchors=self._path_scale[key],
                scale_x_y=self._scale_xy[key],
                use_tie_breaker=self._use_tie_breaker)
        return loss_dict


    def get_config(self):
        return {
            "masks": dict(self._masks),
            "anchors": [list(a) for a in self._anchors],
            "thresh": self._thresh,
            "cls_thresh": self._cls_thresh,
            "max_boxes": self._max_boxes,
        }


if __name__ == "__main__":
    x = tf.ones(shape=(1, 416, 416, 3))
    model = build_model()
    y = model(x)
    print(y)

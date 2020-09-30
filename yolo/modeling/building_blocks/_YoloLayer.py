"""Contains common building blocks for yolo neural networks."""
import tensorflow as tf
import tensorflow.keras as ks
import tensorflow.keras.backend as K

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys

from yolo.utils.loss_utils import GridGenerator


@ks.utils.register_keras_serializable(package='yolo')
class YoloFilterCell(ks.layers.Layer):
    def __init__(self,
                 anchors,
                 thresh,
                 path_scale,
                 path_key=None,
                 max_box=200,
                 **kwargs):
        super().__init__(**kwargs)
        self._mask_len = len(anchors)
        self._anchors = tf.cast(tf.convert_to_tensor(anchors),
                                dtype=self.dtype)
        self._thresh = thresh
        self._path_scale = path_scale

        self._anchor_generator = GridGenerator.get_generator_from_key(path_key)
        #tf.print(path_key)
        if self._anchor_generator == None:
            self._anchor_generator = GridGenerator(self._anchors,
                                                   scale_anchors=path_scale,
                                                   low_memory=False,
                                                   name=path_key)
        return

    def call(self, inputs):
        shape = tf.shape(inputs)
        #reshape the yolo output to (batchsize, width, height, number_anchors, remaining_points)
        data = tf.reshape(inputs,
                          [shape[0], shape[1], shape[2], self._mask_len, -1])
        centers, anchors = self._anchor_generator(shape[1],
                                                  shape[2],
                                                  shape[0],
                                                  dtype=data.dtype)

        # compute the true box output values
        box_xy = centers + (tf.math.sigmoid(data[..., 0:2])) / tf.cast(
            shape[1], dtype=data.dtype)
        box_wh = tf.math.exp(data[..., 2:4]) * anchors

        # convert the box to Tensorflow Expected format
        minpoint = box_xy - box_wh / 2
        maxpoint = box_xy + box_wh / 2
        box = K.stack([
            minpoint[..., 1], minpoint[..., 0], maxpoint[..., 1], maxpoint[...,
                                                                           0]
        ],
                      axis=-1)

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


@ks.utils.register_keras_serializable(package='yolo')
class YoloLayer(ks.Model):
    def __init__(self,
                 masks,
                 anchors,
                 thresh,
                 cls_thresh,
                 max_boxes,
                 scale_boxes=1,
                 scale_mult=1,
                 path_scale=None,
                 **kwargs):
        super().__init__(**kwargs)
        self._masks = masks
        self._anchors = anchors
        self._scale_mult = scale_mult
        self._thresh = thresh
        self._cls_thresh = cls_thresh
        self._max_boxes = max_boxes
        self._keys = list(masks.keys())
        self._len_keys = len(self._keys)
        self._path_scale = path_scale
        return

    def build(self, input_shape):
        if list(input_shape.keys()) != self._keys and list(
                reversed(input_shape.keys())) != self._keys:
            raise Exception(
                f"input size does not match the layers initialization, {self._keys} != {list(input_shape.keys())}"
            )

        self._filters = {}
        for i, key in enumerate(self._keys):
            anchors = [self._anchors[mask] for mask in self._masks[key]]
            self._filters[key] = YoloFilterCell(
                anchors=anchors,
                thresh=self._thresh,
                path_key=key,
                max_box=self._max_boxes,
                path_scale=self._path_scale[key])
        return

    def call(self, inputs):
        boxes, classifs = self._filters[self._keys[0]](inputs[self._keys[0]])
        i = 1
        while i < self._len_keys:
            key = self._keys[i]
            b, c = self._filters[key](inputs[key])
            boxes = K.concatenate([boxes, b], axis=1)
            classifs = K.concatenate([classifs, c], axis=1)
            i += 1

        boxes = tf.cast(boxes, dtype=tf.float32)
        classifs = tf.cast(classifs, dtype=tf.float32)
        nms = tf.image.combined_non_max_suppression(
            tf.expand_dims(boxes, axis=2), classifs, self._max_boxes,
            self._max_boxes, self._thresh, self._cls_thresh)
        return {
            "bbox": nms.nmsed_boxes,
            "classes": nms.nmsed_classes,
            "confidence": nms.nmsed_scores,
            "raw_output": inputs
        }

    def get_config(self):
        return {
            "masks": dict(self._masks),
            "anchors": [list(a) for a in self._anchors],
            "thresh": self._thresh,
            "cls_thresh": self._cls_thresh,
            "max_boxes": self._max_boxes,
            "scale_mult": self._scale_mult,
        }


@ks.utils.register_keras_serializable(package='yolo')
class YoloGT(ks.layers.Layer):
    def __init__(self,
                 anchors,
                 thresh,
                 max_box=200,
                 dtype=tf.float32,
                 reshape=True,
                 **kwargs):
        self._mask_len = len(anchors)
        self._dtype = dtype
        self._anchors = tf.cast(tf.convert_to_tensor(anchors),
                                dtype=self._dtype) / 416
        self._thresh = tf.cast(thresh, dtype=self._dtype)

        self._rebuild = True
        self._rebatch = True
        self._reshape = reshape

        super().__init__(**kwargs)
        return

    def call(self, inputs):
        shape = tf.shape(inputs)
        data = inputs
        data = tf.cast(data, self._dtype)

        # compute the true box output values
        box_xy = data[..., 0:2]
        box_wh = data[..., 2:4]

        # convert the box to Tensorflow Expected format
        minpoint = box_xy - box_wh / 2
        maxpoint = box_xy + box_wh / 2
        box = K.stack([
            minpoint[..., 1], minpoint[..., 0], maxpoint[..., 1], maxpoint[...,
                                                                           0]
        ],
                      axis=-1)

        # computer objectness and generate grid cell mask for where objects are located in the image
        objectness = tf.expand_dims(data[..., 4], axis=-1)
        scaled = data[..., 5:]
        #scaled = classes * objectness

        mask = tf.reduce_any(objectness > tf.cast(0.0, dtype=self._dtype),
                             axis=-1)
        mask = tf.reduce_any(mask, axis=0)

        # reduce the dimentions of the box predictions to (batch size, max predictions, 4)
        box = tf.boolean_mask(box, mask, axis=1)[:, :200, :]

        # # reduce the dimentions of the box predictions to (batch size, max predictions, classes)
        classifications = tf.boolean_mask(scaled, mask, axis=1)[:, :200, :]
        return box, classifications


if __name__ == "__main__":
    x = tf.ones(shape=(1, 416, 416, 3))
    model = build_model()
    y = model(x)
    print(y)

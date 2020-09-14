"""Contains common building blocks for yolo neural networks."""
import tensorflow as tf
import tensorflow.keras as ks
import tensorflow.keras.backend as K

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys

#@ks.utils.register_keras_serializable(package='yolo')
class YoloFilterCell(ks.layers.Layer):
    def __init__(self, anchors, thresh, max_box = 200, dtype = tf.float32, **kwargs):
        super().__init__(**kwargs)
        self._mask_len = len(anchors)
        self._anchors = tf.cast(tf.convert_to_tensor(anchors), dtype = self.dtype)
        self._thresh = tf.cast(thresh, dtype = self.dtype)

        self._rebuild = True
        self._rebatch = True
        return

    def _reshape_batch(self, value, batch_size, axis = 0):
        return tf.repeat(value, batch_size, axis = axis)
    
    def _get_centers(self, lwidth, lheight, num):
        """ generate a grid that is used to detemine the relative centers of the bounding boxs """
        x_left, y_left = tf.meshgrid(tf.range(0, lheight), tf.range(0, lwidth))
        x_y = K.stack([x_left, y_left], axis = -1)
        x_y = tf.cast(x_y, dtype = self.dtype)
        x_y = tf.expand_dims(tf.repeat(tf.expand_dims(x_y, axis = -2), num, axis = -2), axis = 0)
        return x_y

    def _get_anchor_grid(self, width, height, num, anchors):
        """ get the transformed anchor boxes for each dimention """
        anchors = tf.cast(anchors, dtype = self.dtype)
        anchors = tf.reshape(anchors, [1, -1])
        anchors = tf.repeat(anchors, width*height, axis = 0)
        anchors = tf.reshape(anchors, [1, width, height, num, -1])
        return anchors

    def build(self, input_shape):
        self._input_shape = input_shape
        #width or height is None 
        if self._input_shape[1] != None and self._input_shape[2] != None:
            self._rebuild = False 
        
        # if the batch size is not None
        if not self._rebuild and self._input_shape[0] != None:
            self._rebatch = False

        if not self._rebuild: 
            _, width, height, _ = input_shape
            self._anchor_matrix = self._get_anchor_grid(width, height, len(self._anchors), self._anchors)
            self._grid_cells = self._get_centers(width, height, len(self._anchors))

        if not self._rebatch:
            self._anchor_matrix = self._reshape_batch(self._anchor_matrix, input_shape[0])
            self._grid_cells = self._reshape_batch(self._grid_cells, input_shape[0])

        super().build(input_shape)
        return 

    def call(self, inputs):
        shape = tf.shape(inputs)
        #reshape the yolo output to (batchsize, width, height, number_anchors, remaining_points)
        data = tf.reshape(inputs, [shape[0], shape[1], shape[2], self._mask_len, -1])

        data = tf.cast(data, self.dtype)
        # detemine how much of the grid cell needs to be re consturcted
        if self._rebuild:
            anchors = self._get_anchor_grid(shape[1], shape[2], self._mask_len, self._anchors)
            centers = self._get_centers(shape[1], shape[2], self._mask_len)
        else:
            anchors = self._anchor_matrix
            centers = self._grid_cells
        
        if self._rebatch:
            anchors = self._reshape_batch(anchors, shape[0])
            centers = self._reshape_batch(centers, shape[0])

        # compute the true box output values
        box_xy = (tf.math.sigmoid(data[..., 0:2]) + centers)/tf.cast(shape[1], dtype = self.dtype)
        box_wh = tf.math.exp(data[..., 2:4])*anchors

        # convert the box to Tensorflow Expected format
        minpoint = box_xy - box_wh/2
        maxpoint = box_xy + box_wh/2
        box = K.stack([minpoint[..., 1], minpoint[..., 0], maxpoint[..., 1], maxpoint[..., 0]], axis = -1)

        # computer objectness and generate grid cell mask for where objects are located in the image
        objectness = tf.expand_dims(tf.math.sigmoid(data[..., 4]), axis = -1)
        scaled = tf.math.sigmoid(data[..., 5:]) * objectness
        
        #compute the mask of where objects have been located
        mask = tf.reduce_any(objectness > tf.cast(self._thresh, dtype = self.dtype), axis= -1)
        mask = tf.reduce_any(mask, axis= 0) 

        # reduce the dimentions of the box predictions to (batch size, max predictions, 4)
        box = tf.boolean_mask(box, mask, axis = 1)[:, :200, :]
        # reduce the dimentions of the box predictions to (batch size, max predictions, classes)
        classifications = tf.boolean_mask(scaled, mask, axis = 1)[:, :200, :]
        return box, classifications


class YoloGT(ks.layers.Layer):
    def __init__(self, anchors, thresh, max_box = 200, dtype = tf.float32, reshape = True, **kwargs):
        self._mask_len = len(anchors)
        self._dtype = dtype
        self._anchors = tf.cast(tf.convert_to_tensor(anchors), dtype = self._dtype)/416
        self._thresh = tf.cast(thresh, dtype = self._dtype)

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
        minpoint = box_xy - box_wh/2
        maxpoint = box_xy + box_wh/2
        box = K.stack([minpoint[..., 1], minpoint[..., 0], maxpoint[..., 1], maxpoint[..., 0]], axis = -1)

        # computer objectness and generate grid cell mask for where objects are located in the image
        objectness = tf.expand_dims(data[..., 4], axis = -1)
        scaled = data[..., 5:]
        #scaled = classes * objectness
        
        mask = tf.reduce_any(objectness > tf.cast(0.0, dtype = self._dtype), axis= -1)
        mask = tf.reduce_any(mask, axis= 0)  

        # reduce the dimentions of the box predictions to (batch size, max predictions, 4)
        box = tf.boolean_mask(box, mask, axis = 1)[:, :200, :]
        
        # # reduce the dimentions of the box predictions to (batch size, max predictions, classes)
        classifications = tf.boolean_mask(scaled, mask, axis = 1)[:, :200, :]
        return box, classifications
        
#@ks.utils.register_keras_serializable(package='yolo')
class YoloLayer(ks.Model):
    def __init__(self,
                 masks,
                 anchors,
                 thresh,
                 cls_thresh,
                 max_boxes, 
                 dtype,
                 scale_boxes = 1, 
                 scale_mult = 1,
                 **kwargs):
        super().__init__(**kwargs)
        self._masks = masks
        self._anchors = self.scale_anchors(anchors, scale_boxes)
        self._scale_mult = scale_mult
        self._thresh = thresh
        self._cls_thresh = cls_thresh
        self._max_boxes = max_boxes
        self._keys = list(masks.keys())
        self._len_keys = len(self._keys)
        self._dtype = dtype
        return
    
    def scale_anchors(self, anchors, scale):
        if scale == 0:
            raise Exception("zeros division error")
        anchors = list(anchors)
        for i in range(len(anchors)):
            anchors[i] = list(anchors[i])
            for j in range(len(anchors[i])):
                anchors[i][j] = anchors[i][j]/scale
        return anchors

    def build(self, input_shape):
        if list(input_shape.keys()) != self._keys:
            raise Exception(f"input size does not match the layers initialization, {self._keys} != {list(input_shape.keys())}")
        
        self._filters = {}
        for i, key in enumerate(self._keys):
            anchors = [self._anchors[mask] for mask in self._masks[key]]
            self._filters[key] = YoloFilterCell(anchors = anchors, thresh = self._thresh, max_box = self._max_boxes, dtype = self._dtype)
        return
    
    def call(self, inputs):
        boxes, classifs = self._filters[self._keys[0]](inputs[self._keys[0]])

        i = 1
        while i < self._len_keys:
            key = self._keys[i]
            b, c = self._filters[key](inputs[key])
            boxes = K.concatenate([boxes, b], axis = 1)
            classifs = K.concatenate([classifs, c], axis = 1)
            i += 1 
        
        nms = tf.image.combined_non_max_suppression(tf.expand_dims(boxes, axis=2), classifs, self._max_boxes, self._max_boxes, self._thresh, self._cls_thresh)
        return nms.nmsed_boxes,  nms.nmsed_classes, nms.nmsed_scores


if __name__ == "__main__":
    x = tf.ones(shape = (1, 416, 416, 3))
    model = build_model()
    y = model(x)
    print(y)


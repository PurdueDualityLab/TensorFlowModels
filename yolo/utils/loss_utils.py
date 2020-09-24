import tensorflow as tf
from tensorflow.keras import backend as K

@tf.function
def _build_grid_points(lwidth, lheight, num, dtype = None):
    """ generate a grid that is used to detemine the relative centers of the bounding boxs """
    with tf.name_scope("center_grid"):
        if dtype == None:
            dtype = tf.keras.backend.floatx()
        x_left, y_left = tf.meshgrid(tf.range(0, lheight), tf.range(0, lwidth))
        x_y = K.stack([x_left, y_left], axis = -1)
        x_y = tf.cast(x_y, dtype = dtype)/tf.cast(lwidth, dtype = dtype)
        x_y = tf.expand_dims(tf.repeat(tf.expand_dims(x_y, axis = -2), num, axis = -2), axis = 0)
    return x_y

@tf.function
def _build_anchor_grid(width, height, anchors, num, dtype = None):
    with tf.name_scope("anchor_grid"):
        """ get the transformed anchor boxes for each dimention """
        if dtype == None:
            dtype = tf.keras.backend.floatx()
        anchors = tf.cast(anchors, dtype = dtype)
        anchors = tf.reshape(anchors, [1, -1])
        anchors = tf.repeat(anchors, width*height, axis = 0)
        anchors = tf.reshape(anchors, [1, width, height, num, -1])
    return anchors

class GridGenerator(object):
    inuse = dict()
    def __init__(self, anchors, masks = None, scale_anchors = None, name = None, dtype = None):
        if dtype == None:
            self.dtype = tf.keras.backend.floatx()
        else:
            self.dtype = dtype
        if masks != None:
            self._num = len(masks)
        else:
            self._num = tf.shape(anchors)[0]

        if masks != None:
            anchors = [anchors[mask] for mask in masks]

        if scale_anchors != None:
            anchors = self._scale_anchors(anchors, scale_anchors)

        self._anchors = tf.cast(tf.convert_to_tensor(anchors), dtype = self.dtype)

        self._prev_width = 13
        self._grid_points = _build_grid_points(13, 13, self._num, self.dtype)
        self._anchor_grid = _build_anchor_grid(13, 13, self._anchors, self._num, self.dtype)
        
        if name != None:
            if name not in GridGenerator.inuse.keys():
                GridGenerator.inuse[name] = self
            else:
                raise Exception("the name you are using is already in use")
        return
    
    def _scale_anchors(self, anchors, scale):
        if scale == 0:
            raise Exception("zeros division error")
        anchors = list(anchors)
        for i in range(len(anchors)):
            anchors[i] = list(anchors[i])
            for j in range(len(anchors[i])):
                anchors[i][j] = anchors[i][j]/scale
        return anchors
    
    @tf.function
    def _extend_batch(self, grid, batch_size):
        return tf.repeat(grid, batch_size, axis = 0)
    
    @tf.function
    def get_grids(self, width, height, batch_size):
        self.dtype = tf.keras.backend.floatx()
        if width != self._prev_width:
            del self._anchor_grid
            del self._grid_points
            self._grid_points = _build_grid_points(width, height, self._num, self.dtype)
            self._anchor_grid = _build_anchor_grid(width, height, self._anchors, self._num, self.dtype)
            self._prev_width = width
        elif self._grid_points.dtype != self.dtype:
            self._grid_points = tf.cast(self._grid_points, self.dtype)
            self._anchor_grid = tf.cast(self._anchor_grid, self.dtype)

        grid_points = self._extend_batch(self._grid_points, batch_size)
        anchor_grid = self._extend_batch(self._anchor_grid, batch_size)
        return grid_points, anchor_grid
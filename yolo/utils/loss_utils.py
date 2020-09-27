import tensorflow as tf
from tensorflow.keras import backend as K

@tf.function
def _build_grid_points(lwidth, lheight, num, dtype):
    """ generate a grid that is used to detemine the relative centers of the bounding boxs """
    with tf.name_scope("center_grid"):
        x_left, y_left = tf.meshgrid(tf.range(0, lheight), tf.range(0, lwidth))
        x_y = K.stack([x_left, y_left], axis = -1)
        x_y = tf.cast(x_y, dtype = dtype)/tf.cast(lwidth, dtype = dtype)
        x_y = tf.expand_dims(tf.repeat(tf.expand_dims(x_y, axis = -2), num, axis = -2), axis = 0)
    return x_y

@tf.function
def _build_anchor_grid(width, height, anchors, num, dtype):
    with tf.name_scope("anchor_grid"):
        """ get the transformed anchor boxes for each dimention """
        anchors = tf.cast(anchors, dtype = dtype)
        anchors = tf.reshape(anchors, [1, -1])
        anchors = tf.repeat(anchors, width*height, axis = 0)
        anchors = tf.reshape(anchors, [1, width, height, num, -1])
    return anchors

class GridGenerator(object):
    inuse = dict()
    def __init__(self, anchors, masks = None, scale_anchors = None, name = None, low_memory = True):
        self.dtype = tf.keras.backend.floatx()
        if masks != None:
            self._num = len(masks)
        else:
            self._num = tf.shape(anchors)[0]

        self._low_memory = low_memory
        if masks != None:
            anchors = [anchors[mask] for mask in masks]

        self._lock = False
        self._scale_anchors = scale_anchors
        self._anchors = tf.convert_to_tensor(anchors)

        if not self._low_memory:
            self._prev_width = 13
            self._grid_points = _build_grid_points(13, 13, self._num, self.dtype)
            self._anchor_grid = _build_anchor_grid(13, 13, tf.cast(self._anchors, self.dtype)/tf.cast(self._scale_anchors * 13, self.dtype), self._num, self.dtype)
        
        if name != None:
            if name not in GridGenerator.inuse.keys():
                GridGenerator.inuse[name] = self
            else:
                raise Exception("the name you are using is already in use")
        return
    
    @tf.function
    def _extend_batch(self, grid, batch_size):
        return tf.repeat(grid, batch_size, axis = 0)
    
    @tf.function
    def _get_grids_high_memory(self, width, height, batch_size, dtype = None):
        self._lock = True
        if dtype == None:
            self.dtype = tf.keras.backend.floatx()
        else:
            self.dtype = dtype

        if width != self._prev_width:
            del self._anchor_grid
            del self._grid_points
            self._grid_points = _build_grid_points(width, height, self._num, self.dtype)
            self._anchor_grid = _build_anchor_grid(width, height,  tf.cast(self._anchors, self.dtype)/tf.cast(self._scale_anchors * width, self.dtype), self._num, self.dtype)
            self._prev_width = width
        
        if self._grid_points.dtype != self.dtype:
            self._grid_points = tf.cast(self._grid_points, self.dtype)
            self._anchor_grid = tf.cast(self._anchor_grid, self.dtype)

        grid_points = self._extend_batch(self._grid_points, batch_size)
        anchor_grid = self._extend_batch(self._anchor_grid, batch_size)
        self._lock = False
        return grid_points, anchor_grid
    
    @tf.function
    def _get_grids_low_memory(self, width, height, batch_size, dtype = None):
        if not self._lock:
            if dtype == None:
                self.dtype = tf.keras.backend.floatx()
            else:
                self.dtype = dtype
        grid_points = _build_grid_points(width, height, self._num, self.dtype)
        anchor_grid = _build_anchor_grid(width, height,  tf.cast(self._anchors, self.dtype)/tf.cast(self._scale_anchors * width, self.dtype), self._num, self.dtype)
        grid_points = self._extend_batch(grid_points, batch_size)
        anchor_grid = self._extend_batch(anchor_grid, batch_size)
        return grid_points, anchor_grid
    
    @tf.function
    def __call__(self, width, height, batch_size, dtype = None):
        if self._low_memory or self._lock:
            return self._get_grids_low_memory(width, height, batch_size, dtype = dtype)
        else:
            return self._get_grids_high_memory(width, height, batch_size, dtype = dtype)
    
    @staticmethod
    def get_generator_from_key(key):
        if key == None:
            return None
        if key not in GridGenerator.inuse.keys():
            return None
        else:
            return GridGenerator.inuse[key]
        return None
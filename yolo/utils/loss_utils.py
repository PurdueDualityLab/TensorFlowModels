import tensorflow as tf
from tensorflow.keras import backend as K


@tf.function
def build_grided_gt(y_true, mask, size, true_shape, use_tie_breaker):
    """
    convert ground truth for use in loss functions
    Args: 
        y_true: tf.Tensor[] ground truth [box coords[0:4], classes_onehot[0:-1], best_fit_anchor_box]
        mask: list of the anchor boxes choresponding to the output, ex. [1, 2, 3] tells this layer to predict only the first 3 anchors in the total. 
        size: the dimensions of this output, for regular, it progresses from 13, to 26, to 52
    
    Return:
        tf.Tensor[] of shape [batch, size, size, #of_anchors, 4, 1, num_classes]
    """
    batches = tf.shape(y_true)[0]
    num_boxes = tf.shape(y_true)[1]
    len_masks = tf.shape(mask)[0]

    full = tf.zeros([batches, size, size, len_masks, true_shape[-1]])
    depth_track = tf.zeros((batches, size, size, len_masks), dtype=tf.int32)

    x = tf.cast(y_true[..., 0] * tf.cast(size, dtype=tf.float32),
                dtype=tf.int32)
    y = tf.cast(y_true[..., 1] * tf.cast(size, dtype=tf.float32),
                dtype=tf.int32)

    anchors = tf.repeat(tf.expand_dims(y_true[..., -5:], axis=-1),
                        len_masks,
                        axis=-1)

    update_index = tf.TensorArray(tf.int32, size=0, dynamic_size=True)
    update = tf.TensorArray(tf.float32, size=0, dynamic_size=True)

    i = 0
    anchor_id = 0
    for batch in range(batches):
        for box_id in range(num_boxes):
            if K.all(tf.math.equal(y_true[batch, box_id, 2:4], 0)):
                continue
            if K.any(tf.math.less(y_true[batch, box_id, 0:2], 0.0)) or K.any(
                    tf.math.greater_equal(y_true[batch, box_id, 0:2], 1.0)):
                continue

            if use_tie_breaker:
                for anchor_id in range(tf.shape(anchors)[-1]):
                    index = tf.math.equal(anchors[batch, box_id, anchor_id],
                                          mask)
                    if K.any(index):
                        #tf.print(anchor_id, anchors[batch, box_id, anchor_id])
                        p = tf.cast(K.argmax(tf.cast(index, dtype=tf.int32)),
                                    dtype=tf.int32)
                        uid = 1

                        used = depth_track[batch, y[batch, box_id],
                                           x[batch, box_id], p]
                        if anchor_id == 0:
                            # write the box to the update list
                            # the boxes output from yolo are for some reason have the x and y indexes swapped for some reason, I am not sure why
                            """peculiar"""
                            update_index = update_index.write(
                                i,
                                [batch, y[batch, box_id], x[batch, box_id], p])
                            value = K.concatenate([
                                y_true[batch, box_id, 0:4],
                                tf.convert_to_tensor([1.]),
                                y_true[batch, box_id, 4:-5]
                            ])
                            update = update.write(i, value)
                        elif tf.math.equal(used, 2) or tf.math.equal(used, 0):
                            uid = 2
                            # write the box to the update list
                            # the boxes output from yolo are for some reason have the x and y indexes swapped for some reason, I am not sure why
                            """peculiar"""
                            update_index = update_index.write(
                                i,
                                [batch, y[batch, box_id], x[batch, box_id], p])
                            value = K.concatenate([
                                y_true[batch, box_id, 0:4],
                                tf.convert_to_tensor([1.]),
                                y_true[batch, box_id, 4:-5]
                            ])
                            update = update.write(i, value)

                        depth_track = tf.tensor_scatter_nd_update(
                            depth_track,
                            [(batch, y[batch, box_id], x[batch, box_id], p)],
                            [uid])
                        i += 1
            else:
                index = tf.math.equal(anchors[batch, box_id, 0], mask)
                if K.any(index):
                    #tf.print(0, anchors[batch, box_id, 0])
                    p = tf.cast(K.argmax(tf.cast(index, dtype=tf.int32)),
                                dtype=tf.int32)
                    update_index = update_index.write(
                        i, [batch, y[batch, box_id], x[batch, box_id], p])
                    value = K.concatenate([
                        y_true[batch, box_id, 0:4],
                        tf.convert_to_tensor([1.]), y_true[batch, box_id, 4:-5]
                    ])
                    update = update.write(i, value)
                    i += 1

    # if the size of the update list is not 0, do an update, other wise, no boxes and pass an empty grid
    if tf.math.greater(update_index.size(), 0):
        update_index = update_index.stack()
        update = update.stack()
        full = tf.tensor_scatter_nd_add(full, update_index, update)
    #tf.print(K.sum(full), K.sum(y_true))
    return full


@tf.function
def _build_grid_points(lwidth, lheight, num, dtype):
    """ generate a grid that is used to detemine the relative centers of the bounding boxs """
    with tf.name_scope("center_grid"):
        x_left, y_left = tf.meshgrid(tf.range(0, lheight), tf.range(0, lwidth))
        x_y = K.stack([x_left, y_left], axis=-1)
        x_y = tf.cast(x_y, dtype=dtype) / tf.cast(lwidth, dtype=dtype)
        x_y = tf.expand_dims(tf.repeat(tf.expand_dims(x_y, axis=-2),
                                       num,
                                       axis=-2),
                             axis=0)
    return x_y


@tf.function
def _build_anchor_grid(width, height, anchors, num, dtype):
    with tf.name_scope("anchor_grid"):
        """ get the transformed anchor boxes for each dimention """
        anchors = tf.cast(anchors, dtype=dtype)
        anchors = tf.reshape(anchors, [1, -1])
        anchors = tf.repeat(anchors, width * height, axis=0)
        anchors = tf.reshape(anchors, [1, width, height, num, -1])
    return anchors


class GridGenerator(object):
    inuse = dict()

    def __init__(self,
                 anchors,
                 masks=None,
                 scale_anchors=None,
                 name=None,
                 low_memory=True):
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
            self._grid_points = _build_grid_points(13, 13, self._num,
                                                   self.dtype)
            self._anchor_grid = _build_anchor_grid(
                13, 13,
                tf.cast(self._anchors, self.dtype) /
                tf.cast(self._scale_anchors * 13, self.dtype), self._num,
                self.dtype)

        if name != None:
            if name not in GridGenerator.inuse.keys():
                GridGenerator.inuse[name] = self
            else:
                raise Exception("the name you are using is already in use")
        return

    @tf.function
    def _extend_batch(self, grid, batch_size):
        return tf.repeat(grid, batch_size, axis=0)

    @tf.function
    def _get_grids_high_memory(self, width, height, batch_size, dtype=None):
        self._lock = True
        if dtype == None:
            self.dtype = tf.keras.backend.floatx()
        else:
            self.dtype = dtype

        if width != self._prev_width:
            del self._anchor_grid
            del self._grid_points
            self._grid_points = _build_grid_points(width, height, self._num,
                                                   self.dtype)
            self._anchor_grid = _build_anchor_grid(
                width, height,
                tf.cast(self._anchors, self.dtype) /
                tf.cast(self._scale_anchors * width, self.dtype), self._num,
                self.dtype)
            self._prev_width = width

        if self._grid_points.dtype != self.dtype:
            self._grid_points = tf.cast(self._grid_points, self.dtype)
            self._anchor_grid = tf.cast(self._anchor_grid, self.dtype)

        grid_points = self._extend_batch(self._grid_points, batch_size)
        anchor_grid = self._extend_batch(self._anchor_grid, batch_size)
        self._lock = False
        return grid_points, anchor_grid

    @tf.function
    def _get_grids_low_memory(self, width, height, batch_size, dtype=None):
        if not self._lock:
            if dtype == None:
                self.dtype = tf.keras.backend.floatx()
            else:
                self.dtype = dtype
        grid_points = _build_grid_points(width, height, self._num, self.dtype)
        anchor_grid = _build_anchor_grid(
            width, height,
            tf.cast(self._anchors, self.dtype) /
            tf.cast(self._scale_anchors * width, self.dtype), self._num,
            self.dtype)
        grid_points = self._extend_batch(grid_points, batch_size)
        anchor_grid = self._extend_batch(anchor_grid, batch_size)
        return grid_points, anchor_grid

    @tf.function
    def __call__(self, width, height, batch_size, dtype=None):
        if self._low_memory or self._lock:
            return self._get_grids_low_memory(width,
                                              height,
                                              batch_size,
                                              dtype=dtype)
        else:
            return self._get_grids_high_memory(width,
                                               height,
                                               batch_size,
                                               dtype=dtype)

    @staticmethod
    def get_generator_from_key(key):
        if key == None:
            return None
        if key not in GridGenerator.inuse.keys():
            return None
        else:
            return GridGenerator.inuse[key]
        return None

import tensorflow as tf
from tensorflow.keras import backend as K
from yolo.ops import box_ops
from yolo.ops import math_ops

import matplotlib.pyplot as plt
import numpy as np

VISUALIZE = False

def _build_grid_points(lwidth, lheight, anchors, dtype):
  """ generate a grid that is used to detemine the relative centers of the bounding boxs """
  with tf.name_scope('center_grid'):
    y = tf.range(0, lheight)
    x = tf.range(0, lwidth)
    num = tf.shape(anchors)[0]
    x_left = tf.tile(
        tf.transpose(tf.expand_dims(y, axis=-1), perm=[1, 0]), [lwidth, 1])
    y_left = tf.tile(tf.expand_dims(x, axis=-1), [1, lheight])
    x_y = K.stack([x_left, y_left], axis=-1)
    x_y = tf.cast(x_y, dtype=dtype)
    x_y = tf.expand_dims(
        tf.tile(tf.expand_dims(x_y, axis=-2), [1, 1, num, 1]), axis=0)
  return x_y


def _build_anchor_grid(width, height, anchors, dtype):  #, num, dtype):
  with tf.name_scope('anchor_grid'):
    """ get the transformed anchor boxes for each dimention """
    num = tf.shape(anchors)[0]
    anchors = tf.cast(anchors, dtype=dtype)
    anchors = tf.reshape(anchors, [1, 1, 1, num, 2])
  return anchors


class GridGenerator(object):
  def __init__(self, anchors, masks=None, scale_anchors=None):
    self.dtype = tf.keras.backend.floatx()
    if masks is not None:
      self._num = len(masks)
    else:
      self._num = tf.shape(anchors)[0]

    if masks is not None:
      anchors = [anchors[mask] for mask in masks]

    self._scale_anchors = scale_anchors
    self._anchors = tf.convert_to_tensor(anchors)
    return

  def _extend_batch(self, grid, batch_size):
    return tf.tile(grid, [batch_size, 1, 1, 1, 1])

  def __call__(self, width, height, batch_size, dtype=None):
    if dtype is None:
      self.dtype = tf.keras.backend.floatx()
    else:
      self.dtype = dtype
    grid_points = _build_grid_points(width, height, self._anchors, self.dtype)
    anchor_grid = _build_anchor_grid(
        width, height,
        tf.cast(self._anchors, self.dtype) /
        tf.cast(self._scale_anchors, self.dtype), self.dtype)

    grid_points = self._extend_batch(grid_points, batch_size)
    anchor_grid = self._extend_batch(anchor_grid, batch_size)
    return grid_points, anchor_grid


TILE_SIZE = 50
class PairWiseSearch(object):

  def __init__(self, 
               iou_type = "iou", 
               any = True, 
               min_conf = 0.0, 
               track_boxes = False, 
               track_classes = False):
    """
    This method applies a pair wise serach between the ground truth 
    and the labels. The goal is to indicate the locations where the 
    predictions over lap with the groud truth for dynamic ground 
    truth constructions.   
    """
    if iou_type == "giou":
      self.iou_type = 1
    elif iou_type == "ciou":
      self.iou_type = 2
    else:
      self.iou_type = 0

    self._any = any
    self._min_conf = min_conf
    self._track_boxes = track_boxes
    self._track_classes = track_classes
    return 

  def box_iou(self, true_box, pred_box):
    # based on the type of loss, compute the iou loss for a box
    # compute_<name> indicated the type of iou to use
    if self.iou_type == 1:
      _, iou = box_ops.compute_giou(true_box, pred_box)
    elif self.iou_type == 2:
      _, iou = box_ops.compute_ciou(true_box, pred_box)
    else:
      iou = box_ops.compute_iou(true_box, pred_box)
    return iou

  def _search_body(self, pred_box     , pred_class     , 
                         boxes        , classes        , 
                         running_boxes, running_classes, 
                         max_iou      , idx):
    # capture the batch size to be used, and gather a slice of
    # boxes from the ground truth. currently TILE_SIZE = 50, to
    # save memory
    batch_size = tf.shape(boxes)[0]
    box_slice = tf.slice(boxes, [0, idx * TILE_SIZE, 0],
                                [batch_size, TILE_SIZE, 4])

    # match the dimentions of the slice to the model predictions
    # shape: [batch_size, 1, 1, num, TILE_SIZE, 4]
    box_slice = tf.expand_dims(box_slice, axis=1)
    box_slice = tf.expand_dims(box_slice, axis=1)
    box_slice = tf.expand_dims(box_slice, axis=1)

    box_grid = tf.expand_dims(pred_box, axis=-2)

    # capture the classes
    class_slice = tf.slice(classes, [0, idx * TILE_SIZE],
                            [batch_size, TILE_SIZE])
    class_slice = tf.expand_dims(class_slice, axis=1)
    class_slice = tf.expand_dims(class_slice, axis=1)
    class_slice = tf.expand_dims(class_slice, axis=1)

    iou = self.box_iou(box_slice, box_grid) 

    if self._min_conf > 0.0:
      if not self._any:
        class_grid = tf.expand_dims(pred_class, axis=-2)
        class_mask = tf.one_hot(
          tf.cast(class_slice, tf.int32),
          depth=tf.shape(pred_class)[-1],
          dtype=pred_class.dtype)
        class_mask = tf.reduce_any(tf.equal(class_mask, class_grid), axis = -1)
      else:
        class_mask = tf.reduce_max(pred_class, axis = -1, keepdims=True)
      class_mask = tf.cast(class_mask, iou.dtype)
      iou *= class_mask

    max_iou_ = tf.concat([max_iou, iou], axis = -1)
    max_iou = tf.reduce_max(max_iou_, axis = -1, keepdims = True)
    ind = tf.expand_dims(tf.argmax(max_iou_, axis = -1), axis = -1) 

    if self._track_boxes:
      running_boxes = tf.expand_dims(running_boxes, axis = -2)
      box_slice = tf.zeros_like(running_boxes) + box_slice
      box_slice = tf.concat([running_boxes, box_slice], axis = -2)
      running_boxes = tf.gather_nd(box_slice, ind, batch_dims = 4)

    if self._track_classes:
      running_classes = tf.expand_dims(running_classes, axis = -1)
      class_slice = tf.zeros_like(running_classes) + class_slice
      class_slice = tf.concat([running_classes, class_slice], axis = -1)
      running_classes = tf.gather_nd(class_slice, ind, batch_dims = 4)

    return (pred_box     , pred_class     , 
            boxes        , classes        , 
            running_boxes, running_classes, 
            max_iou      , idx + 1)
  
  def visualize(self, pred_boxes, max_iou, running_boxes, running_classes):
    if VISUALIZE:
      iou = self.box_iou(pred_boxes, running_boxes)
      fig, axe = plt.subplots(1, 2)
      axe[0].imshow(max_iou[0, ..., 0].numpy())
      axe[1].imshow(iou[0, ...].numpy())
      plt.show()
    return 

  def __call__(self, pred_boxes, pred_classes, boxes, classes, yxyx = True):
    num_boxes = tf.shape(boxes)[-2]
    num_tiles = num_boxes // TILE_SIZE

    if yxyx: 
      boxes = box_ops.yxyx_to_xcycwh(boxes)

    if self._min_conf > 0.0:
      pred_classes = tf.cast(pred_classes > self._min_conf, pred_classes.dtype)

    def _loop_cond(pred_box     , pred_class     , 
                   boxes        , classes        , 
                   running_boxes, running_classes, 
                   max_iou      , idx):

      # check that the slice has boxes that all zeros
      batch_size = tf.shape(boxes)[0]
      box_slice = tf.slice(boxes, [0, idx * TILE_SIZE, 0],
                           [batch_size, TILE_SIZE, 4])

      return tf.logical_and(idx < num_tiles,
                            tf.math.greater(tf.reduce_sum(box_slice), 0))

    running_boxes = tf.zeros_like(pred_boxes)
    running_classes = tf.zeros_like(tf.reduce_sum(running_boxes, axis = -1))
    max_iou = tf.zeros_like(tf.reduce_sum(running_boxes, axis = -1))
    max_iou = tf.expand_dims(max_iou, axis = -1)

    (pred_boxes   , pred_classes   , 
     boxes        , classes        , 
     running_boxes, running_classes, 
     max_iou      , idx) = tf.while_loop(
          _loop_cond,
          self._search_body, 
          [pred_boxes   , pred_classes   , 
           boxes        , classes        , 
           running_boxes, running_classes, 
           max_iou      , tf.constant(0)])

    mask = tf.cast(max_iou > 0.0, running_boxes.dtype)
    running_boxes *= mask
    running_classes *= tf.squeeze(mask, axis = -1)
    max_iou = tf.squeeze(max_iou, axis = -1)

    return (tf.stop_gradient(running_boxes), 
            tf.stop_gradient(running_classes), 
            tf.stop_gradient(max_iou))

def apply_mask(mask, x, value = 0):
  """This function is used for gradient masking. The YOLO loss function makes 
  extensive use of dynamically shaped tensors. To allow this use case on the 
  TPU while preserving the gradient correctly for back propagation we use this 
  masking function to use a tf.where operation to hard set masked location to 
  have a gradient and a value of zero. 

  Args: 
    mask: A `Tensor` with the same shape as x used to select values of 
      importance.
    x: A `Tensor` with the same shape as mask that will be getting masked.
  
  Returns: 
    x: A masked `Tensor` with the same shape as x.
  """
  mask = tf.cast(mask, tf.bool)
  masked = tf.where(mask, x, tf.zeros_like(x) + value)
  return masked


def build_grid(indexes, truths, preds, ind_mask, update=False):
  # this function is used to broadcast all the indexes to the correct
  # into the correct ground truth mask, used for iou detection map
  # in the scaled loss and the classification mask in the darknet loss
  num_flatten = tf.shape(preds)[-1]

  # is there a way to verify that we are not on the CPU?
  ind_mask = tf.cast(ind_mask, indexes.dtype)

  # find all the batch indexes using the cumulated sum of a ones tensor
  # cumsum(ones) - 1 yeild the zero indexed batches
  bhep = tf.reduce_max(tf.ones_like(indexes), axis=-1, keepdims=True)
  bhep = tf.math.cumsum(bhep, axis=0) - 1

  # concatnate the batch sizes to the indexes
  indexes = tf.concat([bhep, indexes], axis=-1)
  indexes = apply_mask(tf.cast(ind_mask, indexes.dtype), indexes)
  indexes = (indexes + (ind_mask - 1))

  # reshape the indexes into the correct shape for the loss,
  # just flatten all indexes but the last
  indexes = tf.reshape(indexes, [-1, 4])

  # also flatten the ground truth value on all axis but the last
  truths = tf.reshape(truths, [-1, num_flatten])

  # build a zero grid in the samve shape as the predicitons
  grid = tf.zeros_like(preds)
  # remove invalid values from the truths that may have
  # come up from computation, invalid = nan and inf
  truths = math_ops.rm_nan_inf(truths)

  # scatter update the zero grid
  if update:
    grid = tf.tensor_scatter_nd_update(grid, indexes, truths)
  else:
    grid = tf.tensor_scatter_nd_max(grid, indexes, truths)

  # stop gradient and return to avoid TPU errors and save compute
  # resources
  return grid
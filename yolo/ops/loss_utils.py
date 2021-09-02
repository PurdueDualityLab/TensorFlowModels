import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.python.ops.custom_gradient import custom_gradient
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


def clip_by_value(vala, valb, max_delta):
  valb = math_ops.rm_nan_inf(valb)
  vala = math_ops.rm_nan_inf(vala)

  delta = tf.cast(max_delta, valb.dtype)
  vala = tf.clip_by_value(vala, -delta, delta)
  valb = tf.clip_by_value(valb, -delta, delta)
  return vala, valb


def _decode_boxes_yolo(pred_xy, pred_wh, width, height, anchor_grid, grid_points,
                max_delta, scale_xy):
  # build a scaling tensor to get the offset of th ebox relative to the image
  scaler = tf.convert_to_tensor([height, width, height, width])

  # cast the grid scaling value to a tensorflow data type, in yolo each pixel is
  # used to predict the center of a box, the center must be with in the bounds
  # of representation of each pixel, typically 1/width pixels. Scale_xy, extends
  # the view of each pixel such that the offset is no longer bound by 0 and 1.
  scale_xy = tf.cast(scale_xy, pred_xy.dtype)

  # apply the sigmoid
  pred_xy = tf.math.sigmoid(pred_xy)

  # scale the centers and find the offset of each box relative to
  # their center pixel
  pred_xy = pred_xy * scale_xy - 0.5 * (scale_xy - 1)

  # scale the offsets and add them to the grid points or a tensor that is
  # the realtive location of each pixel
  box_xy = grid_points + pred_xy

  # scale the width and height of the predictions and corlate them
  # to anchor boxes
  box_wh = tf.math.exp(pred_wh) * anchor_grid

  # build the final predicted box
  scaled_box = K.concatenate([box_xy, box_wh], axis=-1)
  pred_box = scaled_box / scaler

  # shift scaled boxes
  scaled_box = K.concatenate([pred_xy, box_wh], axis=-1)
  return (scaler, scaled_box, pred_box)


@tf.custom_gradient
def _darknet_boxes_yolo(pred_xy, pred_wh, width, height, anchor_grid, grid_points,
                  max_delta, scale_xy, normalizer):
  (scaler, scaled_box, pred_box) = _decode_boxes_yolo(pred_xy, pred_wh, width, height,
                                               anchor_grid, grid_points,
                                               max_delta, scale_xy)

  def delta(dy_scaler, dy_scaled, dy):
    # here we do not propgate the scaling of the prediction through the network
    # because in back prop is leads to the scaling of the gradient of a box
    # relative to the size of the box and the gradient for small boxes this will
    # mean down scaling the gradient leading to poor convergence on small
    # objects and also very likely over fitting

    # dy_scaled *= scaler
    dy_xy, dy_wh = tf.split(dy, 2, axis=-1)
    dy_xy_, dy_wh_ = tf.split(dy_scaled, 2, axis=-1)

    # add all the gradients that may have been applied to the
    # boxes and those that have been applied to the width and height
    dy_wh += dy_wh_
    dy_xy += dy_xy_

    # propagate the exponential applied to the width and height in
    # order to ensure the gradient propagated is of the correct
    # magnitude
    dy_wh *= tf.math.exp(pred_wh)

    # apply the gradient clipping to xy and wh
    dy_xy, dy_wh = clip_by_value(dy_xy, dy_wh, max_delta)
    return dy_xy, dy_wh, 0.0, 0.0, tf.zeros_like(anchor_grid), tf.zeros_like(
        grid_points), 0.0, 0.0, 0.0

  return (scaler, scaled_box, pred_box), delta


def get_predicted_box(width,
                      height,
                      unscaled_box,
                      anchor_grid,
                      grid_points,
                      scale_xy,
                      darknet=True,
                      max_delta=5.0,
                      normalizer=1.0):
  """Decodes the predicted boxes from the model format to a usable 
  [x, y, w, h] format for use in the loss function as well as for use 
  with in the detection generator.   

  Args: 
    width: `float` scalar indicating the width of the prediction layer.
    height: `float` scalar indicating the height of the prediction layer
    unscaled_box: Tensor of shape [..., height, width, 4] holding encoded boxes.
    anchor_grid: Tensor of shape [..., 1, 1, 2] holding the anchor boxes 
      organized for box decoding box width and height.  
    grid_points: Tensor of shape [..., height, width, 2] holding the anchor 
      boxes for decoding the box centers.
    scale_xy: `float` scaler used to indicating the range for each center 
      outside of its given [..., i, j, 4] index, where i and j are indexing 
      pixels along width and height in the predicited output map. 
    darknet: `bool` used to select between custom gradient and default autograd.  
    max_delta: `float` scaler used for gradient clipping in back propagation. 
  
  Returns: 
    scaler: Tensor of shape [4] returned to allow the scaling of the ground 
      Truth boxes to be of the same magnitude as the decoded predicted boxes.
    scaled_box: Tensor of shape [..., height, width, 4] with the predicted 
      boxes.
    pred_box: Tensor of shape [..., height, width, 4] with the predicted boxes 
      devided by the scaler parameter used to put all boxes in the [0, 1] range. 
  """
  
  pred_xy = unscaled_box[..., 0:2]
  pred_wh = unscaled_box[..., 2:4]

  if darknet:
    # if we are using the darknet loss we shoud nto propagate the
    # decoding of the box
    (scaler, scaled_box,
     pred_box) = _darknet_boxes_yolo(pred_xy, pred_wh, width, height, anchor_grid,
                               grid_points, max_delta, scale_xy, normalizer)
  else:
    # if we are using the scaled loss we should propagate the decoding of
    # the boxes
    (scaler, scaled_box,
     pred_box) = _decode_boxes_yolo(pred_xy, pred_wh, width, height, anchor_grid,
                             grid_points, max_delta, scale_xy)

  return (scaler, scaled_box, pred_box)


def _decode_boxes_scaledyolo(pred_xy, pred_wh, width, height, anchor_grid,
                          grid_points, max_delta, scale_xy):

  # build a scaling tensor to get the offset of th ebox relative to the image
  scaler = tf.convert_to_tensor([height, width, height, width])

  # cast the grid scaling value to a tensorflow data type, in yolo each pixel is
  # used to predict the center of a box, the center must be with in the bounds
  # of representation of each pixel, typically 1/width pixels. Scale_xy, extends
  # the view of each pixel such that the offset is no longer bound by 0 and 1.
  scale_xy = tf.cast(scale_xy, pred_xy.dtype)

  # apply the sigmoid
  pred_xy = tf.math.sigmoid(pred_xy)
  pred_wh = tf.math.sigmoid(pred_wh)

  # scale the xy offset predictions according to the config
  pred_xy = pred_xy * scale_xy - 0.5 * (scale_xy - 1)

  # find the true offset from the grid points and the scaler
  # where the grid points are the relative offset of each pixel with
  # in the image
  box_xy = grid_points + pred_xy

  # decode the widht and height of the boxes and correlate them
  # to the anchor boxes
  box_wh = (2 * pred_wh)**2 * anchor_grid

  # build the final boxes
  scaled_box = K.concatenate([box_xy, box_wh], axis=-1)
  pred_box = scaled_box / scaler

  # shift scaled boxes
  scaled_box = K.concatenate([pred_xy, box_wh], axis=-1)
  return (scaler, scaled_box, pred_box)


@tf.custom_gradient
def _darknet_boxes_scaledyolo(pred_xy, pred_wh, width, height, anchor_grid,
                            grid_points, max_delta, scale_xy, normalizer):

  (scaler, scaled_box,
   pred_box) = _decode_boxes_scaledyolo(pred_xy, pred_wh, width, height,
                                     anchor_grid, grid_points, max_delta,
                                     scale_xy)

  def delta(dy_scaler, dy_scaled, dy):
    # dy_scaled *= scaler
    dy_xy, dy_wh = tf.split(dy, 2, axis=-1)
    dy_xy_, dy_wh_ = tf.split(dy_scaled, 2, axis=-1)

    # add all the gradients that may have been applied to the
    # boxes and those that have been applied to the width and height
    dy_wh += dy_wh_
    dy_xy += dy_xy_

    tf.print("here")

    # apply the gradient clipping to xy and wh
    dy_xy, dy_wh = clip_by_value(dy_xy, dy_wh, max_delta)
    return dy_xy, dy_wh, 0.0, 0.0, tf.zeros_like(anchor_grid), tf.zeros_like(
        grid_points), 0.0, 0.0, 0.0

  return (scaler, scaled_box, pred_box), delta

def get_predicted_box_scaledyolo(width,
                               height,
                               unscaled_box,
                               anchor_grid,
                               grid_points,
                               scale_xy,
                               darknet=False,
                               max_delta=5.0,
                               normalizer=1.0):
  """Decodes the predicted boxes from the model format to a usable 
  [x, y, w, h] format for use in the loss function as well as for use 
  with in the detection generator.   

  Args: 
    width: `float` scalar indicating the width of the prediction layer.
    height: `float` scalar indicating the height of the prediction layer
    unscaled_box: Tensor of shape [..., height, width, 4] holding encoded boxes.
    anchor_grid: Tensor of shape [..., 1, 1, 2] holding the anchor boxes 
      organized for box decoding box width and height.  
    grid_points: Tensor of shape [..., height, width, 2] holding the anchor 
      boxes for decoding the box centers.
    scale_xy: `float` scaler used to indicating the range for each center 
      outside of its given [..., i, j, 4] index, where i and j are indexing 
      pixels along width and height in the predicited output map. 
    darknet: `bool` used to select between custom gradient and default autograd.  
    max_delta: `float` scaler used for gradient clipping in back propagation. 
  
  Returns: 
    scaler: Tensor of shape [4] returned to allow the scaling of the ground 
      Truth boxes to be of the same magnitude as the decoded predicted boxes.
    scaled_box: Tensor of shape [..., height, width, 4] with the predicted 
      boxes.
    pred_box: Tensor of shape [..., height, width, 4] with the predicted boxes 
      devided by the scaler parameter used to put all boxes in the [0, 1] range. 
  """

  pred_xy = unscaled_box[..., 0:2]
  pred_wh = unscaled_box[..., 2:4]

  if darknet:
    # if we are using the darknet loss we shoud nto propagate the decoding
    # of the box
    (scaler, scaled_box,
     pred_box) = _darknet_boxes_scaledyolo(pred_xy, pred_wh, width, height,
                                         anchor_grid, grid_points, max_delta,
                                         scale_xy, normalizer)
  else:
    # if we are using the scaled loss we should propagate the decoding of the
    # boxes
    (scaler, scaled_box,
     pred_box) = _decode_boxes_scaledyolo(pred_xy, pred_wh, width, height,
                                       anchor_grid, grid_points, max_delta,
                                       scale_xy)
  return (scaler, scaled_box, pred_box)

def get_predicted_box_anchorfree(width,
                                 height,
                                 unscaled_box,
                                 stride,
                                 grid_points,
                                 scale_xy,
                                 darknet=False,
                                 max_delta=5.0,
                                 normalizer=1.0):
  """Decodes the predicted boxes from the model format to a usable 
  [x, y, w, h] format for use in the loss function as well as for use 
  with in the detection generator.   

  Args: 
    width: `float` scalar indicating the width of the prediction layer.
    height: `float` scalar indicating the height of the prediction layer
    unscaled_box: Tensor of shape [..., height, width, 4] holding encoded boxes.
    anchor_grid: Tensor of shape [..., 1, 1, 2] holding the anchor boxes 
      organized for box decoding box width and height.  
    grid_points: Tensor of shape [..., height, width, 2] holding the anchor 
      boxes for decoding the box centers.
    scale_xy: `float` scaler used to indicating the range for each center 
      outside of its given [..., i, j, 4] index, where i and j are indexing 
      pixels along width and height in the predicited output map. 
    darknet: `bool` used to select between custom gradient and default autograd.  
    max_delta: `float` scaler used for gradient clipping in back propagation. 
  
  Returns: 
    scaler: Tensor of shape [4] returned to allow the scaling of the ground 
      Truth boxes to be of the same magnitude as the decoded predicted boxes.
    scaled_box: Tensor of shape [..., height, width, 4] with the predicted 
      boxes.
    pred_box: Tensor of shape [..., height, width, 4] with the predicted boxes 
      devided by the scaler parameter used to put all boxes in the [0, 1] range. 
  """

  pred_xy = unscaled_box[..., 0:2]
  pred_wh = unscaled_box[..., 2:4]

  one_grid = tf.ones_like(grid_points)
  (_, _, pred_box) = _decode_boxes_yolo(pred_xy, pred_wh, width, height, 
                                       one_grid, grid_points, max_delta, 
                                       scale_xy)
  scaled_box = pred_box * stride
  scaler = stride
  return (scaler, scaled_box, pred_box)



@tf.custom_gradient
def grad_sigmoid(values):
  # This is an identity operation that will
  # allow us to add some steps to the back propagation.
  def delta(dy):
    # Darknet only propagtes sigmoids for the boxes
    # under some conditions, so we need this to selectively
    # add the sigmoid to the chain rule
    t = tf.math.sigmoid(values)
    return dy * t * (1 - t)

  return values, delta


@tf.custom_gradient
def sigmoid_BCE(y, x_prime, label_smoothing):
  """Applies the Sigmoid Cross Entropy Loss Using the same deriviative as that 
  found in the Darknet C library. The derivative of this method is not the same 
  as the standard binary cross entropy with logits function.

  The BCE with logits function equasion is as follows: 
    x = 1 / (1 + exp(-x_prime))
    bce = -ylog(x) - (1 - y)log(1 - x) 

  The standard BCE with logits function derivative is as follows: 
    dloss = -y/x + (1-y)/(1-x)
    dsigmoid = x * (1 - x)
    dx = dloss * dsigmoid
  
  This derivative can be reduced simply to: 
    dx = (-y + x)
  
  This simplification is used by the darknet library in order to improve 
  training stability. 

  Args: 
    y: Tensor holding the ground truth data. 
    x_prime: Tensor holding the predictions prior to application of the sigmoid 
      operation.
    label_smoothing: float value between 0.0 and 1.0 indicating the amount of 
      smoothing to apply to the data.

  Returns: 
    bce: Tensor of the bce applied loss values.
    delta: callable function indicating the custom gradient for this operation.  
  """
  eps = 1e-9
  x = tf.math.sigmoid(x_prime)
  y = tf.stop_gradient(y * (1 - label_smoothing) + 0.5 * label_smoothing)
  bce = -y * tf.math.log(x + eps) -(1 - y) * tf.math.log(1 - x + eps) 

  def delta(dpass):
    x = tf.math.sigmoid(x_prime)
    dx = (-y + x) * dpass
    dy = tf.zeros_like(y)
    return dy, dx, 0.0
  return bce, delta
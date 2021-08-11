import tensorflow as tf
import tensorflow.keras as ks
from tensorflow.keras import backend as K

from yolo.ops.loss_utils import GridGenerator
from yolo.ops import box_ops
from yolo.ops import math_ops
import numpy as np

from functools import partial

# loss testing
import matplotlib.pyplot as plt
import numpy as np

TILE_SIZE = 50


@tf.custom_gradient
def grad_sigmoid(values):
  # this is an identity operation that will
  # allow us to add some steps to the back propagation
  def delta(dy):
    # darknet only propagtes sigmoids for the boxes
    # under some conditions, so we need this to selectively
    # add the sigmoid to the chain rule
    t = tf.math.sigmoid(values)
    return dy * t * (1 - t)

  return values, delta


@tf.custom_gradient
def sigmoid_BCE(y, x_prime, label_smoothing):
  # this applies the sigmoid cross entropy loss
  x = tf.math.sigmoid(x_prime)
  y = y * (1 - label_smoothing) + 0.5 * label_smoothing
  # x = math_ops.rm_nan_inf(x, val=0.0)
  # bce = tf.reduce_sum(tf.square(-y + x), axis=-1)

  bce = ks.losses.binary_crossentropy(
      y, x_prime, label_smoothing=0.0, from_logits=True)

  def delta(dy):
    # this is a safer version of the sigmoid with binary cross entropy
    # bellow is the mathematic formula reduction that is used to
    # get the simplified derivative. The derivative reduces to
    # y_pred - y_true

    # bce = -ylog(x) - (1 - y)log(1 - x)
    # # bce derive
    # dloss = -y/x + (1-y)/(1-x)
    # # 1 / (1 + exp(-x))
    # dsigmoid = x * (1 - x)
    # dx = dloss * dsigmoid * tf.expand_dims(dy, axis = -1)

    # # bce derive
    # dloss = -y * (1 - x) + (1 - y) * x
    # # 1 / (1 + exp(-x))
    # dx = dloss * tf.expand_dims(dy, axis = -1)

    # bce * sigmoid derivative
    dloss = (-y + x)
    dx = dloss * tf.expand_dims(dy, axis=-1)

    dy = tf.zeros_like(y)
    return dy, dx, 0.0

  return bce, delta

def apply_mask(mask, x):
  mask = tf.cast(mask, tf.bool)
  masked = tf.where(mask, x, tf.zeros_like(x))
  return masked

# @tf.custom_gradient
# def apply_mask(mask, x):
#   # this function is used to apply no nan mask to an input tensor
#   # as such this will apply a mask and remove NAN for both the
#   # forward AND backward propagation
#   mask = tf.cast(mask, tf.bool)
#   masked = tf.where(mask, x, tf.zeros_like(x))

#   def delta(dy):
#     # mask the incoming derivative as well.
#     masked_dy = tf.where(mask, dy, tf.zeros_like(dy))
#     return tf.zeros_like(mask), masked_dy

#   return masked , delta


def scale_boxes(pred_xy, pred_wh, width, height, anchor_grid, grid_points,
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
def darknet_boxes(pred_xy, pred_wh, width, height, anchor_grid, grid_points,
                  max_delta, scale_xy, normalizer):

  (scaler, scaled_box, pred_box) = scale_boxes(pred_xy, pred_wh, width, height,
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

    # # apply scaling for gradients if scaled boxes are
    # sc_xy_, sc_wh_ = tf.split(scaler, 2, axis=-1)
    # dy_xy_ *= sc_xy_
    # dy_wh_ *= anchor_grid

    # add all the gradients that may have been applied to the
    # boxes and those that have been applied to the width and height
    dy_wh += dy_wh_
    dy_xy += dy_xy_

    # propagate the exponential applied to the width and height in
    # order to ensure the gradient propagated is of the correct
    # magnitude
    dy_wh *= tf.math.exp(pred_wh)

    # apply the gradient clipping to xy and wh
    dy_wh = math_ops.rm_nan_inf(dy_wh)
    delta = tf.cast(max_delta, dy_wh.dtype)
    dy_wh = tf.clip_by_value(dy_wh, -delta, delta)

    dy_xy = math_ops.rm_nan_inf(dy_xy)
    delta = tf.cast(max_delta, dy_xy.dtype)
    dy_xy = tf.clip_by_value(dy_xy, -delta, delta)

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

  pred_xy = unscaled_box[..., 0:2]
  pred_wh = unscaled_box[..., 2:4]

  if darknet:
    # if we are using the darknet loss we shoud nto propagate the
    # decoding of the box
    (scaler, scaled_box,
     pred_box) = darknet_boxes(pred_xy, pred_wh, width, height, anchor_grid,
                               grid_points, max_delta, scale_xy, normalizer)
  else:
    # if we are using the scaled loss we should propagate the decoding of
    # the boxes
    (scaler, scaled_box,
     pred_box) = scale_boxes(pred_xy, pred_wh, width, height, anchor_grid,
                             grid_points, max_delta, scale_xy)

  return (scaler, scaled_box, pred_box)


def new_coord_scale_boxes(pred_xy, pred_wh, width, height, anchor_grid,
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
  box_wh = tf.square(2 * pred_wh) * anchor_grid

  # build the final boxes
  scaled_box = K.concatenate([box_xy, box_wh], axis=-1)
  pred_box = scaled_box / scaler

  # shift scaled boxes
  scaled_box = K.concatenate([pred_xy, box_wh], axis=-1)
  return (scaler, scaled_box, pred_box)


@tf.custom_gradient
def darknet_new_coord_boxes(pred_xy, pred_wh, width, height, anchor_grid,
                            grid_points, max_delta, scale_xy, normalizer):
  (scaler, scaled_box,
   pred_box) = new_coord_scale_boxes(pred_xy, pred_wh, width, height,
                                     anchor_grid, grid_points, max_delta,
                                     scale_xy)

  def delta(dy_scaler, dy_scaled, dy):
    # dy_scaled *= scaler
    dy_xy, dy_wh = tf.split(dy, 2, axis=-1)
    dy_xy_, dy_wh_ = tf.split(dy_scaled, 2, axis=-1)

    # # apply scaling for gradients if scaled boxes are
    # sc_xy_, sc_wh_ = tf.split(scaler, 2, axis=-1)
    # dy_xy_ *= sc_xy_
    # dy_wh_ *= anchor_grid

    # add all the gradients that may have been applied to the
    # boxes and those that have been applied to the width and height
    dy_wh += dy_wh_
    dy_xy += dy_xy_

    # apply the gradient clipping to xy and wh
    dy_wh = math_ops.rm_nan_inf(dy_wh)
    delta = tf.cast(max_delta, dy_wh.dtype)
    dy_wh = tf.clip_by_value(dy_wh, -delta, delta)

    dy_xy = math_ops.rm_nan_inf(dy_xy)
    delta = tf.cast(max_delta, dy_xy.dtype)
    dy_xy = tf.clip_by_value(dy_xy, -delta, delta)
    return dy_xy, dy_wh, 0.0, 0.0, tf.zeros_like(anchor_grid), tf.zeros_like(
        grid_points), 0.0, 0.0, 0.0

  return (scaler, scaled_box, pred_box), delta


def get_predicted_box_newcords(width,
                               height,
                               unscaled_box,
                               anchor_grid,
                               grid_points,
                               scale_xy,
                               darknet=False,
                               max_delta=5.0,
                               normalizer=1.0):
  pred_xy = unscaled_box[..., 0:2]
  pred_wh = unscaled_box[..., 2:4]

  if darknet:
    # if we are using the darknet loss we shoud nto propagate the decoding
    # of the box
    (scaler, scaled_box,
     pred_box) = darknet_new_coord_boxes(pred_xy, pred_wh, width, height,
                                         anchor_grid, grid_points, max_delta,
                                         scale_xy, normalizer)
  else:
    # if we are using the scaled loss we should propagate the decoding of the
    # boxes
    (scaler, scaled_box,
     pred_box) = new_coord_scale_boxes(pred_xy, pred_wh, width, height,
                                       anchor_grid, grid_points, max_delta,
                                       scale_xy)
  return (scaler, scaled_box, pred_box)


class Yolo_Loss(object):

  def __init__(self,
               classes,
               mask,
               anchors,
               scale_anchors=1,
               ignore_thresh=0.7,
               truth_thresh=1.0,
               loss_type="ciou",
               iou_normalizer=1.0,
               cls_normalizer=1.0,
               obj_normalizer=1.0,
               objectness_smooth=True,
               use_scaled_loss=False,
               darknet=None,
               label_smoothing=0.0,
               new_cords=False,
               scale_x_y=1.0,
               max_delta=10,
               **kwargs):
    """
    parameters for the loss functions used at each detection head output
    Args:
      classes: `int` for the number of classes 
      mask: `List[int]` for the output level that this specific model output 
        level
      anchors: `List[List[int]]` for the anchor boxes that are used in the model 
        at all levels
      scale_anchors: `int` for how much to scale this level to get the orginal 
        input shape
      ignore_thresh: `float` for the IOU value over which the loss is not 
        propagated, and a detection is assumed to have been made 
      truth_thresh: `float` for the IOU value over which the loss is propagated 
        despite a detection being made 
      loss_type: `str` for the typeof iou loss to use with in {ciou, diou, 
        giou, iou}
      iou_normalizer: `float` for how much to scale the loss on the IOU or the 
        boxes
      cls_normalizer: `float` for how much to scale the loss on the classes
      obj_normalizer: `float` for how much to scale loss on the detection map
      objectness_smooth: `float` for how much to smooth the loss on the 
        detection map 
      use_reduction_sum: `bool` for whether to use the scaled loss 
        or the traditional loss
      label_smoothing: `float` for how much to smooth the loss on the classes
      new_cords: `bool` for which scaling type to use 
      scale_xy: dictionary `float` values inidcating how far each pixel can see 
        outside of its containment of 1.0. a value of 1.2 indicates there is a 
        20% extended radius around each pixel that this specific pixel can 
        predict values for a center at. the center can range from 0 - value/2 
        to 1 + value/2, this value is set in the yolo filter, and resused here. 
        there should be one value for scale_xy for each level from min_level to 
        max_level
      max_delta: gradient clipping to apply to the box loss 

    Return:
      loss: `float` for the actual loss
      box_loss: `float` loss on the boxes used for metrics
      conf_loss: `float` loss on the confidence used for metrics
      class_loss: `float` loss on the classes used for metrics
      avg_iou: `float` metric for the average iou between predictions 
        and ground truth
      avg_obj: `float` metric for the average confidence of the model 
        for predictions
      recall50: `float` metric for how accurate the model is
      precision50: `float` metric for how precise the model is
    """

    if loss_type == "giou":
      self._loss_type = 1
    elif loss_type == "ciou":
      self._loss_type = 2
    else:
      self._loss_type = 0

    self._classes = tf.constant(tf.cast(classes, dtype=tf.int32))
    self._num = tf.cast(len(mask), dtype=tf.int32)
    self._truth_thresh = truth_thresh
    self._ignore_thresh = ignore_thresh
    self._masks = mask
    self._anchors = anchors

    self._iou_normalizer = iou_normalizer
    self._cls_normalizer = cls_normalizer
    self._obj_normalizer = obj_normalizer
    self._scale_x_y = scale_x_y
    self._max_delta = max_delta

    self._label_smoothing = tf.cast(label_smoothing, tf.float32)
    self._objectness_smooth = float(objectness_smooth)
    self._use_reduction_sum = use_scaled_loss
    self._darknet = darknet

    self._new_cords = new_cords
    self._any = True

    self._anchor_generator = GridGenerator(
        masks=mask, anchors=anchors, scale_anchors=scale_anchors)

    box_kwargs = dict(
        scale_xy=self._scale_x_y,
        darknet=not self._use_reduction_sum,
        normalizer=self._iou_normalizer,
        max_delta=self._max_delta)

    if not self._new_cords:
      self._decode_boxes = partial(get_predicted_box, **box_kwargs)
    else:
      self._decode_boxes = partial(get_predicted_box_newcords, **box_kwargs)

  def print_error(self, pred, key):
    # a catch all to indicate an error may have occured in training
    if tf.stop_gradient(tf.reduce_any(tf.math.is_nan(pred))):
      tf.print("\nerror: stop training ", key)

  def APAR(self, pred_conf, true_conf, pct=0.5):
    # capture all predictions of high confidence
    dets = tf.cast(tf.squeeze(pred_conf, axis=-1) > pct, dtype=true_conf.dtype)

    # compute the total number of true positive predictions
    true_pos = tf.reduce_sum(true_conf * dets, axis=(1, 2, 3))
    # compute the total number of poitives
    gt_pos = tf.reduce_sum(true_conf, axis=(1, 2, 3))
    # compute the total number of predictions, positve and negative
    all_pos = tf.reduce_sum(dets, axis=(1, 2, 3))

    # compute total recall = true_predicitons/ground_truth
    recall = tf.reduce_mean(math_ops.divide_no_nan(true_pos, gt_pos))
    # compute total precision = true_predicitons/total_predictions
    precision = tf.reduce_mean(math_ops.divide_no_nan(true_pos, all_pos))
    return tf.stop_gradient(recall), tf.stop_gradient(precision)

  def avgiou(self, iou):
    # compute the average realtive to non zero locations, so the
    # average is not biased in sparse tensors
    iou_sum = tf.reduce_sum(iou, axis=tf.range(1, tf.shape(tf.shape(iou))[0]))
    counts = tf.cast(
        tf.math.count_nonzero(
            iou, axis=tf.range(1,
                               tf.shape(tf.shape(iou))[0])), iou.dtype)
    avg_iou = tf.reduce_mean(math_ops.divide_no_nan(iou_sum, counts))
    return tf.stop_gradient(avg_iou)

  def box_loss(self, true_box, pred_box, darknet=False):
    # based on the type of loss, compute the iou loss for a box
    # compute_<name> indicated the type of iou to use
    if self._loss_type == 1:
      iou, liou = box_ops.compute_giou(true_box, pred_box, darknet=darknet)
    elif self._loss_type == 2:
      iou, liou = box_ops.compute_ciou(true_box, pred_box, darknet=darknet)
      # iou = liou = box_ops.bbox_iou(true_box, pred_box, x1y1x2y2 = False, CIoU=True)
    else:
      iou = box_ops.compute_iou(true_box, pred_box)
      liou = iou
    # compute the inverse of iou or the value you plan to
    # minimize as 1 - iou_type
    loss_box = 1 - liou
    return iou, liou, loss_box

  def _build_mask_body(self, pred_boxes_, pred_classes_, pred_conf,
                       pred_classes_max, boxes, classes, iou_max_, ignore_mask_,
                       conf_loss_, loss_, count, idx):

    # capture the batch size to be used, and gather a slice of
    # boxes from the ground truth. currently TILE_SIZE = 50, to
    # save memory
    batch_size = tf.shape(boxes)[0]
    box_slice = tf.slice(boxes, [0, idx * TILE_SIZE, 0],
                         [batch_size, TILE_SIZE, 4])

    # match the dimentions of the slice to the model predictions
    # shape: [batch_size, 1, 1, num, TILE_SIZE, 4]
    box_slice = tf.expand_dims(box_slice, axis=-2)
    box_slice = tf.expand_dims(box_slice, axis=1)
    box_slice = tf.expand_dims(box_slice, axis=1)

    # if we are not comparing specific classes we do not
    # need this code
    if not self._any:
      # repeat the steps applied to the boxes for the classes
      # shape: [batch_size, 1, 1, num,  TILE_SIZE, 1]
      class_slice = tf.slice(classes, [0, idx * TILE_SIZE],
                             [batch_size, TILE_SIZE])
      class_slice = tf.expand_dims(class_slice, axis=1)
      class_slice = tf.expand_dims(class_slice, axis=1)
      class_slice = tf.expand_dims(class_slice, axis=1)
      # shape: [batch_size, 1, 1, num,  TILE_SIZE, num_classes]
      class_slice = tf.one_hot(
          tf.cast(class_slice, tf.int32),
          depth=tf.shape(pred_classes_max)[-1],
          dtype=pred_classes_max.dtype)

    # compute the iou between all predictions and ground truths for the
    # purpose of comparison
    pred_boxes = tf.expand_dims(pred_boxes_, axis=-3)
    iou, liou, _ = self.box_loss(box_slice, pred_boxes)

    # mask off zero boxes from the grount truth
    mask = tf.cast(tf.reduce_sum(tf.abs(box_slice), axis=-1) > 0.0, iou.dtype)
    iou *= mask

    # capture all instances where the boxes iou with ground truth is larger
    # than the ignore threshold, all these instances will be ignored
    # unless the grount truth indicates that a box exists here
    iou_mask = iou > self._ignore_thresh
    iou_mask = tf.transpose(iou_mask, perm=(0, 1, 2, 4, 3))

    # compute the matched classes, if any is set to true, we don't care
    # which class was predicted, but only that a class was predicted.
    # other wise we also compare that both the classes and the boxes
    # have matched
    if self._any:
      matched_classes = tf.cast(pred_classes_max, tf.bool)
    else:
      matched_classes = tf.equal(class_slice, pred_classes_max)
      matched_classes = tf.logical_and(
          matched_classes, tf.cast(class_slice, matched_classes.dtype))

    # build the full ignore mask, by taking the logical and of the
    # class and iou masks
    matched_classes = tf.reduce_any(matched_classes, axis=-1)
    full_iou_mask = tf.logical_and(iou_mask, matched_classes)

    # update the tensor with ignore locations
    iou_mask = tf.reduce_any(full_iou_mask, axis=-1, keepdims=False)
    ignore_mask_ = tf.logical_or(ignore_mask_, iou_mask)

    # if object ness smoothing is set to true we also update a
    # tensor with the maximum matchin iou ateach index
    if self._objectness_smooth:
      iou_max = tf.transpose(iou, perm=(0, 1, 2, 4, 3))
      iou_max = iou_max * tf.cast(full_iou_mask, iou_max.dtype)
      iou_max = tf.reduce_max(iou_max, axis=-1, keepdims=False)
      iou_max_ = tf.maximum(iou_max, iou_max_)

    return (pred_boxes_, pred_classes_, pred_conf, pred_classes_max, boxes,
            classes, iou_max_, ignore_mask_, conf_loss_, loss_, count, idx + 1)

  def _tiled_global_box_search(self,
                               pred_boxes,
                               pred_classes,
                               pred_conf,
                               boxes,
                               classes,
                               true_conf,
                               fwidth,
                               fheight,
                               smoothed,
                               scale=None):

    # compute the number of boxes and the total number of tiles for the search
    num_boxes = tf.shape(boxes)[-2]
    num_tiles = num_boxes // TILE_SIZE

    # convert the grount truth boxes to the model output format
    boxes = box_ops.yxyx_to_xcycwh(boxes)

    if scale is not None:
      boxes = boxes * tf.stop_gradient(scale)

    # store once the predicted classes with a high confidence, greater
    # than 25%
    pred_classes_mask = tf.cast(pred_classes > 0.25, tf.float32)
    pred_classes_mask = tf.expand_dims(pred_classes_mask, axis=-2)

    # base tensors that we will update in the while loops
    base = tf.cast(tf.zeros_like(tf.reduce_sum(pred_boxes, axis=-1)), tf.bool)
    loss_base = tf.zeros_like(tf.reduce_sum(pred_boxes, axis=-1))
    obns_base = tf.zeros_like(tf.reduce_sum(pred_boxes, axis=-1, keepdims=True))

    def _loop_cond(pred_boxes_, pred_classes_, pred_conf, pred_classes_max,
                   boxes, classes, iou_max_, ignore_mask_, conf_loss_, loss_,
                   count, idx):
      # check that the slice has boxes that all zeros
      batch_size = tf.shape(boxes)[0]
      box_slice = tf.slice(boxes, [0, idx * TILE_SIZE, 0],
                           [batch_size, TILE_SIZE, 4])

      # confirm that the index is less than the total tiles that the
      # slice has values
      return tf.logical_and(idx < num_tiles,
                            tf.math.greater(tf.reduce_sum(box_slice), 0))

    # compute the while loop
    (_, _, _, _, _, _, iou_max, iou_mask, obns_loss, truth_loss, count,
     idx) = tf.while_loop(
         _loop_cond,
         self._build_mask_body, [
             pred_boxes, pred_classes, pred_conf, pred_classes_mask, boxes,
             classes, loss_base, base, obns_base, loss_base, obns_base,
             tf.constant(0)
         ],
         parallel_iterations=20)

    # build the final ignore mask
    ignore_mask = tf.logical_not(iou_mask)
    ignore_mask = tf.stop_gradient(tf.cast(ignore_mask, true_conf.dtype))
    iou_max = tf.stop_gradient(iou_max)

    # depending on smoothed vs not smoothed the build the mask and ground truth
    # map to use
    if not smoothed:
      # higher map lower iou with ground truth
      obj_mask = true_conf + (1 - true_conf) * ignore_mask
    else:
      # lower map, very high iou with ground truth
      obj_mask = tf.ones_like(true_conf)
      iou_ = (1 - self._objectness_smooth) + self._objectness_smooth * iou_max
      iou_ = tf.where(iou_max > 0, iou_, tf.zeros_like(iou_))

      # update the true conffidence mask with the best matching iou
      true_conf = tf.where(iou_mask, iou_, true_conf)
      # true_conf = iou_

    # stop gradient on all components to save resources, we don't
    # need to track the gradient though the while loop as they are
    # not used
    obj_mask = tf.stop_gradient(obj_mask)
    true_conf = tf.stop_gradient(true_conf)
    obns_loss = tf.stop_gradient(obns_loss)
    truth_loss = tf.stop_gradient(truth_loss)
    count = tf.stop_gradient(count)
    return ignore_mask, obns_loss, truth_loss, count, true_conf, obj_mask

  def build_grid(self, indexes, truths, preds, ind_mask, update=False):
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

  def call_scaled(self, true_counts, inds, y_true, boxes, classes, y_pred):
    # 0. generate shape constants using tf.shat to support feature multi scale
    # training
    shape = tf.shape(true_counts)
    batch_size, width, height, num = shape[0], shape[1], shape[2], shape[3]
    fwidth = tf.cast(width, tf.float32)
    fheight = tf.cast(height, tf.float32)

    # 1. cast all input compontnts to float32 and stop gradient to save memory
    y_true = tf.cast(y_true, tf.float32)
    true_counts = tf.cast(true_counts, tf.float32)
    true_conf = tf.clip_by_value(true_counts, 0.0, 1.0)
    grid_points, anchor_grid = self._anchor_generator(
        width, height, batch_size, dtype=tf.float32)

    # 2. split the y_true grid into the usable items, set the shapes correctly
    #    and save the true_confdence mask before it get altered
    (true_box, ind_mask, true_class, _, _) = tf.split(
        y_true, [4, 1, 1, 1, 1], axis=-1)
    grid_mask = true_conf = tf.squeeze(true_conf, axis=-1)
    true_class = tf.squeeze(true_class, axis=-1)
    num_objs = tf.cast(tf.reduce_sum(ind_mask), dtype=y_pred.dtype)

    # 3. split up the predicitons to match the ground truths shapes
    y_pred = tf.cast(
        tf.reshape(y_pred, [batch_size, width, height, num, -1]), tf.float32)
    pred_box, pred_conf, pred_class = tf.split(y_pred, [4, 1, -1], axis=-1)

    # based on input val new_cords decode the box predicitions
    # and because we are using the scaled loss, do not change the gradients
    # at all
    scale, pred_box, _ = self._decode_boxes(
        fwidth, fheight, pred_box, anchor_grid, grid_points, darknet=False)
    offset = tf.cast(
      tf.gather_nd(grid_points, inds, batch_dims=1), true_box.dtype)
    offset = tf.concat([offset, tf.zeros_like(offset)], axis=-1)
    true_box = apply_mask(ind_mask, (scale * true_box) - offset)
    pred_box = apply_mask(ind_mask, tf.gather_nd(pred_box, inds, batch_dims=1))

    # build the class object
    true_class = tf.one_hot(
        tf.cast(true_class, tf.int32),
        depth=tf.shape(pred_class)[-1],
        dtype=pred_class.dtype)
    true_class = apply_mask(ind_mask, true_class)
    pred_class = apply_mask(ind_mask,
                            tf.gather_nd(pred_class, inds, batch_dims=1))


    # compute the loss of all the boxes and apply a mask such that
    # within the 200 boxes, only the indexes of importance are covered
    _, iou, box_loss = self.box_loss(true_box, pred_box, darknet=False)
    box_loss = apply_mask(tf.squeeze(ind_mask, axis=-1), box_loss)
    box_loss = math_ops.divide_no_nan(tf.reduce_sum(box_loss), num_objs)

    # 6.  (confidence loss) build a selective between the ground truth and the
    #     iou to take only a certain percent of the iou or the ground truth,
    #     i.e smooth the detection map
    #     build a the ground truth detection map
    iou = tf.stop_gradient(iou)
    iou = tf.maximum(iou, 0.0)
    smoothed_iou = ((
        (1 - self._objectness_smooth) * tf.cast(ind_mask, iou.dtype)) +
                    self._objectness_smooth * tf.expand_dims(iou, axis=-1))
    smoothed_iou = apply_mask(ind_mask, smoothed_iou)
    true_conf = self.build_grid(
        inds, smoothed_iou, pred_conf, ind_mask, update=False)
    true_conf = tf.squeeze(true_conf, axis=-1)

    #     compute the detection map loss, there should be no masks
    #     applied
    bce = ks.losses.binary_crossentropy(
        K.expand_dims(true_conf, axis=-1), pred_conf, from_logits=True)
    conf_loss = tf.reduce_mean(bce)

    # 7.  (class loss) build the one hot encoded true class values
    #     compute the loss on the classes, apply the same inds mask
    #     and the compute the average of all the values
    class_loss = ks.losses.binary_crossentropy(
        true_class,
        pred_class,
        label_smoothing=self._label_smoothing,
        from_logits=True)
    class_loss = apply_mask(tf.squeeze(ind_mask, axis=-1), class_loss)
    class_loss = math_ops.divide_no_nan(tf.reduce_sum(class_loss), num_objs)

    # 8. apply the weights to each loss
    box_loss *= self._iou_normalizer 
    class_loss *= self._cls_normalizer 
    conf_loss *= self._obj_normalizer 

    # 9. add all the losses together then take the sum over the batches
    mean_loss = box_loss + class_loss + conf_loss
    loss = mean_loss * tf.cast(batch_size, mean_loss.dtype)

    # 4. apply sigmoid to items and use the gradient trap to contol the backprop
    #    and selective gradient clipping
    sigmoid_conf = tf.stop_gradient(tf.sigmoid(pred_conf))

    # 10. compute all the values for the metrics
    recall50, precision50 = self.APAR(sigmoid_conf, grid_mask, pct=0.5)
    avg_iou = self.avgiou(apply_mask(tf.squeeze(ind_mask, axis=-1), iou))
    avg_obj = self.avgiou(tf.squeeze(sigmoid_conf, axis=-1) * grid_mask)
    return (loss, box_loss, conf_loss, class_loss, mean_loss, avg_iou, avg_obj,
            recall50, precision50)

  def call_darknet(self, true_counts, inds, y_true, boxes, classes, y_pred):
    # 0. if smoothign is used, they prop the gradient of the sigmoid first
    #    but the sigmoid, if it is not enabled, they do not use the gradient of
    #    the sigmoid
    if self._new_cords:
      # if smoothing is enabled they for some reason
      # take the sigmoid many times
      y_pred = grad_sigmoid(y_pred)

    # 1. generate and store constants and format output
    shape = tf.shape(true_counts)
    batch_size, width, height, num = shape[0], shape[1], shape[2], shape[3]
    fwidth = tf.cast(width, tf.float32)
    fheight = tf.cast(height, tf.float32)
    grid_points, anchor_grid = self._anchor_generator(
        width, height, batch_size, dtype=tf.float32)

    # 2. cast all input compontnts to float32 and stop gradient to save memory
    boxes = tf.stop_gradient(tf.cast(boxes, tf.float32))
    classes = tf.stop_gradient(tf.cast(classes, tf.float32))
    y_true = tf.stop_gradient(tf.cast(y_true, tf.float32))
    true_counts = tf.stop_gradient(tf.cast(true_counts, tf.float32))
    true_conf = tf.stop_gradient(tf.clip_by_value(true_counts, 0.0, 1.0))
    grid_points = tf.stop_gradient(grid_points)
    anchor_grid = tf.stop_gradient(anchor_grid)

    # 3. split all the ground truths to use as seperate items in loss computation
    (true_box, ind_mask, true_class, _, _) = tf.split(
        y_true, [4, 1, 1, 1, 1], axis=-1)
    true_conf = tf.squeeze(true_conf, axis=-1)
    true_class = tf.squeeze(true_class, axis=-1)
    grid_mask = true_conf

    # 4. splits all predictions to seperate and compute the loss for each map
    #    individually
    y_pred = tf.cast(
        tf.reshape(y_pred, [batch_size, width, height, num, -1]), tf.float32)
    pred_box, pred_conf, pred_class = tf.split(y_pred, [4, 1, -1], axis=-1)

    # 5. compute the sigmoid of the classes and pass the unsigmoided items
    #    through a gradient trap to allow better control over the back
    #    propagation step. the sigmoided class is used for metrics only
    sigmoid_class = tf.stop_gradient(tf.sigmoid(pred_class))
    sigmoid_conf = tf.stop_gradient(tf.sigmoid(pred_conf))

    # 6. decode the boxes to be used for optimization/loss compute
    if self._darknet is None:
      _, _, pred_box = self._decode_boxes(
          fwidth, fheight, pred_box, anchor_grid, grid_points, darknet=True)
      scale = None
    else:
      scale, pred_box, _ = self._decode_boxes(
          fwidth,
          fheight,
          pred_box,
          anchor_grid,
          grid_points,
          darknet=self._darknet)
      true_box = tf.stop_gradient(true_box * scale)

    # 7. compare all the predictions to all the valid or non zero boxes
    #    in the ground truth, based on any/all we will will also compare
    #    classes. This is done to locate pixels where a valid box and class
    #    may have been predicted, but the ground truth may not have placed
    #    a box. For this indexes, the detection map loss will be ignored.
    #    obj_mask dictates the locations where the loss is ignored.
    if self._ignore_thresh != 0.0:
      (_, _, _, _, true_conf, obj_mask) = self._tiled_global_box_search(
          pred_box,
          sigmoid_class,
          sigmoid_conf,
          boxes,
          classes,
          true_conf,
          fwidth,
          fheight,
          smoothed=self._objectness_smooth > 0,
          scale=scale)
    else:
      obj_mask = tf.ones_like(true_conf)

    # 8. compute the one hot class maps that are used for prediction
    #    done in the loss function side to save memory and improve
    #    data pipeline speed
    true_class = tf.one_hot(
        tf.cast(true_class, tf.int32),
        depth=tf.shape(pred_class)[-1],
        dtype=pred_class.dtype)
    true_classes = tf.stop_gradient(apply_mask(ind_mask, true_class))

    # 9. use the update list to build the binary cross entropy label
    #    map. done to allow optimization of many labels to one grid
    #    index
    true_class = self.build_grid(
        inds, true_classes, pred_class, ind_mask, update=False)
    true_class = tf.stop_gradient(true_class)

    # 10. use the class mask to find the number of objects located in
    #     each predicted grid cell/pixel
    counts = true_class
    counts = tf.reduce_sum(counts, axis=-1, keepdims=True)
    reps = tf.gather_nd(counts, inds, batch_dims=1)
    reps = tf.squeeze(reps, axis=-1)
    reps = tf.stop_gradient(tf.where(reps == 0.0, tf.ones_like(reps), reps))

    # 11. gather the boxes of use from the prediction map to compute the
    #     loss on. If the box is gathered then loss is computed, all the other
    #     indexes will get ignored. Use the value reps to compute the average
    #     loss for each pixel. If we have a class mask of [1, 1, 0, 0] with 4
    #     classes at a given pixel (N, M). reps(N, M) = 2, and the box loss at
    #     (N, M), will be the sum of the iou loss of object 1 and object 2
    #     reps(N, M), making it the average loss at each pixel. Take te sum
    #     over all the objects.
    pred_box = apply_mask(ind_mask, tf.gather_nd(pred_box, inds, batch_dims=1))
    iou, liou, box_loss = self.box_loss(true_box, pred_box, darknet=True)
    box_loss = apply_mask(tf.squeeze(ind_mask, axis=-1), box_loss)
    box_loss = math_ops.divide_no_nan(box_loss, reps)
    box_loss = tf.cast(tf.reduce_sum(box_loss, axis=1), dtype=y_pred.dtype)

    # 12. stop the gradient on the loss components to be used as metrics
    iou = tf.stop_gradient(iou)
    liou = tf.stop_gradient(liou)

    # 13. compute the sigmoid binary cross entropy, same as bce with logits but
    #     it is far safer with no holes. gradient reduces to y_pred - y_true.
    class_loss = sigmoid_BCE(
        K.expand_dims(true_class, axis=-1), K.expand_dims(pred_class, axis=-1),
        self._label_smoothing)

    # 14. apply the weight from the configs to the predictions classes
    if self._cls_normalizer < 1.0:
      # we build a mask based on the true class locations
      cls_norm_mask = true_class
      # we only apply the classes weight to class indexes were one_hot is one
      class_loss *= ((1 - cls_norm_mask) + cls_norm_mask * self._cls_normalizer)

    # 15. apply the mask to the class loss and compute the sum over all the
    #     objects
    class_loss = tf.reduce_sum(class_loss, axis=-1)
    class_loss = apply_mask(grid_mask, class_loss)
    class_loss = math_ops.rm_nan_inf(class_loss, val=0.0)
    class_loss = tf.cast(
        tf.reduce_sum(class_loss, axis=(1, 2, 3)), dtype=y_pred.dtype)

    # 16. use the custom sigmoid BCE to compute the loss at each pixel
    #     in the detection map
    bce = sigmoid_BCE(K.expand_dims(true_conf, axis=-1), pred_conf, 0.0)

    # 17. apply the ignore mask to the detection map to zero out all the
    #     indexes where the loss should not be computed. we use the apply
    #     mask function to control the gradient propagation, and garuntee
    #     safe masking with no NANs.
    conf_loss = apply_mask(obj_mask, bce)
    conf_loss = tf.cast(
        tf.reduce_sum(conf_loss, axis=(1, 2, 3)), dtype=y_pred.dtype)

    # NOTE: there may be a way to find the weights by
    #       using a self supervised nerual net. The
    #       neural net determines its own weights

    # 18. apply the fixed weight to each of the losses, the weight is
    #     applied for the box loss in the gradeint for the box decoding
    #     class weights are applied selectively directly after computing
    #     the loss only to locations where a onehot is set to one
    box_loss *= self._iou_normalizer
    conf_loss *= self._obj_normalizer

    # 19. take the sum of the losses for each map, then take the mean
    #     across all the smaples in the batch
    loss = box_loss + class_loss + conf_loss
    loss = tf.reduce_mean(loss)

    # 20. reduce the mean of the losses to use as metrics
    box_loss = tf.reduce_mean(box_loss)
    conf_loss = tf.reduce_mean(conf_loss)
    class_loss = tf.reduce_mean(class_loss)

    # 21. metric compute using the generated values from the loss itself
    #     done here to save time and resources
    recall50, precision50 = self.APAR(sigmoid_conf, grid_mask, pct=0.5)
    avg_iou = self.avgiou(apply_mask(tf.squeeze(ind_mask, axis=-1), iou))
    avg_obj = self.avgiou(tf.squeeze(sigmoid_conf, axis=-1) * grid_mask)
    return (loss, box_loss, conf_loss, class_loss, loss, avg_iou, avg_obj,
            recall50, precision50)

  def __call__(self, true_counts, inds, y_true, boxes, classes, y_pred):
    if self._use_reduction_sum == True:
      return self.call_scaled(true_counts, inds, y_true, boxes, classes, y_pred)
    else:
      return self.call_darknet(true_counts, inds, y_true, boxes, classes,
                               y_pred)

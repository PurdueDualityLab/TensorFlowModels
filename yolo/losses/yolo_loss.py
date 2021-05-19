import tensorflow as tf
import tensorflow.keras as ks
from tensorflow.keras import backend as K

from yolo.ops.loss_utils import GridGenerator
from yolo.ops import box_ops
from yolo.ops import math_ops
import numpy as np

from functools import partial

TILE_SIZE = 50


@tf.custom_gradient
def obj_gradient_trap(y, max_delta=np.inf):
  def trap(dy):
    dy = math_ops.rm_nan_inf(dy)
    delta = tf.cast(max_delta, dy.dtype)
    dy = tf.clip_by_value(dy, -delta, delta)
    return dy, 0.0
  return y, trap


@tf.custom_gradient
def box_gradient_trap(y, max_delta=np.inf):
  def trap(dy):
    dy = math_ops.rm_nan_inf(dy)
    delta = tf.cast(max_delta, dy.dtype)
    dy = tf.clip_by_value(dy, -delta, delta)
    return dy, 0.0
  return y, trap


@tf.custom_gradient
def class_gradient_trap(y, max_delta=np.inf):

  def trap(dy):
    dy = math_ops.rm_nan_inf(dy)
    delta = tf.cast(max_delta, dy.dtype)
    dy = tf.clip_by_value(dy, -delta, delta)
    # tf.print(tf.reduce_sum(tf.square(dy)))
    return dy, 0.0

  return y, trap


def ce(values, labels):
  labels = labels + K.epsilon()
  loss = tf.reduce_mean(
      -math_ops.mul_no_nan(values, tf.math.log(labels)), axis=-1)
  loss = math_ops.rm_nan(loss, val=0.0)
  return loss


@tf.custom_gradient
def grad_sigmoid(values):
  #vals = tf.math.sigmoid(values)
  def delta(dy):
    t = tf.math.sigmoid(values)
    return dy * t * (1 - t)
  return values, delta


@tf.custom_gradient
def sigmoid_BCE(y, x_prime, label_smoothing):
  y = y * (1 - label_smoothing) + 0.5 * label_smoothing
  x = tf.math.sigmoid(x_prime)
  x = math_ops.rm_nan_inf(x, val=0.0)
  #bce = ks.losses.binary_crossentropy(y, x, from_logits=False)
  bce = tf.reduce_sum(tf.square(-y + x), axis=-1)

  def delta(dy):
    # safer and faster gradient compute for bce

    # # bce = -ylog(x) - (1 - y)log(1 - x)
    # # bce derive
    # dloss = -math_ops.divide_no_nan(y, x) + math_ops.divide_no_nan((1-y),(1-x))
    # # 1 / (1 + exp(-x))
    # dsigmoid = x * (1 - x)
    # dx = dloss * dsigmoid * tf.expand_dims(dy, axis = -1)

    # # bce derive
    # dloss = -y * (1 - x) + (1 - y) * x
    # # 1 / (1 + exp(-x))
    # dx = dloss * tf.expand_dims(dy, axis = -1)

    # bce + sigmoid derivative
    dloss = (-y + x) 
    dx = dloss * tf.expand_dims(dy, axis=-1)

    dy = tf.zeros_like(y)
    return dy, dx, 0.0
  return bce, delta


@tf.custom_gradient
def apply_mask(mask, x):
  masked = math_ops.mul_no_nan(mask, x)

  def delta(dy):
    return tf.zeros_like(mask), math_ops.mul_no_nan(mask, dy)

  return masked, delta


# conver to a static class
# no ops in this fn should have grads propagated
def scale_boxes(pred_xy, pred_wh, width, height, anchor_grid, grid_points,
                max_delta, scale_xy):
  scale_xy = tf.cast(scale_xy, pred_xy.dtype)
  pred_xy = tf.math.sigmoid(pred_xy) * scale_xy - 0.5 * (scale_xy - 1)
  # pred_xy = pred_xy * scale_xy - 0.5 * (scale_xy - 1)

  scaler = tf.convert_to_tensor([width, height])
  box_xy = grid_points + pred_xy / scaler
  box_wh = tf.math.exp(pred_wh) * anchor_grid
  pred_box = K.concatenate([box_xy, box_wh], axis=-1)
  # return (box_xy, box_wh, pred_box)
  return (pred_xy, box_wh, pred_box)


@tf.custom_gradient
def darknet_boxes(pred_xy, pred_wh, width, height, anchor_grid, grid_points,
                  max_delta, scale_xy, normalizer):
                  
  (pred_xy, box_wh, pred_box) = scale_boxes(pred_xy, pred_wh, width, height,
                                            anchor_grid, grid_points, max_delta,
                                            scale_xy)

  def delta(dy_xy, dy_wh, dy):
    # here we do not propgate the scaling of the prediction through the network because in back prop is
    # leads to the scaling of the gradient of a box relative to the size of the box and the gradient
    # for small boxes this will mean down scaling the gradient leading to poor convergence on small
    # objects and also very likely over fitting
    dy_xy_, dy_wh_ = tf.split(dy, 2, axis=-1)
    dy_wh += dy_wh_
    dy_xy += dy_xy_
    dy_wh *= tf.math.exp(pred_wh)

    # scaling
    dy_xy *= tf.cast(normalizer, dy_xy.dtype)
    dy_wh *= tf.cast(normalizer, dy_wh.dtype)

    # gradient clipping
    dy_wh = math_ops.rm_nan_inf(dy_wh)
    delta = tf.cast(max_delta, dy_wh.dtype)
    dy_wh = tf.clip_by_value(dy_wh, -delta, delta)

    dy_xy = math_ops.rm_nan_inf(dy_xy)
    delta = tf.cast(max_delta, dy_xy.dtype)
    dy_xy = tf.clip_by_value(dy_xy, -delta, delta)
    return dy_xy, dy_wh, 0.0, 0.0, tf.zeros_like(anchor_grid), tf.zeros_like(
        grid_points), 0.0, 0.0, 0.0

  # return (box_xy, box_wh, pred_box), delta
  return (pred_xy, box_wh, pred_box), delta


def get_predicted_box(width,
                      height,
                      unscaled_box,
                      anchor_grid,
                      grid_points,
                      scale_xy,
                      darknet=True,
                      max_delta=5.0, 
                      normalizer=1.0):

  # TODO: scale_xy should not be propagated in darkent either
  pred_xy = unscaled_box[..., 0:2]  
  # pred_xy = tf.sigmoid(unscaled_box[..., 0:2])  
  pred_wh = unscaled_box[..., 2:4]

  # the gradient for non of this should be propagated
  if darknet:
    # box_xy, box_wh, pred_box = darknet_boxes(pred_xy, pred_wh, width, height, anchor_grid, grid_points, max_delta, scale_xy)
    pred_xy, box_wh, pred_box = darknet_boxes(pred_xy, pred_wh, width, height,
                                              anchor_grid, grid_points,
                                              max_delta, scale_xy, normalizer)
  else:
    # box_xy, box_wh, pred_box = scale_boxes(pred_xy, pred_wh, width, height, anchor_grid, grid_points, max_delta, scale_xy)
    pred_xy, box_wh, pred_box = scale_boxes(pred_xy, pred_wh, width, height,
                                            anchor_grid, grid_points, max_delta,
                                            scale_xy)

  return pred_xy, box_wh, pred_box


# conver to a static class
# no ops in this fn should have grads propagated
def new_coord_scale_boxes(pred_xy, pred_wh, width, height, anchor_grid,
                          grid_points, max_delta, scale_xy):
  scale_xy = tf.cast(scale_xy, pred_xy.dtype)

  pred_xy = tf.math.sigmoid(pred_xy)  
  pred_wh = tf.math.sigmoid(pred_wh)
  pred_xy = pred_xy * scale_xy - 0.5 * (scale_xy - 1)
  scaler = tf.convert_to_tensor([width, height])
  box_xy = grid_points + pred_xy / scaler
  box_wh = tf.square(2 * pred_wh) * anchor_grid
  pred_box = K.concatenate([box_xy, box_wh], axis=-1)
  # return (box_xy, box_wh, pred_box)
  return (pred_xy, box_wh, pred_box)


@tf.custom_gradient
def darknet_new_coord_boxes(pred_xy, pred_wh, width, height, anchor_grid,
                            grid_points, max_delta, scale_xy, normalizer):
  # (box_xy, box_wh, pred_box) = new_coord_scale_boxes(pred_xy, pred_wh, width, height, anchor_grid, grid_points, max_delta, scale_xy)
  (pred_xy, box_wh, pred_box) = new_coord_scale_boxes(pred_xy, pred_wh, width,
                                                      height, anchor_grid,
                                                      grid_points, max_delta,
                                                      scale_xy)

  def delta(dy_xy, dy_wh, dy):
    dy_xy_, dy_wh_ = tf.split(dy, 2, axis=-1)
    dy_wh += dy_wh_
    dy_xy += dy_xy_

    # scaling
    dy_xy *= tf.cast(normalizer, dy_xy.dtype)
    dy_wh *= tf.cast(normalizer, dy_wh.dtype)

    # gradient clipping
    dy_wh = math_ops.rm_nan_inf(dy_wh)
    delta = tf.cast(max_delta, dy_wh.dtype)
    dy_wh = tf.clip_by_value(dy_wh, -delta, delta)

    dy_xy = math_ops.rm_nan_inf(dy_xy)
    delta = tf.cast(max_delta, dy_xy.dtype)
    dy_xy = tf.clip_by_value(dy_xy, -delta, delta)
    return dy_xy, dy_wh, 0.0, 0.0, tf.zeros_like(anchor_grid), tf.zeros_like(
        grid_points), 0.0, 0.0, 0.0

  # return (box_xy, box_wh, pred_box), delta
  return (pred_xy, box_wh, pred_box), delta


def get_predicted_box_newcords(width,
                               height,
                               unscaled_box,
                               anchor_grid,
                               grid_points,
                               scale_xy,
                               darknet=False,
                               max_delta=5.0, 
                               normalizer=1.0):
  # pred_xy = tf.math.sigmoid(unscaled_box[..., 0:2])  
  # pred_wh = tf.math.sigmoid(unscaled_box[..., 2:4])
  pred_xy = unscaled_box[..., 0:2]  
  pred_wh = unscaled_box[..., 2:4]

  if darknet:
    # box_xy, box_wh, pred_box = darknet_new_coord_boxes(pred_xy, pred_wh, width, height, anchor_grid, grid_points, max_delta, scale_xy)
    pred_xy, box_wh, pred_box = darknet_new_coord_boxes(pred_xy, pred_wh, width,
                                                        height, anchor_grid,
                                                        grid_points, max_delta,
                                                        scale_xy, normalizer)
  else:
    # pred_xy_ = pred_xy
    # scale_xy = tf.cast(scale_xy, pred_xy.dtype)
    # pred_xy = pred_xy * scale_xy - 0.5 * (scale_xy - 1)
    # box_xy, box_wh, pred_box = new_coord_scale_boxes(pred_xy_, pred_wh, width, height, anchor_grid, grid_points, max_delta, scale_xy)
    
    pred_xy = grad_sigmoid(pred_xy)
    pred_wh = grad_sigmoid(pred_wh)
    pred_xy, box_wh, pred_box = new_coord_scale_boxes(pred_xy, pred_wh, width,
                                                      height, anchor_grid,
                                                      grid_points, max_delta,
                                                      scale_xy)
  return pred_xy, box_wh, pred_box


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
               use_reduction_sum=False,
               label_smoothing=0.0,
               new_cords=False,
               scale_x_y=1.0,
               max_delta=10,

               use_tie_breaker=True,
               nms_kind="greedynms",
               beta_nms=0.6,
               path_key=None,
               iou_thresh=0.213,
               name=None,

               **kwargs):
    """
    parameters for the loss functions used at each detection head output
    Args:
      classes: `int` for the number of classes 
      mask: `List[int]` for the output level that this specific model output 
        level
      anchors: `List[List[int]]` for the anchor boxes that are used in the model at all levels
      scale_anchors: `int` for how much to scale this level to get the orginal input shape
      ignore_thresh: `float` for the IOU value over which the loss is not propagated, and a detection is assumed to have been made 
      truth_thresh: `float` for the IOU value over which the loss is propagated despite a detection being made 
      loss_type: `str` for the typeof iou loss to use with in {ciou, diou, giou, iou}
      iou_normalizer: `float` for how much to scale the loss on the IOU or the boxes
      cls_normalizer: `float` for how much to scale the loss on the classes
      obj_normalizer: `float` for how much to scale the loss on the detectoion map
      objectness_smooth: `float` for how much to smooth the loss on the detection map 
      use_reduction_sum: `bool` for whether to use the scaled loss or the traditional loss
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
      
      iou_thresh: rmv
      nms_kind: rmv
      beta_nms: rmv
      path_key: rmv
      use_tie_breaker: rmv
      
      name=None,

    Return:
      loss: `float` for the actual loss
      box_loss: `float` loss on the boxes used for metrics
      conf_loss: `float` loss on the confidence used for metrics
      class_loss: `float` loss on the classes used for metrics
      avg_iou: `float` metric for the average iou between predictions and ground truth
      avg_obj: `float` metric for the average confidence of the model for predictions
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
    self._use_reduction_sum = use_reduction_sum

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
    true_pos = tf.reduce_sum(
        math_ops.mul_no_nan(true_conf, dets), axis=(1, 2, 3))
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
    avg_iou = math_ops.divide_no_nan(
        tf.reduce_sum(iou),
        tf.cast(
            tf.math.count_nonzero(tf.cast(iou, dtype=iou.dtype)),
            dtype=iou.dtype))
    return tf.stop_gradient(avg_iou)

  def box_loss(self, true_box, pred_box, darknet=False):
    # based on the type of loss, compute the iou loss for a box
    # compute_<name> indicated the type of iou to use
    if self._loss_type == 1:
      iou, liou = box_ops.compute_giou(true_box, pred_box, darknet=darknet)
    elif self._loss_type == 2:
      iou, liou = box_ops.compute_ciou(true_box, pred_box, darknet=darknet)
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
    mask = tf.cast(tf.reduce_sum(tf.abs(box_slice), axis = -1) > 0.0, iou.dtype)
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
      matched_classes = tf.logical_and(matched_classes,
                                      tf.cast(class_slice, matched_classes.dtype))
    
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

  def _tiled_global_box_search(self, pred_boxes, pred_classes, pred_conf, boxes,
                               classes, true_conf, fwidth, fheight, smoothed):
    
    # compute the number of boxes and the total number of tiles for the search
    num_boxes = tf.shape(boxes)[-2]
    num_tiles = num_boxes // TILE_SIZE

    # convert the grount truth boxes to the model output format
    boxes = box_ops.yxyx_to_xcycwh(boxes)

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

    # find all the batch indexes using the cumulated sum of a ones tensor
    # cumsum(ones) - 1 yeild the zero indexed batches
    bhep = tf.reduce_max(tf.ones_like(indexes), axis=-1, keepdims=True)
    bhep = tf.math.cumsum(bhep, axis=0) - 1

    # concatnate the batch sizes to the indexes 
    indexes = tf.concat([bhep, indexes], axis=-1)
    indexes = apply_mask(tf.cast(ind_mask, indexes.dtype), indexes)

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
      # clip the values between zero and one 
      grid = tf.clip_by_value(grid, 0.0, 1.0)
    
    # stop gradient and return to avoid TPU errors and save compute 
    # resources
    return tf.stop_gradient(grid)

  def _scale_ground_truth_box(self, true_box, inds, ind_mask, fheight, fwidth):
    # used to scale up the groun truth boxes to the shape of the current output 
    # in the scaled yolo loss 
    ind_y, ind_x, ind_a = tf.split(inds, 3, axis=-1)
    ind_zero = tf.zeros_like(ind_x)

    # build out the indexes and the how much all the values must be shifted 
    ind_shift = tf.concat([ind_x, ind_y, ind_zero, ind_zero], axis=-1)
    ind_shift = tf.cast(ind_shift, true_box.dtype)

    # build a scaling tensor
    scale = tf.convert_to_tensor([fwidth, fheight, fwidth, fheight])

    # shift and scale the boxes 
    true_box = (true_box * scale) - ind_shift

    # mask off any of the shorting and scaling that may hav occured to 
    # any all zero boxes
    true_box = apply_mask(ind_mask, true_box)

    # stop gradient and return to avoid TPU errors and save compute 
    # resources
    return tf.stop_gradient(true_box)


  def call_scaled(self, true_counts, inds, y_true, boxes, classes, y_pred):
    # 0. generate shape constants using tf.shat to support feature multi scale
    # training
    shape = tf.shape(true_counts)
    batch_size, width, height, num = shape[0], shape[1], shape[2], shape[3]
    fwidth = tf.cast(width, tf.float32)
    fheight = tf.cast(height, tf.float32)

    # 1. cast all input compontnts to float32 and stop gradient to save memory
    boxes = tf.stop_gradient(tf.cast(boxes, tf.float32))
    classes = tf.stop_gradient(tf.cast(classes, tf.float32))
    y_true = tf.stop_gradient(tf.cast(y_true, tf.float32))
    true_counts = tf.stop_gradient(tf.cast(true_counts, tf.float32))
    true_conf = tf.stop_gradient(tf.clip_by_value(true_counts, 0.0, 1.0))
    grid_points, anchor_grid = self._anchor_generator(
        width, height, batch_size, dtype=tf.float32)

    # 2. split the y_true grid into the usable items, set the shapes correctly 
    #    and save the true_confdence mask before it get altered
    (true_box, ind_mask, true_class, _, _) = tf.split(
        y_true, [4, 1, 1, 1, 1], axis=-1)
    true_conf = tf.stop_gradient(tf.squeeze(true_conf, axis=-1))
    true_class = tf.stop_gradient(tf.squeeze(true_class, axis=-1))
    grid_mask = tf.stop_gradient(true_conf)

    # 3. split up the predicitons to match the ground truths shapes
    y_pred = tf.cast(
        tf.reshape(y_pred, [batch_size, width, height, num, -1]), tf.float32)
    pred_box, pred_conf, pred_class = tf.split(y_pred, [4, 1, -1], axis=-1)

    # 4. apply sigmoid to items and use the gradient trap to contol the back prop 
    #    and selective gradient clipping 
    # sigmoid_class = tf.sigmoid(pred_class)
    sigmoid_conf = tf.sigmoid(pred_conf)
    
    # 5. based on input val new_cords decode the box predicitions and because 
    #    we are using the scaled loss, do not change the gradients at all  
    pred_xy, pred_wh, pred_box = self._decode_boxes(fwidth, fheight, pred_box,
                                                    anchor_grid, grid_points, 
                                                    darknet=False)

    # 6. find out the number of points placed in the grid mask                                         
    num_objs = tf.cast(
        tf.reduce_sum(grid_mask, axis=(1, 2, 3)), dtype=y_pred.dtype)
    # num_objs = tf.cast(tf.reduce_sum(ind_mask, axis=(1, 2)), dtype=y_pred.dtype)

    # 7. build the one hot encoded true class values 
    true_class = tf.one_hot(
        tf.cast(true_class, tf.int32),
        depth=tf.shape(pred_class)[-1],
        dtype=pred_class.dtype)
    true_class = math_ops.mul_no_nan(ind_mask, true_class)

    # 8. scale the boxes properly in order to also scale the gradeints in backprop 
    scale = tf.convert_to_tensor([fheight, fwidth])
    pred_wh = pred_wh * scale
    pred_box = tf.concat([pred_xy, pred_wh], axis=-1)
    true_box = self._scale_ground_truth_box(true_box, inds, ind_mask, fheight, fwidth)

    # 9. gather all the indexes that a loss should be computed at also stop the 
    #    gradient on grount truths to save memory
    pred_box = math_ops.mul_no_nan(ind_mask,
                                   tf.gather_nd(pred_box, inds, batch_dims=1))
    true_box = tf.stop_gradient(math_ops.mul_no_nan(ind_mask, true_box))

    # 10. compute the loss of all the boxes and apply a mask such that 
    #     within the 200 boxes, only the indexes of importance are covered 
    iou, _, box_loss = self.box_loss(true_box, pred_box, darknet=False)
    box_loss = apply_mask(tf.squeeze(ind_mask, axis=-1), box_loss)
    box_loss = tf.cast(tf.reduce_sum(box_loss, axis=1), dtype=y_pred.dtype)
    box_loss = math_ops.divide_no_nan(box_loss, num_objs)
    
    # 11. build a selective between the ground truth and the iou to take only a 
    #     certain percent of the iou or the ground truth, i.e smooth the detection 
    #     map
    smoothed_iou = (((1 - self._objectness_smooth) * tf.cast(ind_mask, iou.dtype)) + self._objectness_smooth * tf.expand_dims(iou, axis=-1))
    smoothed_iou = math_ops.mul_no_nan(ind_mask, smoothed_iou)

    # 12. build a the ground truth detection map 
    true_conf = self.build_grid(inds, smoothed_iou, pred_conf, ind_mask, update=False)
    true_conf = tf.squeeze(true_conf, axis=-1)

    # 13. apply the mask for the classes to again use only the indexes where a 
    #     box exists
    pred_class = apply_mask(
        ind_mask, tf.gather_nd(pred_class, inds, batch_dims=1))

    # 14. compute the loss on the classes, apply the same inds mask 
    #     and the compute the average of all the values
    class_loss = ks.losses.binary_crossentropy(
        K.expand_dims(true_class, axis=-1),
        K.expand_dims(pred_class, axis=-1),
        label_smoothing=self._label_smoothing,
        from_logits=True)
    class_loss = apply_mask(ind_mask, class_loss)
    class_loss = tf.cast(
        tf.reduce_mean(class_loss, axis=(2)), dtype=y_pred.dtype)
    class_loss = tf.cast(
        tf.reduce_sum(class_loss, axis=(1)), dtype=y_pred.dtype)
    class_loss = math_ops.divide_no_nan(class_loss, num_objs)

    # 15. compute the detection map loss, there should be no masks 
    #     applied
    bce = ks.losses.binary_crossentropy(
        K.expand_dims(true_conf, axis=-1), pred_conf, from_logits=True)
    conf_loss = tf.cast(
        tf.reduce_mean(bce, axis=(1, 2, 3)), dtype=y_pred.dtype)

    # 16. apply the weights to each loss
    box_loss *= self._iou_normalizer
    class_loss *= self._cls_normalizer
    conf_loss *= self._obj_normalizer

    # 17. add all the losses together then take the sum over the batches 
    loss = box_loss + class_loss + conf_loss
    loss = tf.reduce_sum(loss)

    # 18. compute all the individual losses to use as metrics
    box_loss = tf.reduce_sum(box_loss)
    conf_loss = tf.reduce_sum(conf_loss)
    class_loss = tf.reduce_sum(class_loss)

    # 19. compute all the values for the metrics
    recall50, precision50 = self.APAR(sigmoid_conf, grid_mask, pct=0.5)
    avg_iou = self.avgiou(iou * tf.gather_nd(grid_mask, inds, batch_dims=1))
    avg_obj = self.avgiou(tf.squeeze(sigmoid_conf, axis=-1) * grid_mask)
    return loss, box_loss, conf_loss, class_loss, avg_iou, avg_obj, recall50, precision50

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

    # 2. cast all input compontnts to float32 and stop gradient to save memory
    boxes = tf.stop_gradient(tf.cast(boxes, tf.float32))
    classes = tf.stop_gradient(tf.cast(classes, tf.float32))
    y_true = tf.stop_gradient(tf.cast(y_true, tf.float32))
    true_counts = tf.stop_gradient(tf.cast(true_counts, tf.float32))
    true_conf = tf.stop_gradient(tf.clip_by_value(true_counts, 0.0, 1.0))
    grid_points, anchor_grid = self._anchor_generator(
        width, height, batch_size, dtype=tf.float32)

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
    sigmoid_class = tf.sigmoid(pred_class)
    pred_class = class_gradient_trap(pred_class, np.inf)
    sigmoid_conf = tf.sigmoid(pred_conf)
    sigmoid_conf = math_ops.rm_nan_inf(sigmoid_conf, val=0.0)
    pred_conf = obj_gradient_trap(pred_conf, np.inf)

    # 6. decode the boxes to be used for optimization/loss compute
    _, _, pred_box = self._decode_boxes(fwidth, fheight, pred_box,
                                                    anchor_grid, grid_points, darknet=True)

    # 7. compare all the predictions to all the valid or non zero boxes 
    #    in the ground truth, based on any/all we will will also compare 
    #    classes. This is done to locate pixels where a valid box and class 
    #    may have been predicted, but the ground truth may not have placed 
    #    a box. For this indexes, the detection map loss will be ignored. 
    #    obj_mask dictates the locations where the loss is ignored. 
    (_, _, _, _, true_conf, obj_mask) = self._tiled_global_box_search(
         pred_box,
         sigmoid_class,
         sigmoid_conf,
         boxes,
         classes,
         true_conf,
         fwidth,
         fheight,
         smoothed=self._objectness_smooth > 0)

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
    #     (N, M), will be the sum of the iou loss of object 1 and object 2 reps(N, M), 
    #     making it the average loss at each pixel. Take te sum over all the objects. 
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

    # 15. apply the mask to the class loss and compute the sum over all the objects 
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

    # box_loss *= self._iou_normalizer
    # class_loss *= self._cls_normalizer
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
    avg_iou = self.avgiou(iou * tf.gather_nd(grid_mask, inds, batch_dims=1))
    avg_obj = self.avgiou(tf.squeeze(sigmoid_conf, axis=-1) * grid_mask)
    return (loss, box_loss, conf_loss, class_loss, avg_iou, avg_obj, recall50, precision50)

  def __call__(self, true_counts, inds, y_true, boxes, classes, y_pred):
    if self._use_reduction_sum:
      return self.call_scaled(true_counts, inds, y_true, boxes, classes,
                               y_pred)
    else:
      return self.call_darknet(true_counts, inds, y_true, boxes, classes,
                               y_pred)
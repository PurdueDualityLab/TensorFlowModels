import tensorflow as tf
from functools import partial
from yolo.ops import (loss_utils, box_ops, math_ops)

# # loss testing
# import matplotlib.pyplot as plt
# import numpy as np


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
  bce = -y * tf.math.log(x + eps) - (1 - y) * tf.math.log(1 - x + eps)

  def delta(dpass):
    x = tf.math.sigmoid(x_prime)
    dx = (-y + x) * dpass
    dy = tf.zeros_like(y)
    return dy, dx, 0.0

  return bce, delta


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
  scaled_box = tf.concat([box_xy, box_wh], axis=-1)
  pred_box = scaled_box / scaler

  # shift scaled boxes
  scaled_box = tf.concat([pred_xy, box_wh], axis=-1)
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
  box_wh = (2 * pred_wh)**2 * anchor_grid

  # build the final boxes
  scaled_box = tf.concat([box_xy, box_wh], axis=-1)
  pred_box = scaled_box / scaler

  # shift scaled boxes
  scaled_box = tf.concat([pred_xy, box_wh], axis=-1)
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
               update_on_repeat=False,
               darknet=None,
               label_smoothing=0.0,
               new_cords=False,
               scale_x_y=1.0,
               max_delta=10,
               **kwargs):
    """Parameters for the YOLO loss functions used at each detection head 
    output. This method builds the loss to be used in both the Scaled YOLO 
    papers and the standard YOLO papers. The Scaled loss is most optimal on 
    images with a high resolution and models that are very large. The standard
    Loss function will perform better on images that are smaller, have more 
    sparse labels, and or on very small models. 

    Args:
      classes: `int` for the number of classes 
      mask: `List[int]` for the output level that this specific model output 
        level
      anchors: `List[List[int]]` for the anchor boxes that are used in the model 
        at all levels. For anchor free prediction set the anchor list to be the 
        same as the image resolution. 
      scale_anchors: `int` for how much to scale this level to get the orginal 
        input shape.
      ignore_thresh: `float` for the IOU value over which the loss is not 
        propagated, and a detection is assumed to have been made.
      truth_thresh: `float` for the IOU value over which the loss is propagated 
        despite a detection being made.
      loss_type: `str` for the typeof iou loss to use with in {ciou, diou, 
        giou, iou}.
      iou_normalizer: `float` for how much to scale the loss on the IOU or the 
        boxes.
      cls_normalizer: `float` for how much to scale the loss on the classes.
      obj_normalizer: `float` for how much to scale loss on the detection map.
      objectness_smooth: `float` for how much to smooth the loss on the 
        detection map.
      use_reduction_sum: `bool` for whether to use the scaled loss 
        or the traditional loss.
      update_on_repeat: `bool` for whether to replace with the newest or the
        best value when an index is consumed by multiple objects. 
      label_smoothing: `float` for how much to smooth the loss on the classes
      new_cords: `bool` for which scaling type to use. 
      scale_xy: dictionary `float` values inidcating how far each pixel can see 
        outside of its containment of 1.0. a value of 1.2 indicates there is a 
        20% extended radius around each pixel that this specific pixel can 
        predict values for a center at. the center can range from 0 - value/2 
        to 1 + value/2, this value is set in the yolo filter, and resused here. 
        there should be one value for scale_xy for each level from min_level to 
        max_level.
      max_delta: gradient clipping to apply to the box loss. 

    Return:
      loss: `float` for the actual loss.
      box_loss: `float` loss on the boxes used for metrics.
      conf_loss: `float` loss on the confidence used for metrics.
      class_loss: `float` loss on the classes used for metrics.
      avg_iou: `float` metric for the average iou between predictions 
        and ground truth.
      avg_obj: `float` metric for the average confidence of the model 
        for predictions.
      recall50: `float` metric for how accurate the model is.
      precision50: `float` metric for how precise the model is.
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
    self._update_on_repeat = update_on_repeat

    self._new_cords = new_cords
    self._any = True

    self._anchor_generator = loss_utils.GridGenerator(
        masks=mask, anchors=anchors, scale_anchors=scale_anchors)

    if ignore_thresh > 0.0:
      self._search_pairs = loss_utils.PairWiseSearch(
          iou_type="iou", any=not self._use_reduction_sum, min_conf=0.25)

    box_kwargs = dict(
        scale_xy=self._scale_x_y,
        darknet=not self._use_reduction_sum,
        normalizer=self._iou_normalizer,
        max_delta=self._max_delta)

    if not self._new_cords:
      self._decode_boxes = partial(get_predicted_box, **box_kwargs)
    else:
      self._decode_boxes = partial(get_predicted_box_newcords, **box_kwargs)

  def box_loss(self, true_box, pred_box, darknet=False):
    # based on the type of loss, compute the iou loss for a box
    # compute_<name> indicated the type of iou to use
    if self._loss_type == 1:
      iou, liou = box_ops.compute_giou(true_box, pred_box)
      loss_box = 1 - liou
    elif self._loss_type == 2:
      iou, liou = box_ops.compute_ciou(true_box, pred_box, darknet=darknet)
      loss_box = 1 - liou
    else:
      iou = box_ops.compute_iou(true_box, pred_box, darknet=darknet)
      liou = iou
      loss_box = 1 - liou

    return iou, liou, loss_box

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

    _, _, iou_max, _ = self._search_pairs(
        pred_boxes, pred_classes, boxes, classes, scale=scale, yxyx=True)

    ignore_mask = tf.cast(iou_max < self._ignore_thresh, pred_boxes.dtype)
    iou_mask = tf.cast(iou_max > self._ignore_thresh, pred_boxes.dtype)

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
    return true_conf, obj_mask

  def call_scaled(self, true_counts, inds, y_true, boxes, classes, y_pred):
    # Generate shape constants.
    shape = tf.shape(true_counts)
    batch_size, width, height, num = shape[0], shape[1], shape[2], shape[3]
    fwidth = tf.cast(width, tf.float32)
    fheight = tf.cast(height, tf.float32)

    # Cast all input compontnts to float32 and stop gradient to save memory.
    y_true = tf.cast(y_true, tf.float32)
    true_counts = tf.cast(true_counts, tf.float32)
    true_conf = tf.clip_by_value(true_counts, 0.0, 1.0)
    grid_points, anchor_grid = self._anchor_generator(
        width, height, batch_size, dtype=tf.float32)

    # Split the y_true list.
    (true_box, ind_mask, true_class, _, _) = tf.split(
        y_true, [4, 1, 1, 1, 1], axis=-1)
    grid_mask = true_conf = tf.squeeze(true_conf, axis=-1)
    true_class = tf.squeeze(true_class, axis=-1)
    num_objs = tf.cast(tf.reduce_sum(ind_mask), dtype=y_pred.dtype)

    # Split up the predicitons.
    y_pred = tf.cast(
        tf.reshape(y_pred, [batch_size, width, height, num, -1]), tf.float32)
    pred_box, pred_conf, pred_class = tf.split(y_pred, [4, 1, -1], axis=-1)

    # Decode the boxes for loss compute.
    scale, pred_box, _ = self._decode_boxes(
        fwidth, fheight, pred_box, anchor_grid, grid_points, darknet=False)

    # If the ignore threshold is enabled, search all boxes ignore all
    # IOU valeus larger than the ignore threshold that are not in the
    # noted ground truth list.
    if self._ignore_thresh != 0.0:
      (_, obj_mask) = self._tiled_global_box_search(
          pred_box,
          tf.stop_gradient(tf.sigmoid(pred_class)),
          tf.stop_gradient(tf.sigmoid(pred_conf)),
          boxes,
          classes,
          true_conf,
          fwidth,
          fheight,
          smoothed=False,
          scale=scale)

    # Scale and shift and select the ground truth boxes
    # and predictions to the prediciton domain.
    offset = tf.cast(
        tf.gather_nd(grid_points, inds, batch_dims=1), true_box.dtype)
    offset = tf.concat([offset, tf.zeros_like(offset)], axis=-1)
    true_box = loss_utils.apply_mask(ind_mask, (scale * true_box) - offset)
    pred_box = loss_utils.apply_mask(ind_mask,
                                     tf.gather_nd(pred_box, inds, batch_dims=1))

    # Select the correct/used prediction classes.
    true_class = tf.one_hot(
        tf.cast(true_class, tf.int32),
        depth=tf.shape(pred_class)[-1],
        dtype=pred_class.dtype)
    true_class = loss_utils.apply_mask(ind_mask, true_class)
    pred_class = loss_utils.apply_mask(
        ind_mask, tf.gather_nd(pred_class, inds, batch_dims=1))

    # Compute the box loss.
    _, iou, box_loss = self.box_loss(true_box, pred_box, darknet=False)
    box_loss = loss_utils.apply_mask(tf.squeeze(ind_mask, axis=-1), box_loss)
    box_loss = math_ops.divide_no_nan(tf.reduce_sum(box_loss), num_objs)

    # Use the box IOU to build the map for confidence loss computation.
    iou = tf.maximum(tf.stop_gradient(iou), 0.0)
    smoothed_iou = ((
        (1 - self._objectness_smooth) * tf.cast(ind_mask, iou.dtype)) +
                    self._objectness_smooth * tf.expand_dims(iou, axis=-1))
    smoothed_iou = loss_utils.apply_mask(ind_mask, smoothed_iou)
    true_conf = loss_utils.build_grid(
        inds, smoothed_iou, pred_conf, ind_mask, update=self._update_on_repeat)
    true_conf = tf.squeeze(true_conf, axis=-1)

    # Compute the cross entropy loss for the confidence map.
    bce = tf.keras.losses.binary_crossentropy(
        tf.expand_dims(true_conf, axis=-1), pred_conf, from_logits=True)
    # bce = tf.reduce_mean(
    #   sigmoid_BCE(tf.expand_dims(true_conf, axis=-1), pred_conf, 0.0), axis = -1)
    if self._ignore_thresh != 0.0:
      bce = loss_utils.apply_mask(obj_mask, bce)
    conf_loss = tf.reduce_mean(bce)

    # Compute the cross entropy loss for the class maps.
    class_loss = tf.keras.losses.binary_crossentropy(
        true_class,
        pred_class,
        label_smoothing=self._label_smoothing,
        from_logits=True)
    # class_loss = tf.reduce_mean(
    #   sigmoid_BCE(true_class, pred_class, self._label_smoothing), axis = -1)
    class_loss = loss_utils.apply_mask(
        tf.squeeze(ind_mask, axis=-1), class_loss)
    class_loss = math_ops.divide_no_nan(tf.reduce_sum(class_loss), num_objs)

    # Apply the weights to each loss.
    box_loss *= self._iou_normalizer
    class_loss *= self._cls_normalizer
    conf_loss *= self._obj_normalizer

    # Add all the losses together then take the sum over the batches.
    mean_loss = box_loss + class_loss + conf_loss
    loss = mean_loss * tf.cast(batch_size, mean_loss.dtype)

    return (loss, box_loss, conf_loss, class_loss, mean_loss, iou, pred_conf,
            ind_mask, grid_mask)

  def call_darknet(self, true_counts, inds, y_true, boxes, classes, y_pred):
    if self._new_cords:
      # Darknet Model Propagates a sigmoid once in back prop so we replicate
      # that behaviour
      y_pred = grad_sigmoid(y_pred)

    # Generate and store constants and format output.
    shape = tf.shape(true_counts)
    batch_size, width, height, num = shape[0], shape[1], shape[2], shape[3]
    fwidth = tf.cast(width, tf.float32)
    fheight = tf.cast(height, tf.float32)
    grid_points, anchor_grid = self._anchor_generator(
        width, height, batch_size, dtype=tf.float32)

    # Cast all input compontnts to float32 and stop gradient to save memory.
    boxes = tf.stop_gradient(tf.cast(boxes, tf.float32))
    classes = tf.stop_gradient(tf.cast(classes, tf.float32))
    y_true = tf.stop_gradient(tf.cast(y_true, tf.float32))
    true_counts = tf.stop_gradient(tf.cast(true_counts, tf.float32))
    true_conf = tf.stop_gradient(tf.clip_by_value(true_counts, 0.0, 1.0))
    grid_points = tf.stop_gradient(grid_points)
    anchor_grid = tf.stop_gradient(anchor_grid)

    # Split all the ground truths to use as seperate items in loss computation.
    (true_box, ind_mask, true_class, _, _) = tf.split(
        y_true, [4, 1, 1, 1, 1], axis=-1)
    true_conf = tf.squeeze(true_conf, axis=-1)
    true_class = tf.squeeze(true_class, axis=-1)
    grid_mask = true_conf

    # Splits all predictions.
    y_pred = tf.cast(
        tf.reshape(y_pred, [batch_size, width, height, num, -1]), tf.float32)
    pred_box, pred_conf, pred_class = tf.split(y_pred, [4, 1, -1], axis=-1)

    # Decode the boxes to be used for loss compute.
    _, _, pred_box = self._decode_boxes(
        fwidth, fheight, pred_box, anchor_grid, grid_points, darknet=True)

    # If the ignore threshold is enabled, search all boxes ignore all
    # IOU valeus larger than the ignore threshold that are not in the
    # noted ground truth list.
    if self._ignore_thresh != 0.0:
      (true_conf, obj_mask) = self._tiled_global_box_search(
          pred_box,
          tf.stop_gradient(tf.sigmoid(pred_class)),
          tf.stop_gradient(tf.sigmoid(pred_conf)),
          boxes,
          classes,
          true_conf,
          fwidth,
          fheight,
          smoothed=self._objectness_smooth > 0)

    # Build the one hot class list that are used for class loss.
    true_class = tf.one_hot(
        tf.cast(true_class, tf.int32),
        depth=tf.shape(pred_class)[-1],
        dtype=pred_class.dtype)
    true_classes = tf.stop_gradient(loss_utils.apply_mask(ind_mask, true_class))

    # Reorganize the one hot class list as a grid.
    true_class = loss_utils.build_grid(
        inds, true_classes, pred_class, ind_mask, update=False)
    true_class = tf.stop_gradient(true_class)

    # Use the class mask to find the number of objects located in
    # each predicted grid cell/pixel.
    counts = true_class
    counts = tf.reduce_sum(counts, axis=-1, keepdims=True)
    reps = tf.gather_nd(counts, inds, batch_dims=1)
    reps = tf.squeeze(reps, axis=-1)
    reps = tf.stop_gradient(tf.where(reps == 0.0, tf.ones_like(reps), reps))

    # Compute the loss for only the cells in which the boxes are located.
    pred_box = loss_utils.apply_mask(ind_mask,
                                     tf.gather_nd(pred_box, inds, batch_dims=1))
    iou, _, box_loss = self.box_loss(true_box, pred_box, darknet=True)
    box_loss = loss_utils.apply_mask(tf.squeeze(ind_mask, axis=-1), box_loss)
    box_loss = math_ops.divide_no_nan(box_loss, reps)
    box_loss = tf.cast(tf.reduce_sum(box_loss, axis=1), dtype=y_pred.dtype)

    # Compute the sigmoid binary cross entropy for the class maps.
    class_loss = tf.reduce_mean(
        sigmoid_BCE(
            tf.expand_dims(true_class, axis=-1),
            tf.expand_dims(pred_class, axis=-1), self._label_smoothing),
        axis=-1)

    # Apply normalization to the class losses.
    if self._cls_normalizer < 1.0:
      # Build a mask based on the true class locations.
      cls_norm_mask = true_class
      # Apply the classes weight to class indexes were one_hot is one.
      class_loss *= ((1 - cls_norm_mask) + cls_norm_mask * self._cls_normalizer)

    # Mask to the class loss and compute the sum over all the objects.
    class_loss = tf.reduce_sum(class_loss, axis=-1)
    class_loss = loss_utils.apply_mask(grid_mask, class_loss)
    class_loss = math_ops.rm_nan_inf(class_loss, val=0.0)
    class_loss = tf.cast(
        tf.reduce_sum(class_loss, axis=(1, 2, 3)), dtype=y_pred.dtype)

    # Compute the sigmoid binary cross entropy for the confidence maps.
    bce = tf.reduce_mean(
        sigmoid_BCE(tf.expand_dims(true_conf, axis=-1), pred_conf, 0.0),
        axis=-1)

    # Mask the confidence loss and take the sum across all the grid cells.
    if self._ignore_thresh != 0.0:
      bce = loss_utils.apply_mask(obj_mask, bce)
    conf_loss = tf.cast(tf.reduce_sum(bce, axis=(1, 2, 3)), dtype=y_pred.dtype)

    # Apply the weights to each loss.
    box_loss *= self._iou_normalizer
    conf_loss *= self._obj_normalizer

    # Add all the losses together then take the mean over the batches.
    loss = box_loss + class_loss + conf_loss
    loss = tf.reduce_mean(loss)

    # Reduce the mean of the losses to use as a metric.
    box_loss = tf.reduce_mean(box_loss)
    conf_loss = tf.reduce_mean(conf_loss)
    class_loss = tf.reduce_mean(class_loss)

    return (loss, box_loss, conf_loss, class_loss, loss, iou, pred_conf,
            ind_mask, grid_mask)

  def __call__(self, true_counts, inds, y_true, boxes, classes, y_pred):
    if self._use_reduction_sum == True:
      (loss, box_loss, conf_loss, class_loss, mean_loss, iou, pred_conf,
       ind_mask, grid_mask) = self.call_scaled(true_counts, inds, y_true, boxes,
                                               classes, y_pred)
    else:
      (loss, box_loss, conf_loss, class_loss, mean_loss, iou, pred_conf,
       ind_mask, grid_mask) = self.call_darknet(true_counts, inds, y_true,
                                                boxes, classes, y_pred)

    # Temporary metrics
    box_loss = 0.05 * box_loss/self._iou_normalizer

    # Metric compute using done here to save time and resources.
    sigmoid_conf = tf.stop_gradient(tf.sigmoid(pred_conf))
    iou = tf.stop_gradient(iou)
    avg_iou = loss_utils.avgiou(
        loss_utils.apply_mask(tf.squeeze(ind_mask, axis=-1), iou))
    avg_obj = loss_utils.avgiou(tf.squeeze(sigmoid_conf, axis=-1) * grid_mask)
    return (loss, box_loss, conf_loss, class_loss, mean_loss,
            tf.stop_gradient(avg_iou), tf.stop_gradient(avg_obj))

import tensorflow as tf
from collections import defaultdict
import abc

from functools import partial
from yolo.ops import (loss_utils, box_ops, math_ops)

class YoloLossBase(object, metaclass=abc.ABCMeta):

  def __init__(self,
               classes,
               mask,
               anchors,
               path_stride=1,
               ignore_thresh=0.7,
               truth_thresh=1.0,
               loss_type="ciou",
               iou_normalizer=1.0,
               cls_normalizer=1.0,
               obj_normalizer=1.0,
               objectness_smooth=True,
               use_scaled_loss=False,
               update_on_repeat=False,
               label_smoothing=0.0,
               new_cords=False,
               scale_x_y=1.0,
               max_delta=10):
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
    self._loss_type = loss_type
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
    self._update_on_repeat = update_on_repeat
    self._new_cords = new_cords
    self._path_stride = path_stride

    box_kwargs = dict(
        scale_xy=self._scale_x_y,
        newcoords=self._new_cords,
        max_delta=self._max_delta)
    self._decode_boxes = partial(loss_utils.get_predicted_box, **box_kwargs)
    self._build_per_path_attributes()

  def box_loss(self, true_box, pred_box, darknet=False):
    if self._loss_type == "giou":
      iou, liou = box_ops.compute_giou(true_box, pred_box)
    elif self._loss_type == "ciou":
      iou, liou = box_ops.compute_ciou(true_box, pred_box, darknet=darknet)
    else:
      liou = iou = box_ops.compute_iou(true_box, pred_box)
    loss_box = 1 - liou
    return iou, liou, loss_box

  def _tiled_global_box_search(self,
                               pred_boxes,
                               pred_classes,
                               boxes,
                               classes,
                               true_conf,
                               smoothed,
                               scale=None):

    # Search all predictions against ground truths to find mathcing boxes for 
    # each pixel.
    _, _, iou_max, _ = self._search_pairs(
        pred_boxes, pred_classes, boxes, classes, scale=scale, yxyx=True)

    # Find the exact indexes to ignore and keep.
    ignore_mask = tf.cast(iou_max < self._ignore_thresh, pred_boxes.dtype)
    iou_mask = iou_max > self._ignore_thresh

    if not smoothed:
      # Ignore all pixels where a box was not supposed to be predicted but a 
      # high confidence box was predicted.
      obj_mask = true_conf + (1 - true_conf) * ignore_mask
    else:
      # Replace pixels in the tre confidence map with the max iou predicted 
      # with in that cell.
      obj_mask = tf.ones_like(true_conf)
      iou_ = (1 - self._objectness_smooth) + self._objectness_smooth * iou_max
      iou_ = tf.where(iou_max > 0, iou_, tf.zeros_like(iou_))
      true_conf = tf.where(iou_mask, iou_, true_conf)

    # Stop gradient so while loop is not tracked.
    obj_mask = tf.stop_gradient(obj_mask)
    true_conf = tf.stop_gradient(true_conf)
    return true_conf, obj_mask

  def __call__(self, true_counts, inds, y_true, boxes, classes, y_pred):
    (loss, box_loss, conf_loss, class_loss, mean_loss, iou, pred_conf,
       ind_mask, grid_mask) = self.call(true_counts, inds, y_true,
                                                boxes, classes, y_pred)

    # Temporary metrics
    box_loss = tf.stop_gradient(0.05 * box_loss/self._iou_normalizer)

    # Metric compute using done here to save time and resources.
    sigmoid_conf = tf.stop_gradient(tf.sigmoid(pred_conf))
    iou = tf.stop_gradient(iou)
    avg_iou = loss_utils.avgiou(
        loss_utils.apply_mask(tf.squeeze(ind_mask, axis=-1), iou))
    avg_obj = loss_utils.avgiou(tf.squeeze(sigmoid_conf, axis=-1) * grid_mask)
    return (loss, box_loss, conf_loss, class_loss, mean_loss,
            tf.stop_gradient(avg_iou), tf.stop_gradient(avg_obj))

  @abc.abstractmethod
  def _build_per_path_attributes(self):
    ...
    
  @abc.abstractmethod
  def call():
    ...

  def post_path_aggregation(self, loss, ground_truths, predictions):
    """this method is not specific to each loss path, but each loss type"""
    return loss
  
  @abc.abstractmethod
  def cross_replica_aggregation(self, loss, num_replicas_in_sync):
    """this method is not specific to each loss path, but each loss type"""
    ...

class DarknetLoss(YoloLossBase):

  def _build_per_path_attributes(self):
    self._anchor_generator = loss_utils.GridGenerator(
        masks=self._masks, 
        anchors=self._anchors, 
        scale_anchors=self._path_stride)

    if self._ignore_thresh > 0.0:
      self._search_pairs = loss_utils.PairWiseSearch(
          iou_type="iou", any=True, min_conf=0.25)
    return

  def call(self, true_counts, inds, y_true, boxes, classes, y_pred):
    if self._new_cords:
      # Darknet Model Propagates a sigmoid once in back prop so we replicate
      # that behaviour
      y_pred = loss_utils.grad_sigmoid(y_pred)

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
          boxes,
          classes,
          true_conf,
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
        loss_utils.sigmoid_BCE(
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
        loss_utils.sigmoid_BCE(
          tf.expand_dims(true_conf, axis=-1), pred_conf, 0.0),
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
    # loss = tf.reduce_mean(loss)

    # Reduce the mean of the losses to use as a metric.
    box_loss = tf.reduce_mean(box_loss)
    conf_loss = tf.reduce_mean(conf_loss)
    class_loss = tf.reduce_mean(class_loss)

    return (loss, box_loss, conf_loss, class_loss, loss, iou, pred_conf,
            ind_mask, grid_mask)

  def cross_replica_aggregation(self, loss, num_replicas_in_sync):
    """this method is not specific to each loss path, but each loss type"""
    return loss/num_replicas_in_sync


class ScaledLoss(YoloLossBase):
  def _build_per_path_attributes(self):
    self._anchor_generator = loss_utils.GridGenerator(
        masks=self._masks, anchors=self._anchors, scale_anchors=self._path_stride)

    if self._ignore_thresh > 0.0:
      self._search_pairs = loss_utils.PairWiseSearch(
          iou_type="iou", any=False, min_conf=0.25)
    return

  def call(self, true_counts, inds, y_true, boxes, classes, y_pred):
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
          boxes,
          classes,
          true_conf,
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
    if self._ignore_thresh != 0.0:
      bce = loss_utils.apply_mask(obj_mask, bce)
    conf_loss = tf.reduce_mean(bce)

    # Compute the cross entropy loss for the class maps.
    class_loss = tf.keras.losses.binary_crossentropy(
        true_class,
        pred_class,
        label_smoothing=self._label_smoothing,
        from_logits=True)
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

  def post_path_aggregation(self, loss, ground_truths, predictions):
    scale = 3 / len(list(predictions.keys()))
    return loss * scale

  def cross_replica_aggregation(self, loss, num_replicas_in_sync):
    """this method is not specific to each loss path, but each loss type"""
    return loss


LOSSES = {
  "darknet": DarknetLoss,
  "scaled" : ScaledLoss
}

class YoloLoss(object):
  def __init__( self,
                keys,
                classes,
                anchors,

                masks = None, 
                path_strides=None,

                truth_thresholds=None,
                ignore_thresholds=None,
                
                loss_types=None,
                iou_normalizers=None,
                cls_normalizers=None,
                obj_normalizers=None,
                objectness_smooths=None,
                box_types=None,
                scale_xys=None,
                max_deltas=None,

                label_smoothing=0.0,
                use_scaled_loss=False,
                update_on_repeat=False):

    if use_scaled_loss: 
      loss_type = "scaled"
    else:
      loss_type = "darknet"

    self._loss_dict = {}
    for key in keys:
      self._loss_dict[key] = LOSSES[loss_type](
        classes=classes,
        anchors=anchors,

        mask=masks[key],
        truth_thresh=truth_thresholds[key],
        ignore_thresh=ignore_thresholds[key],
        loss_type=loss_types[key],
        iou_normalizer=iou_normalizers[key],
        cls_normalizer=cls_normalizers[key],
        obj_normalizer=obj_normalizers[key],
        new_cords=box_types[key],
        objectness_smooth=objectness_smooths[key],
        max_delta=max_deltas[key],
        path_stride=path_strides[key],
        scale_x_y=scale_xys[key], 
        
        use_scaled_loss=use_scaled_loss,
        update_on_repeat=update_on_repeat,
        label_smoothing=label_smoothing
      )

  def __call__(self, ground_truth, predictions, use_reduced_logs = True):
    metric_dict = defaultdict(dict)
    metric_dict['net']['box'] = 0
    metric_dict['net']['class'] = 0
    metric_dict['net']['conf'] = 0

    loss_val = 0
    metric_loss = 0

    num_replicas_in_sync = tf.distribute.get_strategy().num_replicas_in_sync

    for key in predictions.keys():
      (_loss, _loss_box, _loss_conf, _loss_class, _mean_loss, _avg_iou,
       _avg_obj) = self._loss_dict[key](ground_truth['true_conf'][key], 
                                        ground_truth['inds'][key], 
                                        ground_truth['upds'][key],
                                        ground_truth['bbox'], 
                                        ground_truth['classes'],
                                        predictions[key])
      
      # after computing the loss, scale loss as needed for aggregation 
      # across FPN levels 
      _loss = self._loss_dict[key].post_path_aggregation(_loss, 
                                                          ground_truth, 
                                                          predictions)
      # after completing the scaling of the loss on each replica, handle 
      # scaling the loss for mergeing the loss across replicas
      _loss = self._loss_dict[key].cross_replica_aggregation(_loss, 
                                                        num_replicas_in_sync)
      loss_val += _loss
      

      # detach all the below gradients: none of them should make a
      # contribution to the gradient form this point forwards
      metric_loss += tf.stop_gradient(_mean_loss)
      metric_dict[key]['loss'] = tf.stop_gradient(_mean_loss)
      metric_dict[key]['avg_iou'] = tf.stop_gradient(_avg_iou)
      metric_dict[key]["avg_obj"] = tf.stop_gradient(_avg_obj)

      if not use_reduced_logs:
        metric_dict[key]['conf_loss'] = tf.stop_gradient(_loss_conf)
        metric_dict[key]['box_loss'] = tf.stop_gradient(_loss_box)
        metric_dict[key]['class_loss'] = tf.stop_gradient(_loss_class)

      metric_dict['net']['box'] += tf.stop_gradient(_loss_box)
      metric_dict['net']['class'] += tf.stop_gradient(_loss_class)
      metric_dict['net']['conf'] += tf.stop_gradient(_loss_conf)

    tf.print(loss_val)
    # loss_val = self._loss_dict[key].post_path_aggregation(loss_val, 
    #                                                       ground_truth, 
    #                                                       predictions)
    # loss_val = self._loss_dict[key].cross_replica_aggregation(loss_val, 
    #                                                    num_replicas_in_sync)

    return loss_val, metric_loss, metric_dict
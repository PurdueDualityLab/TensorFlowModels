import tensorflow as tf
import tensorflow.keras as ks
from tensorflow.keras import backend as K

# from yolo.ops.loss_utils import GridGenerator
# from yolo.ops.loss_utils import PairWiseSearch
# from yolo.ops.loss_utils import build_grid
# from yolo.ops.loss_utils import apply_mask
from yolo.ops.loss_utils import *
from yolo.ops import box_ops
from yolo.ops import math_ops
import numpy as np

from functools import partial

# loss testing
import matplotlib.pyplot as plt
import numpy as np

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

    self._anchor_generator = GridGenerator(
        masks=mask, anchors=anchors, scale_anchors=scale_anchors)

    if ignore_thresh > 0.0 and not self._use_reduction_sum:
      self._box_pairing = PairWiseSearch(any = True, min_conf= 0.25)
    elif ignore_thresh > 0.0 and self._use_reduction_sum:
      self._box_pairing = PairWiseSearch(any = self._truth_thresh < 1.0, 
                                         min_conf= 0.25, 
                                         track_boxes=self._truth_thresh < 1.0, 
                                         track_classes=self._truth_thresh < 1.0)

    box_kwargs = dict(
        scale_xy=self._scale_x_y,
        darknet=not self._use_reduction_sum,
        normalizer=self._iou_normalizer,
        max_delta=self._max_delta)

    if not self._new_cords:
      self._decode_boxes = partial(get_predicted_box, **box_kwargs)
    else:
      self._decode_boxes = partial(get_predicted_box_scaledyolo, **box_kwargs)

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
      iou, liou = box_ops.compute_giou(true_box, pred_box)
      loss_box = 1 - liou
    elif self._loss_type == 2:
      iou, liou = box_ops.compute_ciou(true_box, pred_box, darknet=darknet)
      # iou = liou = box_ops.bbox_iou(true_box, pred_box, x1y1x2y2 = False, CIoU=True)
      loss_box = 1 - liou
    else:
      iou, liou = box_ops.compute_ciou(true_box, pred_box, darknet=darknet)
      loss_box = 1 - liou
      liou = iou
    return iou, liou, loss_box

  def call_free(self, true_counts, inds, y_true, boxes, classes, y_pred):
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
    grid_points = tf.stop_gradient(grid_points)
    anchor_grid = tf.stop_gradient(anchor_grid)

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
    pred_box, pred_conf, pred_class_g = tf.split(y_pred, [4, 1, -1], axis=-1)

    scale, pred_box_g, unscaled_box = self._decode_boxes(
        fwidth, fheight, pred_box, anchor_grid, grid_points, darknet=False)
    true_box = scale * true_box
    true_box = apply_mask(ind_mask, true_box)
    pred_box = apply_mask(ind_mask, tf.gather_nd(pred_box_g, inds, batch_dims=1))

    # build the class object
    true_class = tf.one_hot(
        tf.cast(true_class, tf.int32),
        depth=tf.shape(pred_class_g)[-1],
        dtype=pred_class_g.dtype)
    true_class = apply_mask(ind_mask, true_class)
    pred_class = apply_mask(ind_mask,
                            tf.gather_nd(pred_class_g, inds, batch_dims=1))

    # compute the loss of all the boxes and apply a mask such that
    # within the 200 boxes, only the indexes of importance are covered
    _, iou, box_loss = self.box_loss(true_box, pred_box, darknet=False)
    box_loss = apply_mask(tf.squeeze(ind_mask, axis=-1), box_loss)
    box_loss = tf.reduce_sum(box_loss)


    # 7.  (class loss) build the one hot encoded true class values
    #     compute the loss on the classes, apply the same inds mask
    #     and the compute the average of all the values
    class_loss = ks.losses.binary_crossentropy(
        true_class,
        pred_class,
        label_smoothing=self._label_smoothing,
        from_logits=True)
    class_loss = apply_mask(tf.squeeze(ind_mask, axis=-1), class_loss)
    class_loss = tf.reduce_sum(class_loss)


    # 6.  (confidence loss) build a selective between the ground truth and the
    #     iou to take only a certain percent of the iou or the ground truth,
    #     i.e smooth the detection map
    #     build a the ground truth detection map
    iou = tf.maximum(tf.stop_gradient(iou), 0.0)
    smoothed_iou = ((
        (1 - self._objectness_smooth) * tf.cast(ind_mask, iou.dtype)) +
                    self._objectness_smooth * tf.expand_dims(iou, axis=-1))
    smoothed_iou = apply_mask(ind_mask, smoothed_iou)
    true_conf = build_grid(
        inds, smoothed_iou, 
        pred_conf, ind_mask, 
        update=self._update_on_repeat)
    true_conf = tf.squeeze(true_conf, axis=-1)
    tfg = true_conf

    if self._ignore_thresh > 0.0:
      # pair wise search
      sigmoid_class = tf.stop_gradient(tf.sigmoid(pred_class_g))
      rb, rc, max_iou, mask = self._box_pairing(pred_box_g, 
                                          sigmoid_class, 
                                          boxes * scale, 
                                          classes, 
                                          clip_thresh = self._ignore_thresh)
      true_conf = true_conf + max_iou * (1 - grid_mask)

      if self._truth_thresh < 1.0:
        mask = (1 - grid_mask) * tf.cast(
          (true_conf > self._truth_thresh), mask.dtype)
        num_objs = tf.reduce_sum(mask) + num_objs

        # box extra loss
        _, _, box_loss_g = self.box_loss(rb, pred_box_g, darknet=False)
        box_loss_g = apply_mask(mask, box_loss_g) * true_conf
        
        # class extra loss
        rc = tf.one_hot(tf.cast(rc, tf.int32), depth=tf.shape(pred_class)[-1],
            dtype=pred_class.dtype)
        class_loss_g = ks.losses.binary_crossentropy(rc, pred_class_g,
            label_smoothing=self._label_smoothing,
            from_logits=True)
        class_loss_g = apply_mask(mask, class_loss_g) * true_conf

        box_loss = tf.reduce_sum(box_loss_g) + box_loss
        class_loss = tf.reduce_sum(class_loss_g) + class_loss
    

    bce = ks.losses.binary_crossentropy(
        K.expand_dims(true_conf, axis=-1), pred_conf, from_logits=True)
    conf_loss = tf.reduce_mean(bce)
    box_loss = math_ops.divide_no_nan(box_loss, num_objs)
    class_loss = math_ops.divide_no_nan(class_loss, num_objs)

    # 8. apply the weights to each loss
    box_loss *= self._iou_normalizer
    class_loss *= self._cls_normalizer
    conf_loss *= self._obj_normalizer

    # 9. add all the losses together then take the sum over the batches
    mean_loss = box_loss + class_loss + conf_loss
    loss = mean_loss * tf.cast(batch_size, mean_loss.dtype)

    # fig, axe = plt.subplots(1, 3)
    # axe[0].imshow(tfg[0, ...].numpy())
    # axe[1].imshow(true_conf[0, ...].numpy())
    # axe[2].imshow(tf.sigmoid(pred_conf)[0, ..., 0].numpy())
    # plt.show()
    # tf.print(mean_loss)

    # 4. apply sigmoid to items and use the gradient trap to contol the backprop
    #    and selective gradient clipping
    sigmoid_conf = tf.stop_gradient(tf.sigmoid(pred_conf))

    # 10. compute all the values for the metrics
    recall50, precision50 = self.APAR(sigmoid_conf, grid_mask, pct=0.5)
    avg_iou = self.avgiou(apply_mask(tf.squeeze(ind_mask, axis=-1), iou))
    avg_obj = self.avgiou(tf.squeeze(sigmoid_conf, axis=-1) * grid_mask)
    return (loss, box_loss, conf_loss, class_loss, mean_loss, avg_iou, avg_obj,
            recall50, precision50)



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
    grid_points = tf.stop_gradient(grid_points)
    anchor_grid = tf.stop_gradient(anchor_grid)

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
    sigmoid_class = tf.stop_gradient(tf.sigmoid(pred_class))
    sigmoid_conf = tf.stop_gradient(tf.sigmoid(pred_conf))

    scale, pred_box, _ = self._decode_boxes(
        fwidth, fheight, pred_box, anchor_grid, grid_points, darknet=False)

    # based on input val new_cords decode the box predicitions
    # and because we are using the scaled loss, do not change the gradients
    # at all
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
    iou = tf.maximum(tf.stop_gradient(iou), 0.0)
    smoothed_iou = ((
        (1 - self._objectness_smooth) * tf.cast(ind_mask, iou.dtype)) +
                    self._objectness_smooth * tf.expand_dims(iou, axis=-1))
    smoothed_iou = apply_mask(ind_mask, smoothed_iou)
    true_conf = build_grid(
        inds, smoothed_iou, pred_conf, ind_mask, update=self._update_on_repeat)
    true_conf = tf.stop_gradient(tf.squeeze(true_conf, axis=-1))

    #     compute the detection map loss, there should be no masks
    #     applied
    bce = ks.losses.binary_crossentropy(
        K.expand_dims(true_conf, axis=-1), pred_conf, from_logits=True)
    # bce = tf.reduce_mean(
    #   sigmoid_BCE(K.expand_dims(true_conf, axis=-1), pred_conf, 0.0), axis = -1)
    conf_loss = tf.reduce_mean(bce)

    # 7.  (class loss) build the one hot encoded true class values
    #     compute the loss on the classes, apply the same inds mask
    #     and the compute the average of all the values
    class_loss = ks.losses.binary_crossentropy(
        true_class,
        pred_class,
        label_smoothing=self._label_smoothing,
        from_logits=True)
    # class_loss = tf.reduce_mean(
    #   sigmoid_BCE(true_class, pred_class, self._label_smoothing), axis = -1)
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
      _, _, pred_box, _ = self._decode_boxes(
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
    if self._ignore_thresh > 0.0:
      rb, rc, max_iou = self._box_pairing(pred_box, sigmoid_class, boxes, classes)
      max_iou = tf.cast(max_iou < self._ignore_thresh, dtype = true_conf.dtype)
      obj_mask = true_conf + (1 - true_conf) * max_iou
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
    true_class = build_grid(
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
    class_loss = tf.reduce_mean(sigmoid_BCE(
        K.expand_dims(true_class, axis=-1), K.expand_dims(pred_class, axis=-1),
        self._label_smoothing), axis = -1)

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
    bce = tf.reduce_mean(
      sigmoid_BCE(K.expand_dims(true_conf, axis=-1), pred_conf, 0.0), axis = -1)

    # 17. apply the ignore mask to the detection map to zero out all the
    #     indexes where the loss should not be computed. we use the apply
    #     mask function to control the gradient propagation, and garuntee
    #     safe masking with no NANs.
    conf_loss = apply_mask(obj_mask, bce)
    conf_loss = tf.cast(
        tf.reduce_sum(conf_loss, axis=(1, 2, 3)), dtype=y_pred.dtype)


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
      return self.call_free(true_counts, inds, y_true, boxes, classes, y_pred)
    # elif self._use_reduction_sum == True:
    #   return self.call_scaled(true_counts, inds, y_true, boxes, classes, y_pred)
    else:
      return self.call_darknet(true_counts, inds, y_true, boxes, classes, y_pred)

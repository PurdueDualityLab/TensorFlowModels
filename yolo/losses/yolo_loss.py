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
def obj_gradient_trap(y):
  def trap(dy):
    return dy
  return y, trap

@tf.custom_gradient
def box_gradient_trap(y, max_delta = np.inf):
  def trap(dy):
    # tf.print(dy[0,0,0,0,0,...])
    dy = math_ops.rm_nan_inf(dy)
    delta = tf.cast(max_delta, dy.dtype)
    dy = tf.clip_by_value(dy, -delta, delta)
    return dy, 0.0
  return y, trap

@tf.custom_gradient
def class_gradient_trap(y):
  def trap(dy):
    dy = math_ops.rm_nan_inf(dy)
    return dy
  return y, trap

def get_predicted_box(width, height, unscaled_box, anchor_grid, grid_points, scale_x_y, max_delta = 5.0):
  pred_xy = tf.math.sigmoid(unscaled_box[...,0:2]) * scale_x_y - 0.5 * (scale_x_y - 1)
  pred_wh = unscaled_box[..., 2:4]

  pred_xy = box_gradient_trap(pred_xy, max_delta)
  pred_wh = box_gradient_trap(pred_wh, max_delta)

  box_xy = tf.stack([pred_xy[..., 0] / width, pred_xy[..., 1] / height], axis=-1) + grid_points
  box_wh = tf.math.exp(pred_wh) * anchor_grid
  pred_box = K.concatenate([box_xy, box_wh], axis=-1)
  return pred_xy, pred_wh, pred_box

def get_predicted_box_newcords(width, height, unscaled_box, anchor_grid, grid_points, scale_x_y, max_delta = 5.0):
  pred_xy = tf.math.sigmoid(unscaled_box[...,0:2]) * scale_x_y - 0.5 * (scale_x_y - 1)
  pred_wh = tf.math.sigmoid(unscaled_box[..., 2:4])

  pred_xy = box_gradient_trap(pred_xy, max_delta)
  pred_wh = box_gradient_trap(pred_wh, max_delta)

  box_xy = tf.stack([pred_xy[..., 0] / width, pred_xy[..., 1] / height], axis=-1) + grid_points
  box_wh = 4 * tf.square(pred_wh) * anchor_grid
  pred_box = K.concatenate([box_xy, box_wh], axis=-1)
  return pred_xy, pred_wh, pred_box


class Yolo_Loss(object):
  def __init__(self,
               classes,
               mask,
               anchors,
               scale_anchors=1,
               num_extras=0,
               ignore_thresh=0.7,
               truth_thresh=1,
               loss_type="ciou",
               iou_normalizer=1.0,
               cls_normalizer=1.0,
               obj_normalizer=1.0,
               use_reduction_sum = False, 
               iou_thresh = 0.213, 
               new_cords = False, 
               scale_x_y=1.0,
               max_delta=10,    
               nms_kind="greedynms",
               beta_nms=0.6,
               reduction=tf.keras.losses.Reduction.NONE,
               path_key=None,
               use_tie_breaker=True,
               name=None,
               **kwargs):
    """
        parameters for the loss functions used at each detection head output

        Args:
          mask: list of indexes for which anchors in the anchors list should be 
            used in prediction
          anchors: list of tuples (w, h) representing the anchor boxes to be 
            used in prediction
          num_extras: number of indexes predicted in addition to 4 for the box 
            and N + 1 for classes
          ignore_thresh: float for the threshold for if iou > threshold the 
            network has made a prediction, and should not be penealized for 
            p(object) prediction if an object exists at this location
          truth_thresh: float thresholding the groud truth to get the true mask
          loss_type: string for the key of the loss to use,
            options -> mse, giou, ciou
          iou_normalizer: float used for appropriatly scaling the iou or the 
            loss used for the box prediction error
          cls_normalizer: float used for appropriatly scaling the classification 
            error
          scale_x_y: float used to scale the predictied x and y outputs
          nms_kind: string used for filtering the output and ensuring each 
            object has only one prediction
          beta_nms: float for the thresholding value to apply in non max 
            supression(nms) -> not yet implemented

        call Return:
          float: for the average loss
        """
    self._classes = tf.constant(tf.cast(classes, dtype=tf.int32))
    self._num = tf.cast(len(mask), dtype=tf.int32)
    self._num_extras = tf.cast(num_extras, dtype=tf.int32)
    self._truth_thresh = truth_thresh
    self._ignore_thresh = ignore_thresh
    self._iou_thresh = iou_thresh
    self._masks = mask
    self._anchors = anchors

    self._use_tie_breaker = tf.cast(use_tie_breaker, tf.bool)
    if loss_type == "giou":
      self._loss_type = 1
    elif loss_type == "ciou":
      self._loss_type = 2
    else:
      self._loss_type = 0

    self._iou_normalizer = iou_normalizer
    self._cls_normalizer = cls_normalizer
    self._obj_normalizer = obj_normalizer
    self._scale_x_y = scale_x_y
    self._max_delta = max_delta

    self._label_smoothing = tf.cast(0.0, tf.float32)
    self._use_reduction_sum = use_reduction_sum

    # used in detection filtering
    self._beta_nms = beta_nms
    self._nms_kind = tf.cast(nms_kind, tf.string)
    self._new_cords = new_cords

    box_kwargs = dict(scale_x_y = self._scale_x_y, 
                  # iou_normalizer=self._iou_normalizer, 
                  max_delta=self._max_delta)

    if not self._new_cords:
      self._decode_boxes = partial(get_predicted_box, **box_kwargs)
    else:
      self._decode_boxes = partial(get_predicted_box_newcords, **box_kwargs)

    # grid comp
    self._anchor_generator = GridGenerator(
        masks=mask, anchors=anchors, scale_anchors=scale_anchors)

    # metric struff
    self._path_key = path_key
    return

  def print_error(self, pred, key):
    if tf.stop_gradient(tf.reduce_any(tf.math.is_nan(pred))):
      tf.print("\nerror: stop training ", key)

  def _get_label_attributes(self, width, height, batch_size, dtype):
    grid_points, anchor_grid = self._anchor_generator(
        width, height, batch_size, dtype=dtype)
    
    return tf.stop_gradient(grid_points), tf.stop_gradient(
        anchor_grid)

  def _get_predicted_box(self, width, height, unscaled_box, anchor_grid,
                         grid_points):
    pred_xy = tf.math.sigmoid(unscaled_box[...,
                                           0:2]) * self._scale_x_y - 0.5 * (
                                               self._scale_x_y - 1)
    pred_wh = unscaled_box[..., 2:4]
    box_xy = tf.stack([pred_xy[..., 0] / width, pred_xy[..., 1] / height],
                      axis=-1) + grid_points
    box_wh = tf.math.exp(pred_wh) * anchor_grid
    pred_box = K.concatenate([box_xy, box_wh], axis=-1)
    return pred_xy, pred_wh, pred_box

  def _scale_ground_truth_box(self, box, width, height, anchor_grid,
                              grid_points, dtype):
    xy = tf.nn.relu(box[..., 0:2] - grid_points)
    xy = K.concatenate([
        K.expand_dims(xy[..., 0] * width, axis=-1),
        K.expand_dims(xy[..., 1] * height, axis=-1)
    ],
                       axis=-1)
    wh = tf.math.log(box[..., 2:4] / anchor_grid)
    wh = math_ops.rm_nan_inf(wh)
    return tf.stop_gradient(xy), tf.stop_gradient(wh)

  def recall(self, pred_conf, true_conf, pct = 0.5):
    recall = tf.reduce_mean(
        tf.math.divide_no_nan(
            tf.reduce_sum(
                math_ops.mul_no_nan(
                  true_conf, 
                  tf.cast(
                    tf.squeeze(
                      tf.math.sigmoid(pred_conf), 
                      axis=-1) > pct, 
                      dtype=true_conf.dtype)),
                axis=(1, 2, 3)), (tf.reduce_sum(true_conf, axis=(1, 2, 3)))))
    return tf.stop_gradient(recall)

  def avgiou(self, iou):
    avg_iou = math_ops.divide_no_nan(
      tf.reduce_sum(iou),
      tf.cast(
          tf.math.count_nonzero(tf.cast(iou, dtype=iou.dtype)),
          dtype=iou.dtype))
    return tf.stop_gradient(avg_iou)

  def box_loss(self, true_box, pred_box):
    if self._loss_type == 1:
      iou, liou = box_ops.compute_giou(true_box, pred_box)
      loss_box = (1 - liou)
    elif self._loss_type == 2:
      iou, liou = box_ops.compute_ciou(true_box, pred_box)
      loss_box = (1 - liou) 
    else:
      iou = box_ops.compute_iou(true_box, pred_box)
      liou = iou
      loss_box = (1 - liou) 
      # # mse loss computation :: yolo_layer.c: scale = (2-truth.w*truth.h)
      # scale = (2 - true_box[..., 2] * true_box[..., 3]) 
      # true_xy, true_wh = self._scale_ground_truth_box(true_box, fwidth, fheight,
      #                                                 anchor_grid, grid_points,
      #                                                 y_pred.dtype)
      # loss_xy = tf.reduce_sum(K.square(true_xy - pred_xy), axis=-1)
      # loss_wh = tf.reduce_sum(K.square(true_wh - pred_wh), axis=-1)
      # loss_box = (loss_wh + loss_xy) * scale

    return iou, liou, loss_box


  def _build_mask_body(self, pred_boxes_, pred_classes_, pred_classes_max, boxes, classes, ignore_mask_, idx):
    batch_size = tf.shape(boxes)[0]
    box_slice = tf.slice(boxes, [0, idx * TILE_SIZE, 0], [batch_size, TILE_SIZE, 4])
    class_slice = tf.slice(classes, [0, idx * TILE_SIZE], [batch_size, TILE_SIZE])

    box_slice = tf.expand_dims(box_slice, axis = -2)
    box_slice = tf.expand_dims(box_slice, axis = 1)
    box_slice = tf.expand_dims(box_slice, axis = 1)
    # box_slice = box_ops.yxyx_to_xcycwh(box_slice)
    
    pred_boxes = tf.expand_dims(pred_boxes_, axis = -3)

    iou, ciou, _ = self.box_loss(box_slice, pred_boxes)
    iou_mask = iou > self._ignore_thresh

    class_slice = tf.expand_dims(class_slice, axis = 1)
    class_slice = tf.expand_dims(class_slice, axis = 1)
    class_slice = tf.expand_dims(class_slice, axis = 1)

    # cconfidence is low
    iou_mask = tf.transpose(iou_mask, perm = (0, 1, 2, 4, 3))    
    matched_classes = tf.equal(class_slice, pred_classes_max)
    iou_mask = tf.logical_and(iou_mask, matched_classes)
    iou_mask =  tf.reduce_any(iou_mask, axis = -1, keepdims=False)
    ignore_mask = tf.logical_not(iou_mask)
    
    ignore_mask = tf.logical_and(ignore_mask, ignore_mask_)
    return pred_boxes_, pred_classes_, pred_classes_max, boxes, classes, ignore_mask, idx + 1

  def _build_truth_thresh_body(self, pred_boxes_, pred_classes_, pred_conf, boxes, classes, conf_loss_, loss_, count, idx):
    batch_size = tf.shape(boxes)[0]
    box_slice = tf.slice(boxes, [0, idx * TILE_SIZE, 0], [batch_size, TILE_SIZE, 4])
    class_slice = tf.slice(classes, [0, idx * TILE_SIZE], [batch_size, TILE_SIZE])

    box_slice = tf.expand_dims(box_slice, axis = -2)
    box_slice = tf.expand_dims(box_slice, axis = 1)
    box_slice = tf.expand_dims(box_slice, axis = 1)
    class_slice = tf.expand_dims(class_slice, axis = 1)
    class_slice = tf.expand_dims(class_slice, axis = 1)
    class_slice = tf.expand_dims(class_slice, axis = 1)
    # tf.print(tf.shape(class_slice))

    pred_classes = tf.expand_dims(pred_classes_, axis = -2)
    pred_boxes = tf.expand_dims(pred_boxes_, axis = -3)

    # truth box loss
    iou, liou, loss_box = self.box_loss(box_slice, pred_boxes)
    iou_mask = iou > self._truth_thresh
    loss_box = math_ops.mul_no_nan(tf.cast(iou_mask, loss_box.dtype), loss_box)
    loss_box = tf.reduce_sum(loss_box, axis = -2)

    # truth_class loss
    iou_mask = tf.transpose(iou_mask, perm = (0, 1, 2, 4, 3))    
    class_slice = tf.one_hot(
        tf.cast(class_slice, tf.int32),
        depth=tf.shape(pred_classes)[-1],
        dtype=loss_box.dtype)
    class_loss = tf.reduce_sum(
        ks.losses.binary_crossentropy(
            K.expand_dims(class_slice, axis=-1),
            K.expand_dims(pred_classes, axis=-1),
            label_smoothing=self._label_smoothing,
            from_logits=False),
        axis=-1) 
    class_loss = class_loss * tf.cast(iou_mask, class_loss.dtype)
    class_loss = tf.reduce_sum(class_loss, axis = -1)

    obns_grid = tf.cast(tf.reduce_any(iou_mask, axis = -1, keepdims = True), loss_box.dtype)
    pred_conf *= obns_grid
    conf_loss = ks.losses.binary_crossentropy(
      K.expand_dims(obns_grid, axis=-1), K.expand_dims(pred_conf, axis = -1), from_logits=True)

    # loss = (loss_box + class_loss + conf_loss) * obns_grid
    loss = (loss_box + class_loss) * tf.squeeze(obns_grid, axis = -1)
    conf_loss_ += conf_loss
    loss += loss_
    count += obns_grid 
    return pred_boxes_, pred_classes_, pred_conf, boxes, classes, conf_loss_, loss, count, idx + 1

  def _tiled_global_box_search(self, pred_boxes, pred_classes, pred_conf, boxes, classes, true_conf):
    num_boxes = tf.shape(boxes)[-2]
    num_tiles = num_boxes//TILE_SIZE
    base = tf.cast(tf.ones_like(tf.reduce_sum(pred_boxes, axis = -1)), tf.bool)
    boxes = box_ops.yxyx_to_xcycwh(boxes)

    pred_classes_max = tf.cast(
      tf.expand_dims(tf.argmax(pred_classes, axis = -1), axis = -1), tf.float32)
    pred_classes_conf = tf.cast(
      tf.expand_dims(tf.reduce_max(pred_classes, axis = -1), axis = -1), tf.float32)
    pred_classes_mask = tf.cast(pred_classes_conf > 0.25, tf.float32)
    pred_classes_max = (pred_classes_max * pred_classes_mask) - (1.0 - pred_classes_mask)

    def _loop_cond(pred_boxes, pred_classes, pred_classes_max, boxes, classes, ignore_mask, idx):
      return idx < num_tiles

    _, _, _, _, _, ignore_mask, idx = tf.while_loop(
      _loop_cond, self._build_mask_body, [
        pred_boxes, pred_classes, pred_classes_max, boxes, classes, base, tf.constant(0)]
    )

    # loss_base = tf.zeros_like(tf.reduce_sum(pred_boxes, axis = -1))
    # obns_base = tf.zeros_like(tf.reduce_sum(pred_boxes, axis = -1, keepdims = True))

    # def _loop_cond(pred_boxes_, pred_classes, pred_conf, boxes, classes, conf_loss, loss_, count, idx):
    #   return idx < num_tiles
    
    # _, _, _, _, _, obns_loss, truth_loss, count, idx = tf.while_loop(
    #   _loop_cond, self._build_truth_thresh_body, [
    #     pred_boxes, pred_classes, pred_conf, boxes, classes, obns_base, loss_base, obns_base, tf.constant(0)]
    # )

    # tf.print(tf.reduce_sum(count))

    ignore_mask = tf.stop_gradient(tf.cast(ignore_mask, pred_classes.dtype))
    obj_mask =  tf.stop_gradient((true_conf + (1 - true_conf) * ignore_mask)) 
    true_conf = tf.stop_gradient(true_conf)
    return ignore_mask, 0.0, 0.0, true_conf, obj_mask


  def __call__(self, true_conf, inds, y_true, boxes, classes, y_pred):
    # 1. generate and store constants and format output
    shape = tf.shape(true_conf)
    batch_size, width, height, num = shape[0], shape[1], shape[2], shape[3]
    fwidth = tf.cast(width, tf.float32)
    fheight = tf.cast(height, tf.float32)


    inds = tf.stop_gradient(tf.where(inds == -1, 0, inds))
    boxes =  tf.stop_gradient(tf.cast(boxes, tf.float32))
    classes =  tf.stop_gradient(tf.cast(classes, tf.float32))
    y_true =  tf.stop_gradient(tf.cast(y_true, tf.float32))
    true_conf =  tf.stop_gradient(tf.cast(true_conf, tf.float32))
    grid_points, anchor_grid = self._get_label_attributes(
      width, height, batch_size, tf.float32)

    y_pred = tf.cast(
      tf.reshape(y_pred, 
                 [batch_size, width, height, num, -1]), 
                 tf.float32)
    pred_box, pred_conf, pred_class = tf.split(y_pred, [4, 1, -1], axis = -1)
    pred_class = class_gradient_trap(tf.sigmoid(pred_class))
    pred_conf = obj_gradient_trap(pred_conf)
    pred_xy, pred_wh, pred_box = self._decode_boxes(fwidth, 
                                                    fheight, 
                                                    pred_box, 
                                                    anchor_grid, 
                                                    grid_points)
    


    # 3. split up ground_truth into components, xy, wh, confidence, 
    # class -> apply calculations to acchive safe format as predictions
    (true_box, 
     ind_mask, 
     true_class, 
     best_iou_match, 
     reps) = tf.split(y_true, [4, 1, 1, 1, 1], axis=-1)
    true_conf = tf.squeeze(true_conf, axis = -1)
    true_class = tf.squeeze(true_class, axis = -1)
    reps = tf.squeeze(reps, axis = -1)
    true_class = tf.one_hot(
        tf.cast(true_class, tf.int32),
        depth=tf.shape(pred_class)[-1],
        dtype=y_pred.dtype)


    (mask_loss, 
     thresh_class_loss, 
     thresh_box_loss, 
     true_conf, 
     obj_mask) = self._tiled_global_box_search(
       pred_box, pred_class, pred_conf, boxes, classes, true_conf)

    pred_box = math_ops.mul_no_nan(ind_mask, tf.gather_nd(pred_box, inds, batch_dims=1))
    pred_class = math_ops.mul_no_nan(ind_mask, tf.gather_nd(pred_class, inds, batch_dims=1))
    true_class = math_ops.mul_no_nan(ind_mask, true_class)

    iou, liou, box_loss = self.box_loss(true_box, pred_box)
    box_loss = math_ops.mul_no_nan(tf.squeeze(ind_mask, axis = -1), math_ops.divide_no_nan(box_loss, reps))
    box_loss = tf.cast(
        tf.reduce_sum(box_loss, axis=1), dtype=y_pred.dtype)

    class_loss = tf.reduce_sum(
        ks.losses.binary_crossentropy(
            K.expand_dims(true_class, axis=-1),
            K.expand_dims(pred_class, axis=-1),
            label_smoothing=self._label_smoothing,
            from_logits=False),
        axis=-1) 
    class_loss = math_ops.mul_no_nan(tf.squeeze(ind_mask, axis = -1), math_ops.divide_no_nan(class_loss, reps))
    class_loss = tf.cast(
        tf.reduce_sum(class_loss, axis=1), dtype=y_pred.dtype)

    bce = ks.losses.binary_crossentropy(
      K.expand_dims(true_conf, axis=-1), pred_conf, from_logits=True)
    conf_loss = math_ops.mul_no_nan(obj_mask, bce)
    conf_loss = tf.cast(
        tf.reduce_sum(conf_loss, axis=(1, 2, 3)), dtype=y_pred.dtype)

    box_loss *= self._iou_normalizer
    class_loss *= self._cls_normalizer
    conf_loss *= self._obj_normalizer

    loss = box_loss + class_loss + conf_loss
    loss = tf.reduce_mean(loss)

    box_loss = tf.reduce_mean(box_loss)
    conf_loss = tf.reduce_mean(conf_loss)
    class_loss = tf.reduce_mean(class_loss)

    recall50 = self.recall(pred_conf, true_conf, pct = 0.5)
    avg_iou = self.avgiou(iou)
    avg_obj = self.avgiou(
      tf.sigmoid(tf.squeeze(pred_conf, axis = -1)) * true_conf)

    return loss, box_loss, conf_loss, class_loss, avg_iou, avg_obj, recall50
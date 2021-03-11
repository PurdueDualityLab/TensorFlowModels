import tensorflow as tf
import tensorflow.keras as ks
from tensorflow.keras import backend as K

from yolo.ops.loss_utils import GridGenerator
from yolo.ops import box_ops
from yolo.ops import math_ops
import numpy as np

@tf.custom_gradient
def gradient_trap(y):
  def trap(dy):
    tf.print(dy)
    return dy
  return y, trap

@tf.custom_gradient
def obj_gradient_trap(y, obj_normalizer = 1.0):
  def trap(dy):
    return dy * obj_normalizer, 0
  return y, trap

@tf.custom_gradient
def box_gradient_trap(y, iou_normalizer = 1.0, max_delta = np.inf):
  def trap(dy):
    dy *= iou_normalizer
    dy = math_ops.rm_nan_inf(dy)
    delta = tf.cast(max_delta, dy.dtype)
    dy = tf.clip_by_value(dy, -delta, delta)

    tf.print(tf.reduce_max(dy))
    return dy, 0.0, 0.0
  return y, trap

@tf.custom_gradient
def class_gradient_trap(y, class_normalizer = 1.0):
  def trap(dy):
    dy *= class_normalizer
    dy = math_ops.rm_nan_inf(dy)
    return dy, 0.0
  return y, trap

def get_predicted_box(width, height, unscaled_box, anchor_grid, grid_points, scale_x_y):
  pred_xy = tf.math.sigmoid(unscaled_box[...,0:2]) * scale_x_y - 0.5 * (scale_x_y - 1)
  pred_wh = unscaled_box[..., 2:4]
  box_xy = tf.stack([pred_xy[..., 0] / width, pred_xy[..., 1] / height],axis=-1) + grid_points
  box_wh = tf.math.exp(pred_wh) * anchor_grid
  pred_box = K.concatenate([box_xy, box_wh], axis=-1)
  return pred_xy, pred_wh, pred_box

def get_predicted_box_newcords(width, height, unscaled_box, anchor_grid, grid_points, scale_x_y):
  pred_xy = tf.math.sigmoid(unscaled_box[...,0:2]) * scale_x_y - 0.5 * (scale_x_y - 1)
  pred_wh = tf.math.sigmoid(unscaled_box[..., 2:4])
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
          nms_kind: string used for filtering the output and ensuring each object 
            ahs only one prediction
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

    # grid comp
    self._anchor_generator = GridGenerator(
        masks=mask, anchors=anchors, scale_anchors=scale_anchors)

    # metric struff
    self._path_key = path_key
    return

  def print_error(self, pred, key):
    if tf.stop_gradient(tf.reduce_any(tf.math.is_nan(pred))):
      tf.print("\nerror: stop training ", key)

  def _get_label_attributes(self, width, height, batch_size, y_true, y_pred,
                            dtype):
    grid_points, anchor_grid = self._anchor_generator(
        width, height, batch_size, dtype=dtype)
    y_true = tf.cast(y_true, dtype)
    return tf.stop_gradient(grid_points), tf.stop_gradient(
        anchor_grid), tf.stop_gradient(y_true)

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

  def ce(self, target, output):

    def _smooth_labels(y_true):
      return tf.stop_gradient(y_true * (1.0 - self._label_smoothing) +
                              0.5 * self._label_smoothing)

    target = _smooth_labels(target)
    output = tf.clip_by_value(output, K.epsilon(), 1. - K.epsilon())
    # loss = -target * tf.math.log(output + K.epsilon())
    loss = math_ops.mul_no_nan(-target, tf.math.log(output + K.epsilon()))
    return loss

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
          tf.math.count_nonzero(tf.cast(iou > 0, dtype=iou.dtype)),
          dtype=iou.dtype))
    return tf.stop_gradient(avg_iou)

  def get_mask(self, iou, classes, true_classes, dtype):
    box_match = iou > self._ignore_thresh
    truth_alter = iou > self._truth_thresh
    class_match = tf.argmax(tf.sigmoid(classes), axis = -1) == tf.argmax(true_classes, axis = -1)
    matched_boxes = tf.logical_and(box_match, class_match)
    unmatched_boxes = tf.logical_not(matched_boxes)
    mask = tf.cast(tf.logical_or(unmatched_boxes, truth_alter), dtype)
    mask = math_ops.rm_nan_inf(mask, val=0)
    return (tf.stop_gradient(mask), 
            tf.stop_gradient(matched_boxes),
            tf.stop_gradient(truth_alter))


  def __call__(self, y_true, y_pred):
    # 1. generate and store constants and format output
    shape = tf.shape(y_pred)
    batch_size, width, height = shape[0], shape[1], shape[2]
    num = tf.shape(y_true)[-2]

    # y_pred = gradient_trap(y_pred)if n:

    y_pred = tf.cast(
        tf.reshape(y_pred, [batch_size, width, height, num, -1]), tf.float32)

    grid_points, anchor_grid, y_true = self._get_label_attributes(
        width, height, batch_size, y_true, y_pred, tf.float32)

    fwidth = tf.cast(width, y_pred.dtype)
    fheight = tf.cast(height, y_pred.dtype)

    # 2. split up layer output into components, xy, wh, confidence, class -> then apply activations to the correct items
    pred_box = box_gradient_trap(y_pred[..., 0:4], self._iou_normalizer, self._max_delta)
    if not self._new_cords:
      pred_xy, pred_wh, pred_box = get_predicted_box(
          fwidth, fheight, , anchor_grid, grid_points, self._scale_x_y)
    else:
      pred_xy, pred_wh, pred_box = get_predicted_box_newcords(
          fwidth, fheight, y_pred[..., 0:4], anchor_grid, grid_points, self._scale_x_y)
    pred_conf = tf.expand_dims(y_pred[..., 4], axis=-1)
    pred_class = y_pred[..., 5:]

    self.print_error(pred_box, "boxes")
    self.print_error(pred_conf, "confidence")
    self.print_error(pred_class, "classes")

    # 3. split up ground_truth into components, xy, wh, confidence, 
    # class -> apply calculations to acchive safe format as predictions
    true_box, true_conf, true_class, best_iou_match = tf.split(y_true, [4, 1, 1, 1], axis=-1)
    true_class = tf.squeeze(true_class, axis=-1)
    true_conf = tf.squeeze(true_conf, axis=-1)
    best_iou_match = tf.squeeze(best_iou_match, axis=-1)
    true_class = tf.one_hot(
        tf.cast(true_class, tf.int32),
        depth=tf.shape(pred_class)[-1],
        dtype=y_pred.dtype)

    # pred_box = box_gradient_trap(pred_box, self._iou_normalizer, self._max_delta)
    # 5. apply generalized IOU or mse to the box predictions -> only the indexes 
    # where an object exists will affect the total loss -> found via the 
    # true_confidnce in ground truth
    if self._loss_type == 1:
      iou, liou = box_ops.compute_giou(true_box, pred_box)
      loss_box = math_ops.mul_no_nan(true_conf, (1 - liou)) #* self._iou_normalizer
    elif self._loss_type == 2:
      iou, liou = box_ops.compute_ciou(true_box, pred_box)
      loss_box = math_ops.mul_no_nan(true_conf, (1 - liou)) # * self._iou_normalizer
    else:
      # iou mask computation
      iou = box_ops.compute_iou(true_box, pred_box)
      liou = iou
      # mse loss computation :: yolo_layer.c: scale = (2-truth.w*truth.h)
      scale = (2 - true_box[..., 2] * true_box[..., 3]) * self._iou_normalizer
      true_xy, true_wh = self._scale_ground_truth_box(true_box, fwidth, fheight,
                                                      anchor_grid, grid_points,
                                                      y_pred.dtype)
      loss_xy = tf.reduce_sum(K.square(true_xy - pred_xy), axis=-1)
      loss_wh = tf.reduce_sum(K.square(true_wh - pred_wh), axis=-1)
      loss_box = math_ops.mul_no_nan(true_conf, (loss_wh + loss_xy) * scale)

    
    # 6. apply binary cross entropy(bce) to class attributes -> only the indexes
    # where an object exists will affect the total loss -> found via the 
    # true_confidnce in ground truth
    pred_class = class_gradient_trap(pred_class, self._cls_normalizer)
    class_loss = tf.reduce_sum(
        ks.losses.binary_crossentropy(
            K.expand_dims(true_class, axis=-1),
            K.expand_dims(pred_class, axis=-1),
            label_smoothing=self._label_smoothing,
            from_logits=True),
        axis=-1) #* self._cls_normalizer
    class_loss = math_ops.mul_no_nan(true_conf, class_loss)

    # 7. apply bce to confidence at all points and then strategiacally penalize 
    # the network for making predictions of objects at locations were no object 
    # exists
    pred_conf = obj_gradient_trap(pred_conf, self._obj_normalizer)
    mask_loss, matched_mask, truth_alter = self.get_mask(iou,
                                                         pred_class, 
                                                         true_class, 
                                                         y_pred.dtype)

    if self._new_cords:
      # objectness scaling 
      obj_mask =  tf.stop_gradient((true_conf + (1 - true_conf))) #* self._obj_normalizer
      true_conf = tf.where(matched_mask, liou, true_conf)
      true_conf = tf.where(truth_alter, tf.square(liou), true_conf)
      true_conf = tf.stop_gradient(true_conf)
    else:
      obj_mask =  tf.stop_gradient((true_conf + (1 - true_conf) * mask_loss)) #* self._obj_normalizer

    bce = ks.losses.binary_crossentropy(
      K.expand_dims(true_conf, axis=-1), pred_conf, from_logits=True)
    conf_loss = math_ops.mul_no_nan(obj_mask, bce)

    # 8. take the sum of all the dimentions and reduce the loss such that each
    #  batch has a unique loss value
    loss_box = tf.cast(
        tf.reduce_sum(loss_box, axis=(1, 2, 3)), dtype=y_pred.dtype)
    conf_loss = tf.cast(
        tf.reduce_sum(conf_loss, axis=(1, 2, 3)), dtype=y_pred.dtype)
    class_loss = tf.cast(
        tf.reduce_sum(class_loss, axis=(1, 2, 3)), dtype=y_pred.dtype)

    # 9. i beleive tensorflow will take the average of all the batches loss, so 
    # add them and let TF do its thing
    if self._use_reduction_sum:
      loss = tf.reduce_sum(class_loss + conf_loss + loss_box)  
    else:
      loss = tf.reduce_mean(class_loss + conf_loss + loss_box)

    loss_box = tf.reduce_mean(loss_box)
    conf_loss = tf.reduce_mean(conf_loss)
    class_loss = tf.reduce_mean(class_loss)

    # 10. store values for use in metrics
    recall50 = self.recall(pred_conf, true_conf, pct = 0.5)
    avg_iou = self.avgiou(iou)
    avg_obj = self.avgiou(tf.sigmoid(tf.squeeze(pred_conf, axis = -1)) * true_conf)
    return loss, loss_box, conf_loss, class_loss, avg_iou, avg_obj, recall50
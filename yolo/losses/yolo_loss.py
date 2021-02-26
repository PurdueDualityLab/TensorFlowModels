import tensorflow as tf
import tensorflow.keras as ks
from tensorflow.keras import backend as K

from yolo.ops import box_ops
from yolo.ops.loss_utils import GridGenerator


def smooth_labels(y_true, num_classes, label_smoothing):
  num_classes = tf.cast(num_classes, label_smoothing.dtype)
  return y_true * (1.0 - label_smoothing) + (label_smoothing / num_classes)


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
               scale_x_y=1.0,
               nms_kind="greedynms",
               beta_nms=0.6,
               reduction=tf.keras.losses.Reduction.NONE,
               path_key=None,
               max_val=5,
               use_tie_breaker=True,
               name=None,
               **kwargs):
    """
        parameters for the loss functions used at each detection head output

        Args:
          mask: list of indexes for which anchors in the anchors list should be used in prediction
          anchors: list of tuples (w, h) representing the anchor boxes to be used in prediction
          num_extras: number of indexes predicted in addition to 4 for the box and N + 1 for classes
          ignore_thresh: float for the threshold for if iou > threshold the network has made a prediction,
            and should not be penealized for p(object) prediction if an object exists at this location
          truth_thresh: float thresholding the groud truth to get the true mask
          loss_type: string for the key of the loss to use,
            options -> mse, giou, ciou
          iou_normalizer: float used for appropriatly scaling the iou or the loss used for the box prediction error
          cls_normalizer: float used for appropriatly scaling the classification error
          scale_x_y: float used to scale the predictied x and y outputs
          nms_kind: string used for filtering the output and ensuring each object ahs only one prediction
          beta_nms: float for the thresholding value to apply in non max supression(nms) -> not yet implemented

        call Return:
          float: for the average loss
        """
    self._classes = tf.constant(tf.cast(classes, dtype=tf.int32))
    self._num = tf.cast(len(mask), dtype=tf.int32)
    self._num_extras = tf.cast(num_extras, dtype=tf.int32)
    self._truth_thresh = truth_thresh
    self._ignore_thresh = ignore_thresh
    self._masks = mask

    # used (mask_n >= 0 && n != best_n && l.iou_thresh < 1.0f) for id n != nest_n
    # checks all anchors to see if another anchor was used on this ground truth box to make a prediction
    # if iou > self._iou_thresh then the network check the other anchors, so basically
    # checking anchor box 1 on prediction for anchor box 2
    # self._iou_thresh = 0.213 # recomended use = 0.213 in [yolo]
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
    self._max_value = max_val
    self._label_smoothing = tf.cast(0.0, tf.float32)

    # used in detection filtering
    self._beta_nms = beta_nms
    self._nms_kind = tf.cast(nms_kind, tf.string)

    # grid comp
    self._anchor_generator = GridGenerator(
        masks=mask, anchors=anchors, scale_anchors=scale_anchors)

    # metric struff
    self._path_key = path_key
    return

  def print_error(self, pred_conf):
    if tf.stop_gradient(tf.reduce_any(tf.math.is_nan(pred_conf))):
      tf.print("\nerror: stop training")

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
    wh = box_ops.rm_nan_inf(wh)
    return tf.stop_gradient(xy), tf.stop_gradient(wh)

  def ce(self, target, output):
    def _smooth_labels(y_true):
      return tf.stop_gradient(y_true * (1.0 - self._label_smoothing) + 0.5 * self._label_smoothing)
    target = _smooth_labels(target)
    output = tf.clip_by_value(output, K.epsilon(), 1. - K.epsilon())
    loss = -tf.math.xlogy(target, output + K.epsilon())
    #loss = tf.where(tf.math.is_nan(loss), 1.0, loss)
    return loss

  def __call__(self, y_true, y_pred):
    # 1. generate and store constants and format output
    shape = tf.shape(y_pred)
    batch_size, width, height = shape[0], shape[1], shape[2]
    num = tf.shape(y_true)[-2]

    grid_points, anchor_grid, y_true = self._get_label_attributes(
        width, height, batch_size, y_true, y_pred, tf.float32)

    y_pred = tf.cast(
        tf.reshape(y_pred, [batch_size, width, height, num, self._classes + 5]),
        tf.float32)

    fwidth = tf.cast(width, y_pred.dtype)
    fheight = tf.cast(height, y_pred.dtype)

    # 2. split up layer output into components, xy, wh, confidence, class -> then apply activations to the correct items
    pred_xy, pred_wh, pred_box = self._get_predicted_box(
        fwidth, fheight, y_pred[..., 0:4], anchor_grid, grid_points)
    pred_conf = tf.expand_dims(tf.math.sigmoid(y_pred[..., 4]), axis=-1)
    pred_class = tf.math.sigmoid(y_pred[..., 5:])
    self.print_error(pred_box)

    # 3. split up ground_truth into components, xy, wh, confidence, class -> apply calculations to acchive safe format as predictions
    # true_box, true_conf, true_class = tf.split(y_true, [4, 1, -1], axis=-1)
    true_box, true_conf, true_class = tf.split(y_true, [4, 1, 1], axis=-1)
    true_class = tf.squeeze(true_class, axis=-1)
    true_conf = tf.squeeze(true_conf, axis=-1)
    true_class = tf.one_hot(
        tf.cast(true_class, tf.int32),
        depth= self._classes,
        dtype=y_pred.dtype)

    # 5. apply generalized IOU or mse to the box predictions -> only the indexes where an object exists will affect the total loss -> found via the true_confidnce in ground truth
    if self._loss_type == 1:
      iou, giou = box_ops.compute_giou(true_box, pred_box)
      mask_iou = tf.cast(iou < self._ignore_thresh, dtype=y_pred.dtype)
      loss_box = (1 - giou) * self._iou_normalizer * true_conf
      #loss_box = tf.math.minimum(loss_box, self._max_value)
    elif self._loss_type == 2:
      iou, ciou = box_ops.compute_ciou(true_box, pred_box)
      mask_iou = tf.cast(iou < self._ignore_thresh, dtype=y_pred.dtype)
      loss_box = (1 - ciou) * self._iou_normalizer * true_conf
      #loss_box = tf.math.minimum(loss_box, self._max_value)
    else:
      # iou mask computation
      iou = box_ops.compute_iou(true_box, pred_box)
      mask_iou = tf.cast(iou < self._ignore_thresh, dtype=y_pred.dtype)

      # mse loss computation :: yolo_layer.c: scale = (2-truth.w*truth.h)
      scale = (2 - true_box[..., 2] * true_box[..., 3]) * self._iou_normalizer
      true_xy, true_wh = self._scale_ground_truth_box(true_box, fwidth, fheight,
                                                      anchor_grid, grid_points,
                                                      y_pred.dtype)
      loss_xy = tf.reduce_sum(K.square(true_xy - pred_xy), axis=-1)
      loss_wh = tf.reduce_sum(K.square(true_wh - pred_wh), axis=-1)
      loss_box = (loss_wh + loss_xy) * true_conf * scale
      #loss_box = tf.math.minimum(loss_box, self._max_value)

    # 6. apply binary cross entropy(bce) to class attributes -> only the indexes where an object exists will affect the total loss -> found via the true_confidnce in ground truth
    class_loss = self._cls_normalizer * tf.reduce_sum(
        ks.losses.binary_crossentropy(
            K.expand_dims(true_class, axis=-1),
            K.expand_dims(pred_class, axis=-1), 
            label_smoothing=self._label_smoothing),
        axis=-1) * true_conf
    
    # 7. apply bce to confidence at all points and then strategiacally penalize the network for making predictions of objects at locations were no object exists
    # bce = ks.losses.binary_crossentropy(
    #     K.expand_dims(true_conf, axis=-1), pred_conf)
    # #bce = self.bce(true_conf, tf.squeeze(pred_conf))
    # conf_loss = (true_conf + (1 - true_conf) * mask_iou) * bce * self._obj_normalizer
    ce = self.ce(true_conf, tf.squeeze(pred_conf))
    ce_neg = self.ce((1-true_conf) * mask_iou, tf.squeeze(1-pred_conf))
    conf_loss = (ce + ce_neg) * self._obj_normalizer

    # 8. take the sum of all the dimentions and reduce the loss such that each batch has a unique loss value
    loss_box = tf.cast(tf.reduce_sum(loss_box, axis=(1, 2, 3)), dtype=y_pred.dtype)
    conf_loss =tf.cast(tf.reduce_sum(conf_loss, axis=(1, 2, 3)), dtype=y_pred.dtype)
    class_loss = tf.cast(tf.reduce_sum(class_loss, axis=(1, 2, 3)), dtype=y_pred.dtype)


    # 9. i beleive tensorflow will take the average of all the batches loss, so add them and let TF do its thing
    loss = tf.reduce_mean(class_loss + conf_loss + loss_box)
    loss_box = tf.reduce_mean(loss_box)
    conf_loss = tf.reduce_mean(conf_loss)
    class_loss = tf.reduce_mean(class_loss)

    # 10. store values for use in metrics
    recall50 = tf.stop_gradient(tf.reduce_mean(
        tf.math.divide_no_nan(
            tf.reduce_sum(
                tf.cast(
                    tf.squeeze(pred_conf, axis=-1) > 0.5, dtype=true_conf.dtype)
                * true_conf,
                axis=(1, 2, 3)), (tf.reduce_sum(true_conf, axis=(1, 2, 3))))))
    avg_iou =  tf.stop_gradient(tf.math.divide_no_nan(
        tf.reduce_sum(iou),
        tf.cast(
            tf.math.count_nonzero(tf.cast(iou > 0, dtype=y_pred.dtype)),
            dtype=y_pred.dtype)))
    return loss, loss_box, conf_loss, class_loss, avg_iou, recall50

"""Contains common building blocks for yolo neural networks."""
import tensorflow as tf
import tensorflow.keras as ks
import tensorflow.keras.backend as K

from yolo.losses.yolo_loss import Yolo_Loss
from yolo.ops import box_ops as box_utils
from yolo.ops import loss_utils, nms_ops


@ks.utils.register_keras_serializable(package='yolo')
class YoloLayer(ks.Model):

  def __init__(self,
               masks,
               anchors,
               classes,
               iou_thresh=0.0,
               ignore_thresh=0.7,
               nms_thresh=0.6,
               max_delta=10.0,
               loss_type='ciou',
               use_tie_breaker=True,
               iou_normalizer=1.0,
               cls_normalizer=1.0,
               obj_normalizer=1.0,
               max_boxes=200,
               path_scale=None,
               scale_xy=None,
               use_nms=True,
               **kwargs):
    super().__init__(**kwargs)
    self._masks = masks
    self._anchors = anchors
    self._thresh = iou_thresh
    self._ignore_thresh = ignore_thresh
    self._iou_normalizer = iou_normalizer
    self._cls_normalizer = cls_normalizer
    self._obj_normalizer = obj_normalizer
    self._nms_thresh = 1 - nms_thresh
    self._max_boxes = max_boxes
    self._max_delta = max_delta
    self._classes = classes
    self._loss_type = loss_type
    self._use_tie_breaker = use_tie_breaker
    self._keys = list(masks.keys())
    self._len_keys = len(self._keys)
    self._path_scale = path_scale or {
        key: 2**int(key) for key, _ in masks.items()
    }
    self._use_nms = use_nms
    self._scale_xy = scale_xy or {key: 1.0 for key, _ in masks.items()}

    print("detget", self._scale_xy)

    self._generator = {}
    self._len_mask = {}
    for key in self._keys:
      anchors = [self._anchors[mask] for mask in self._masks[key]]
      self._generator[key] = self.get_generators(anchors, self._path_scale[key],
                                                 key)
      self._len_mask[key] = len(self._masks[key])
    return

  def get_generators(self, anchors, path_scale, path_key):
    anchor_generator = loss_utils.GridGenerator(
        anchors, scale_anchors=path_scale)
    return anchor_generator

  def rm_nan_inf(self, x, val=0.0):
    x = tf.where(tf.math.is_nan(x), tf.cast(val, dtype=x.dtype), x)
    x = tf.where(tf.math.is_inf(x), tf.cast(val, dtype=x.dtype), x)
    return x

  def parse_yolo_box_predictions(self,
                                 unscaled_box,
                                 width,
                                 height,
                                 anchor_grid,
                                 grid_points,
                                 scale_x_y=1.0):

    ubxy, pred_wh = tf.split(unscaled_box, 2, axis=-1)
    pred_xy = tf.math.sigmoid(ubxy) * scale_x_y - 0.5 * (scale_x_y - 1)
    x, y = tf.split(pred_xy, 2, axis=-1)
    box_xy = tf.concat([x / width, y / height], axis=-1) + grid_points
    box_wh = tf.math.exp(pred_wh) * anchor_grid
    pred_box = K.concatenate([box_xy, box_wh], axis=-1)
    pred_box = box_utils.xcycwh_to_yxyx(pred_box)
    return pred_box

  def parse_prediction_path(self, generator, len_mask, scale_xy, inputs):
    shape = tf.shape(inputs)
    # reshape the yolo output to (batchsize, width, height, number_anchors, remaining_points)
    batchsize, height, width = shape[0], shape[1], shape[2]
    data = tf.reshape(inputs,
                      [batchsize, height, width, len_mask, self._classes + 5])
    centers, anchors = generator(height, width, batchsize, dtype=data.dtype)
    #boxes, obns_scores, class_scores = tf.split(data, [4, 1, -1], axis=-1)
    boxes, obns_scores, class_scores = tf.split(
        data, [4, 1, self._classes], axis=-1)
    classes = tf.shape(class_scores)[-1]

    boxes = self.parse_yolo_box_predictions(
        boxes,
        tf.cast(height, data.dtype),
        tf.cast(width, data.dtype),
        anchors,
        centers,
        scale_x_y=scale_xy)
    obns_scores = tf.math.sigmoid(obns_scores)
    class_scores = tf.math.sigmoid(class_scores) * obns_scores

    boxes = tf.reshape(boxes, [shape[0], -1, 4])
    class_scores = tf.reshape(class_scores, [shape[0], -1, classes])
    obns_scores = tf.reshape(obns_scores, [shape[0], -1])
    return obns_scores, boxes, class_scores

  def call(self, inputs):
    boxes = []
    class_scores = []
    object_scores = []
    levels = list(inputs.keys())
    min_level = int(min(levels))
    max_level = int(max(levels))

    for i in range(min_level, max_level + 1):
      key = str(i)
      object_scores_, boxes_, class_scores_ = self.parse_prediction_path(
          self._generator[key], self._len_mask[key], self._scale_xy[key],
          inputs[key])
      boxes.append(boxes_)
      class_scores.append(class_scores_)
      object_scores.append(object_scores_)

    boxes = tf.concat(boxes, axis=1)
    object_scores = K.concatenate(object_scores, axis=1)
    class_scores = K.concatenate(class_scores, axis=1)

    boxes, class_scores, object_scores = nms_ops.nms(
        boxes,
        class_scores,
        object_scores,
        self._max_boxes,
        self._thresh,
        self._nms_thresh,
        use_classes=True)

    return {
        'bbox': boxes,
        'classes': class_scores,
        'confidence': object_scores,
    }

  @property
  def losses(self):
    loss_dict = {}
    for key in self._keys:
      loss_dict[key] = Yolo_Loss(
          classes=self._classes,
          anchors=self._anchors,
          ignore_thresh=self._ignore_thresh,
          loss_type=self._loss_type,
          iou_normalizer=self._iou_normalizer,
          cls_normalizer=self._cls_normalizer,
          obj_normalizer=self._obj_normalizer,
          path_key=key,
          mask=self._masks[key],
          max_delta=self._max_delta,
          scale_anchors=self._path_scale[key],
          scale_x_y=self._scale_xy[key],
          use_tie_breaker=self._use_tie_breaker)
    return loss_dict

  @property
  def key_dict(self):
    return self._scale_xy

  def get_config(self):
    return {
        'masks': dict(self._masks),
        'anchors': [list(a) for a in self._anchors],
        'thresh': self._thresh,
        'max_boxes': self._max_boxes,
    }

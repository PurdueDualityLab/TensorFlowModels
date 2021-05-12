import tensorflow as tf

from yolo.ops import box_ops as box_ops

NMS_TILE_SIZE = 512


def aggregated_comparitive_iou(boxes1, boxes2=None, iou_type=0, xyxy=True):
  k = tf.shape(boxes1)[-2]

  boxes1 = tf.expand_dims(boxes1, axis=-2)
  boxes1 = tf.tile(boxes1, [1, 1, k, 1])

  if boxes2 is not None:
    boxes2 = tf.expand_dims(boxes2, axis=-2)
    boxes2 = tf.tile(boxes2, [1, 1, k, 1])
    boxes2 = tf.transpose(boxes2, perm=(0, 2, 1, 3))
  else:
    boxes2 = tf.transpose(boxes1, perm=(0, 2, 1, 3))

  if iou_type == 0:  #diou
    _, iou = box_ops.compute_diou(boxes1, boxes2, yxyx=True)
  elif iou_type == 1:  #giou
    _, iou = box_ops.compute_giou(boxes1, boxes2, yxyx=True)
  else:
    iou = box_ops.compute_iou(boxes1, boxes2, yxyx=True)
  return iou


def sort_drop(objectness, box, classificationsi, k):
  objectness, ind = tf.math.top_k(objectness, k=k)

  ind_m = tf.ones_like(ind) * tf.expand_dims(
      tf.range(0,
               tf.shape(objectness)[0]), axis=-1)
  bind = tf.stack([tf.reshape(ind_m, [-1]), tf.reshape(ind, [-1])], axis=-1)

  box = tf.gather_nd(box, bind)
  classifications = tf.gather_nd(classificationsi, bind)

  bsize = tf.shape(ind)[0]
  box = tf.reshape(box, [bsize, k, -1])
  classifications = tf.reshape(classifications, [bsize, k, -1])
  return objectness, box, classifications


def segment_nms(boxes, classes, confidence, k, iou_thresh):
  mrange = tf.range(k)
  mask_x = tf.tile(
      tf.transpose(tf.expand_dims(mrange, axis=-1), perm=[1, 0]), [k, 1])
  mask_y = tf.tile(tf.expand_dims(mrange, axis=-1), [1, k])
  mask_diag = tf.expand_dims(mask_x > mask_y, axis=0)

  iou = aggregated_comparitive_iou(boxes, iou_type=0)

  # duplicate boxes
  iou_mask = iou >= iou_thresh
  iou_mask = tf.logical_and(mask_diag, iou_mask)
  iou *= tf.cast(iou_mask, iou.dtype)

  can_suppress_others = 1 - tf.cast(
      tf.reduce_any(iou_mask, axis=-2), boxes.dtype)

  # iou_sum = tf.reduce_sum(iou, [1])

  # option 1
  # can_suppress_others = tf.expand_dims(can_suppress_others, axis=-1)
  # supressed_i = can_suppress_others * iou
  # supressed = tf.reduce_max(supressed_i, -2) <= 0.5
  # raw = tf.cast(supressed, boxes.dtype)

  # option 2
  raw = tf.cast(can_suppress_others, boxes.dtype)

  boxes *= tf.expand_dims(raw, axis=-1)
  confidence *= tf.cast(raw, confidence.dtype)
  classes *= tf.cast(raw, classes.dtype)

  return boxes, classes, confidence


def nms(boxes,
        classes,
        confidence,
        k,
        pre_nms_thresh,
        nms_thresh,
        limit_pre_thresh=False,
        use_classes=True):

  if limit_pre_thresh:
    confidence, boxes, classes = sort_drop(confidence, boxes, classes, k)

  mask = tf.fill(
      tf.shape(confidence), tf.cast(pre_nms_thresh, dtype=confidence.dtype))
  mask = tf.math.ceil(tf.nn.relu(confidence - mask))
  confidence = confidence * mask
  mask = tf.expand_dims(mask, axis=-1)
  boxes = boxes * mask
  classes = classes * mask

  if use_classes:
    confidence = tf.reduce_max(classes, axis=-1)
  confidence, boxes, classes = sort_drop(confidence, boxes, classes, k)

  classes = tf.cast(tf.argmax(classes, axis=-1), tf.float32)
  boxes, classes, confidence = segment_nms(boxes, classes, confidence, k,
                                           nms_thresh)
  confidence, boxes, classes = sort_drop(confidence, boxes, classes, k)
  classes = tf.squeeze(classes, axis=-1)
  return boxes, classes, confidence

import tensorflow as tf

from yolo.ops import box_ops as box_ops


def aggregated_comparitive_iou(boxes1, boxes2=None, iou_type="iou", xyxy=True):
  k = tf.shape(boxes1)[-2]

  boxes1 = tf.expand_dims(boxes1, axis=-2)
  boxes1 = tf.repeat(boxes1, [k], axis=-2)

  if boxes2 is not None:
    boxes2 = tf.expand_dims(boxes2, axis=-2)
    boxes2 = tf.repeat(boxes2, [k], axis=-2)
    boxes2 = tf.transpose(boxes2, perm=(0, 2, 1, 3))
  else:
    boxes2 = tf.transpose(boxes1, perm=(0, 2, 1, 3))

  if iou_type == "diou":
    _, iou = box_ops.compute_diou(boxes1, boxes2, yxyx=True)
  elif iou_type == "ciou":
    _, iou = box_ops.compute_ciou(boxes1, boxes2, yxyx=True)
  elif iou_type == "giou":
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


def segment_nms(boxes, classes, confidence, k, class_thresh, iou_thresh):
  mrange = tf.range(k)
  mask_x = tf.repeat(
      tf.transpose(tf.expand_dims(mrange, axis=-1), perm=[1, 0]), [k], axis=0)
  mask_y = tf.repeat(tf.expand_dims(mrange, axis=-1), [k], axis=1)
  mask_diag = tf.expand_dims(mask_x > mask_y, axis=0)

  iou = aggregated_comparitive_iou(boxes, iou_type="diou")

  # duplicate boxes
  iou_mask = iou >= iou_thresh
  iou_mask = tf.logical_and(mask_diag, iou_mask)
  iou *= tf.cast(iou_mask, iou.dtype)

  can_suppress_others = 1 - tf.cast(
      tf.reduce_any(iou_mask, axis=-2), boxes.dtype)

  iou_sum = tf.reduce_sum(iou, [1])

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
        class_thresh,
        iou_thresh,
        sorted=False,
        one_hot=True):
  if one_hot:
    confidence = tf.reduce_max(classes, axis=-1)
    classes = tf.cast(tf.argmax(classes, axis=-1), tf.float32)

  if not sorted:
    confidence, boxes, classes = sort_drop(confidence, boxes, classes, k)
    classes = tf.squeeze(classes, axis=-1)

  box_l, class_l, conf_l = segment_nms(boxes, classes, confidence, k,
                                       class_thresh, iou_thresh)
  conf_l, box_l, class_l = sort_drop(conf_l, box_l, class_l, k)
  class_l = tf.squeeze(class_l, axis=-1)

  return box_l, class_l, conf_l


# def nms2(boxes,
#          classes,
#          confidence,
#          k,
#          class_thresh,
#          iou_thresh,
#          sorted=False,
#          one_hot=True):
#   if not sorted:
#     confidence, boxes, classes = sort_drop(confidence, boxes, classes, k)

#   if one_hot:
#     confidence = tf.reduce_max(classes, axis=-1)
#     classes = tf.argmax(classes, axis=-1)

#   scores, boxes, classes = sorted_non_max_suppression_padded(
#       confidence, boxes, classes, k, iou_thresh)
#   return boxes, classes, scores

# # very very slow
# def _nms(boxes, classif, confid, k, class_thresh, iou_thresh):
#   batch_size = tf.shape(boxes)[0]
#   classes = tf.cast(classif, dtype=boxes.dtype)
#   confidence = tf.cast(confid, dtype=boxes.dtype)

#   # u_box = tf.zeros_like(boxes)
#   # u_class = tf.zeros_like(classes)
#   # u_conf = tf.zeros_like(confidence)

#   #box_thresh = tf.cast(tf.ones_like(confidence) - tf.math.ceil(confidence), tf.bool)
#   box_thresh = tf.cast(tf.math.ceil(confidence), tf.bool)
#   lim = tf.reduce_sum(tf.cast(tf.reduce_any(box_thresh, axis=0), tf.int32))

#   #box_thresh = tf.cast(tf.math.ceil(confidence), tf.bool)

#   i = 0
#   added_boxes = 0

#   tiledID = tf.repeat(tf.convert_to_tensor([added_boxes]), [batch_size])
#   mask = tf.one_hot(tiledID, k, dtype=boxes.dtype, axis=-1)

#   u_conf = confidence * mask
#   u_class = classes * mask
#   u_box = boxes * tf.expand_dims(mask, axis=-1)
#   one_mask = mask

#   one_mask = tf.cast(one_mask, tf.bool)

#   i += 1
#   added_boxes += 1

#   # using continue gives off performance jitters
#   while (i < lim):

#     tiledID = tf.repeat(tf.convert_to_tensor([added_boxes]), [batch_size])
#     mask = tf.one_hot(tiledID, k, dtype=u_box.dtype, axis=-1)

#     conf = confidence * mask
#     clas = classes * mask
#     boxs = boxes * tf.expand_dims(mask, axis=-1)
#     boxs = tf.expand_dims(tf.reduce_max(boxs, axis=1), axis=1)

#     conf_mask = tf.cast(conf > class_thresh, dtype=boxes.dtype)

#     # the issue is here along with the mask, we need to select the box..
#     iou = box_ops.compute_iou(u_box, boxs, yxyx=True)

#     iou_mask = iou <= iou_thresh
#     class_mask = tf.logical_and(clas == u_class, one_mask)

#     plus_mask = tf.reduce_all(iou_mask, axis=-1)
#     plus_mult = tf.expand_dims(tf.cast(plus_mask, boxes.dtype), axis=-1)
#     plus_mask = (plus_mult * conf_mask)

#     change_mask = tf.logical_and(tf.logical_not(iou_mask), one_mask)
#     change_mask = tf.logical_and(change_mask, class_mask)
#     change_mask = tf.logical_and(change_mask,
#                                  tf.logical_and(conf > conf_mask, one_mask))

#     u_box = update_mask(tf.expand_dims(change_mask, axis=-1), u_box, boxs)
#     u_class = update_mask(change_mask, u_class, clas)
#     u_conf = update_mask(change_mask, u_conf, conf)

#     u_box = add_mask(tf.expand_dims(plus_mask, axis=-1), u_box, boxs)
#     u_class = add_mask(plus_mask, u_class, clas)
#     u_conf = add_mask(plus_mask, u_conf, conf)

#     if tf.reduce_any(tf.cast(plus_mask, tf.bool)):
#       added_boxes += 1
#       one_mask = tf.logical_or(one_mask, tf.cast(mask, tf.bool))

#     i += 1

#   u_class = tf.cast(u_class, classif.dtype)
#   u_conf = tf.cast(u_conf, confid.dtype)

#   return u_box, u_class, u_conf

# NMS_TILE_SIZE = 512

# def _self_suppression(iou, _, iou_sum):
#   batch_size = tf.shape(iou)[0]
#   can_suppress_others = tf.cast(
#       tf.reshape(tf.reduce_max(iou, 1) <= 0.5, [batch_size, -1, 1]), iou.dtype)
#   iou_suppressed = tf.reshape(
#       tf.cast(tf.reduce_max(can_suppress_others * iou, 1) <= 0.5, iou.dtype),
#       [batch_size, -1, 1]) * iou
#   iou_sum_new = tf.reduce_sum(iou_suppressed, [1, 2])
#   return [
#       iou_suppressed,
#       tf.reduce_any(iou_sum - iou_sum_new > 0.5), iou_sum_new
#   ]

# def _cross_suppression(boxes, box_slice, iou_threshold, inner_idx):
#   batch_size = tf.shape(boxes)[0]
#   new_slice = tf.slice(boxes, [0, inner_idx * NMS_TILE_SIZE, 0],
#                        [batch_size, NMS_TILE_SIZE, 4])
#   #iou = box_ops.bbox_overlap(new_slice, box_slice)
#   iou = aggregated_comparitive_iou(new_slice, box_slice, iou_type="diou")
#   ret_slice = tf.expand_dims(
#       tf.cast(
#           tf.logical_not(tf.reduce_any(iou < iou_threshold, [1])),
#           box_slice.dtype), 2) * box_slice
#   return boxes, ret_slice, iou_threshold, inner_idx + 1

# def _suppression_loop_body(boxes, iou_threshold, output_size, idx):
#   """Process boxes in the range [idx*NMS_TILE_SIZE, (idx+1)*NMS_TILE_SIZE).

#   Args:
#     boxes: a tensor with a shape of [batch_size, anchors, 4].
#     iou_threshold: a float representing the threshold for deciding whether boxes
#       overlap too much with respect to IOU.
#     output_size: an int32 tensor of size [batch_size]. Representing the number
#       of selected boxes for each batch.
#     idx: an integer scalar representing induction variable.

#   Returns:
#     boxes: updated boxes.
#     iou_threshold: pass down iou_threshold to the next iteration.
#     output_size: the updated output_size.
#     idx: the updated induction variable.
#   """
#   num_tiles = tf.shape(boxes)[1] // NMS_TILE_SIZE
#   batch_size = tf.shape(boxes)[0]

#   # Iterates over tiles that can possibly suppress the current tile.
#   box_slice = tf.slice(boxes, [0, idx * NMS_TILE_SIZE, 0],
#                        [batch_size, NMS_TILE_SIZE, 4])
#   _, box_slice, _, _ = tf.while_loop(
#       lambda _boxes, _box_slice, _threshold, inner_idx: inner_idx < idx,
#       _cross_suppression, [boxes, box_slice, iou_threshold,
#                            tf.constant(0)])

#   # Iterates over the current tile to compute self-suppression.
#   # iou = box_ops.bbox_overlap(box_slice, box_slice)
#   iou = aggregated_comparitive_iou(box_slice, box_slice, iou_type="diou")
#   mask = tf.expand_dims(
#       tf.reshape(tf.range(NMS_TILE_SIZE), [1, -1]) > tf.reshape(
#           tf.range(NMS_TILE_SIZE), [-1, 1]), 0)
#   iou *= tf.cast(tf.logical_and(mask, iou >= iou_threshold), iou.dtype)
#   suppressed_iou, _, _ = tf.while_loop(
#       lambda _iou, loop_condition, _iou_sum: loop_condition, _self_suppression,
#       [iou, tf.constant(True),
#        tf.reduce_sum(iou, [1, 2])])
#   suppressed_box = tf.reduce_sum(suppressed_iou, 1) > 0
#   box_slice *= tf.expand_dims(1.0 - tf.cast(suppressed_box, box_slice.dtype), 2)

#   # Uses box_slice to update the input boxes.
#   mask = tf.reshape(
#       tf.cast(tf.equal(tf.range(num_tiles), idx), boxes.dtype), [1, -1, 1, 1])
#   boxes = tf.tile(tf.expand_dims(
#       box_slice, [1]), [1, num_tiles, 1, 1]) * mask + tf.reshape(
#           boxes, [batch_size, num_tiles, NMS_TILE_SIZE, 4]) * (1 - mask)
#   boxes = tf.reshape(boxes, [batch_size, -1, 4])

#   # Updates output_size.
#   output_size += tf.reduce_sum(
#       tf.cast(tf.reduce_any(box_slice > 0, [2]), tf.int32), [1])
#   return boxes, iou_threshold, output_size, idx + 1

# def sorted_non_max_suppression_padded(scores, boxes, classes, max_output_size,
#                                       iou_threshold):
#   """A wrapper that handles non-maximum suppression.

#   Assumption:
#     * The boxes are sorted by scores unless the box is a dot (all coordinates
#       are zero).
#     * Boxes with higher scores can be used to suppress boxes with lower scores.

#   The overal design of the algorithm is to handle boxes tile-by-tile:

#   boxes = boxes.pad_to_multiply_of(tile_size)
#   num_tiles = len(boxes) // tile_size
#   output_boxes = []
#   for i in range(num_tiles):
#     box_tile = boxes[i*tile_size : (i+1)*tile_size]
#     for j in range(i - 1):
#       suppressing_tile = boxes[j*tile_size : (j+1)*tile_size]
#       iou = bbox_overlap(box_tile, suppressing_tile)
#       # if the box is suppressed in iou, clear it to a dot
#       box_tile *= _update_boxes(iou)
#     # Iteratively handle the diagnal tile.
#     iou = _box_overlap(box_tile, box_tile)
#     iou_changed = True
#     while iou_changed:
#       # boxes that are not suppressed by anything else
#       suppressing_boxes = _get_suppressing_boxes(iou)
#       # boxes that are suppressed by suppressing_boxes
#       suppressed_boxes = _get_suppressed_boxes(iou, suppressing_boxes)
#       # clear iou to 0 for boxes that are suppressed, as they cannot be used
#       # to suppress other boxes any more
#       new_iou = _clear_iou(iou, suppressed_boxes)
#       iou_changed = (new_iou != iou)
#       iou = new_iou
#     # remaining boxes that can still suppress others, are selected boxes.
#     output_boxes.append(_get_suppressing_boxes(iou))
#     if len(output_boxes) >= max_output_size:
#       break

#   Args:
#     scores: a tensor with a shape of [batch_size, anchors].
#     boxes: a tensor with a shape of [batch_size, anchors, 4].
#     max_output_size: a scalar integer `Tensor` representing the maximum number
#       of boxes to be selected by non max suppression.
#     iou_threshold: a float representing the threshold for deciding whether boxes
#       overlap too much with respect to IOU.

#   Returns:
#     nms_scores: a tensor with a shape of [batch_size, anchors]. It has same
#       dtype as input scores.
#     nms_proposals: a tensor with a shape of [batch_size, anchors, 4]. It has
#       same dtype as input boxes.
#   """
#   batch_size = tf.shape(boxes)[0]
#   num_boxes = tf.shape(boxes)[1]
#   pad = tf.cast(
#       tf.math.ceil(tf.cast(num_boxes, tf.float32) / NMS_TILE_SIZE),
#       tf.int32) * NMS_TILE_SIZE - num_boxes
#   boxes = tf.pad(tf.cast(boxes, tf.float32), [[0, 0], [0, pad], [0, 0]])
#   scores = tf.pad(
#       tf.cast(scores, tf.float32), [[0, 0], [0, pad]], constant_values=-1)
#   classes = tf.pad(
#       tf.cast(classes, tf.float32), [[0, 0], [0, pad]], constant_values=-1)
#   num_boxes += pad

#   def _loop_cond(unused_boxes, unused_threshold, output_size, idx):
#     return tf.logical_and(
#         tf.reduce_min(output_size) < max_output_size,
#         idx < num_boxes // NMS_TILE_SIZE)

#   selected_boxes, _, output_size, _ = tf.while_loop(
#       _loop_cond, _suppression_loop_body,
#       [boxes, iou_threshold,
#        tf.zeros([batch_size], tf.int32),
#        tf.constant(0)])
#   idx = num_boxes - tf.cast(
#       tf.nn.top_k(
#           tf.cast(tf.reduce_any(selected_boxes > 0, [2]), tf.int32) *
#           tf.expand_dims(tf.range(num_boxes, 0, -1), 0), max_output_size)[0],
#       tf.int32)
#   idx = tf.minimum(idx, num_boxes - 1)
#   idx = tf.reshape(idx + tf.reshape(tf.range(batch_size) * num_boxes, [-1, 1]),
#                    [-1])
#   boxes = tf.reshape(
#       tf.gather(tf.reshape(boxes, [-1, 4]), idx),
#       [batch_size, max_output_size, 4])
#   boxes = boxes * tf.cast(
#       tf.reshape(tf.range(max_output_size), [1, -1, 1]) < tf.reshape(
#           output_size, [-1, 1, 1]), boxes.dtype)
#   scores = tf.reshape(
#       tf.gather(tf.reshape(scores, [-1, 1]), idx),
#       [batch_size, max_output_size])
#   scores = scores * tf.cast(
#       tf.reshape(tf.range(max_output_size), [1, -1]) < tf.reshape(
#           output_size, [-1, 1]), scores.dtype)
#   classes = tf.reshape(
#       tf.gather(tf.reshape(classes, [-1, 1]), idx),
#       [batch_size, max_output_size])
#   classes = classes * tf.cast(
#       tf.reshape(tf.range(max_output_size), [1, -1]) < tf.reshape(
#           output_size, [-1, 1]), classes.dtype)
#   return scores, boxes, classes

import tensorflow as tf
from yolo.ops import box_ops
from yolo.ops import preprocessing_ops
from yolo.ops import loss_utils


def get_best_anchor(y_true,
                    anchors,
                    stride,
                    width=1,
                    height=1,
                    iou_thresh=0.25,
                    best_match_only=False, 
                    use_tie_breaker=True):
  """
  get the correct anchor that is assoiciated with each box using IOU
  
  Args:
    y_true: tf.Tensor[] for the list of bounding boxes in the yolo format
    anchors: list or tensor for the anchor boxes to be used in prediction
      found via Kmeans
    width: int for the image width
    height: int for the image height
  Return:
    tf.Tensor: y_true with the anchor associated with each ground truth
    box known
  """
  with tf.name_scope('get_best_anchor'):
    is_batch = True
    ytrue_shape = y_true.get_shape()
    if ytrue_shape.ndims == 2:
      is_batch = False
      y_true = tf.expand_dims(y_true, 0)
    elif ytrue_shape.ndims is None:
      is_batch = False
      y_true = tf.expand_dims(y_true, 0)
      y_true.set_shape([None] * 3)
    elif ytrue_shape.ndims != 3:
      raise ValueError('\'box\' (shape %s) must have either 3 or 4 dimensions.')

    width = tf.cast(width, dtype=tf.float32)
    height = tf.cast(height, dtype=tf.float32)
    scaler = tf.convert_to_tensor([width, height])

    true_wh = tf.cast(y_true[..., 2:4], dtype=tf.float32) * scaler
    anchors = tf.cast(anchors, dtype=tf.float32)/stride

    k = tf.shape(anchors)[0]

    anchors = tf.expand_dims(
        tf.concat([tf.zeros_like(anchors), anchors], axis=-1), axis=0)
    truth_comp = tf.concat([tf.zeros_like(true_wh), true_wh], axis=-1)

    if iou_thresh >= 1.0:
      anchors = tf.expand_dims(anchors, axis=-2)
      truth_comp = tf.expand_dims(truth_comp, axis=-3)

      aspect = truth_comp[..., 2:4] / anchors[..., 2:4]
      aspect = tf.where(tf.math.is_nan(aspect), tf.zeros_like(aspect), aspect)
      aspect = tf.maximum(aspect, 1 / aspect)
      aspect = tf.where(tf.math.is_nan(aspect), tf.zeros_like(aspect), aspect)
      aspect = tf.reduce_max(aspect, axis=-1)

      values, indexes = tf.math.top_k(
          tf.transpose(-aspect, perm=[0, 2, 1]),
          k=tf.cast(k, dtype=tf.int32),
          sorted=True)
      values = -values
      ind_mask = tf.cast(values < iou_thresh, dtype=indexes.dtype)
    else:
      # iou_raw = box_ops.compute_iou(truth_comp, anchors)
      truth_comp = box_ops.xcycwh_to_yxyx(truth_comp)
      anchors = box_ops.xcycwh_to_yxyx(anchors)
      iou_raw = box_ops.aggregated_comparitive_iou(
          truth_comp,
          anchors,
          iou_type=3,
      )
      values, indexes = tf.math.top_k(
          iou_raw,  #tf.transpose(iou_raw, perm=[0, 2, 1]),
          k=tf.cast(k, dtype=tf.int32),
          sorted=True)
      ind_mask = tf.cast(values >= iou_thresh, dtype=indexes.dtype)

    # pad the indexs such that all values less than the thresh are -1
    # add one, multiply the mask to zeros all the bad locations
    # subtract 1 makeing all the bad locations 0.
    if best_match_only:
      iou_index = ((indexes[..., 0:] + 1) * ind_mask[..., 0:]) - 1
    elif use_tie_breaker:
      iou_index = tf.concat([
          tf.expand_dims(indexes[..., 0], axis=-1),
          ((indexes[..., 1:] + 1) * ind_mask[..., 1:]) - 1], axis=-1)
    else:
      iou_index = tf.concat([
          tf.expand_dims(indexes[..., 0], axis=-1),
          tf.zeros_like(indexes[..., 1:]) - 1], axis=-1)

    true_prod = tf.reduce_prod(true_wh, axis=-1, keepdims=True)
    iou_index = tf.where(true_prod > 0, iou_index, tf.zeros_like(iou_index) - 1)

    if not is_batch:
      iou_index = tf.squeeze(iou_index, axis=0)
      values = tf.squeeze(values, axis=0)
  return tf.cast(iou_index, dtype=tf.float32), tf.cast(values, dtype=tf.float32)

def _write_anchor_free_grid(boxes,
                            classes,
                            height,
                            width,
                            stride,
                            fpn_limits,
                            center_radius=2.5):
  """Iterate all boxes and write to grid without anchors boxes."""
  gen = loss_utils.GridGenerator(
      masks=None, anchors=[[1, 1]], scale_anchors=stride)
  grid_points = gen(width, height, 1, boxes.dtype)[0]
  grid_points = tf.squeeze(grid_points, axis=0)
  box_list = boxes
  class_list = classes

  grid_points = (grid_points + 0.5) * stride
  x_centers, y_centers = grid_points[..., 0], grid_points[..., 1]
  boxes *= (tf.convert_to_tensor([width, height, width, height]) * stride)
  tlbr_boxes = box_ops.xcycwh_to_yxyx(boxes)

  boxes = tf.reshape(boxes, [1, 1, -1, 4])
  tlbr_boxes = tf.reshape(tlbr_boxes, [1, 1, -1, 4])
  mask = tf.reshape(class_list != -1, [1, 1, -1])

  # check if the box is in the receptive feild of the this fpn level
  b_t = y_centers - tlbr_boxes[..., 0]
  b_l = x_centers - tlbr_boxes[..., 1]
  b_b = tlbr_boxes[..., 2] - y_centers
  b_r = tlbr_boxes[..., 3] - x_centers
  box_delta = tf.stack([b_t, b_l, b_b, b_r], axis=-1)
  if fpn_limits is not None:
    max_reg_targets_per_im = tf.reduce_max(box_delta, axis=-1)
    gt_min = max_reg_targets_per_im >= fpn_limits[0]
    gt_max = max_reg_targets_per_im <= fpn_limits[1]
    is_in_boxes = tf.logical_and(gt_min, gt_max)
  else:
    is_in_boxes = tf.reduce_min(box_delta, axis=-1) > 0.0
  is_in_boxes = tf.logical_and(is_in_boxes, mask)
  is_in_boxes_all = tf.reduce_any(is_in_boxes, axis=(0, 1), keepdims=True)

  # check if the center is in the receptive feild of the this fpn level
  c_t = y_centers - (boxes[..., 1] - center_radius * stride)
  c_l = x_centers - (boxes[..., 0] - center_radius * stride)
  c_b = (boxes[..., 1] + center_radius * stride) - y_centers
  c_r = (boxes[..., 0] + center_radius * stride) - x_centers
  centers_delta = tf.stack([c_t, c_l, c_b, c_r], axis=-1)
  is_in_centers = tf.reduce_min(centers_delta, axis=-1) > 0.0
  is_in_centers = tf.logical_and(is_in_centers, mask)
  is_in_centers_all = tf.reduce_any(is_in_centers, axis=(0, 1), keepdims=True)

  # colate all masks to get the final locations
  is_in_index = tf.logical_or(is_in_boxes_all, is_in_centers_all)
  is_in_boxes_and_center = tf.logical_and(is_in_boxes, is_in_centers)
  is_in_boxes_and_center = tf.logical_and(is_in_index, is_in_boxes_and_center)

  # construct the index update grid
  reps = tf.reduce_sum(tf.cast(is_in_boxes_and_center, tf.int16), axis=-1)
  indexes = tf.cast(tf.where(is_in_boxes_and_center), tf.int32)
  y, x, t = tf.split(indexes, 3, axis=-1)

  boxes = tf.gather_nd(box_list, t)
  classes = tf.cast(tf.gather_nd(class_list, t), boxes.dtype)
  reps = tf.gather_nd(reps, tf.concat([y, x], axis=-1))
  reps = tf.cast(tf.expand_dims(reps, axis=-1), boxes.dtype)
  conf = tf.ones_like(classes)

  # return the samples and the indexes
  samples = tf.concat([boxes, conf, classes, conf, reps], axis=-1)
  indexes = tf.concat([y, x, tf.zeros_like(t)], axis=-1)
  return indexes, samples

class YoloAnchorLabeler:
  def __init__(self, 
               match_threshold = 0.25, 
               best_matches_only = False, 
               use_tie_breaker = True):
    self.match_threshold = match_threshold
    self.best_matches_only = best_matches_only
    self.use_tie_breaker = use_tie_breaker

  def _get_anchor_id(self, boxes, classes, anchors, width, height, stride):
    iou_index, ious = get_best_anchor(boxes, anchors, stride,
                                      width=width, height=height, 
                                      best_match_only=self.best_matches_only, 
                                      use_tie_breaker=self.use_tie_breaker,
                                      iou_thresh=self.match_threshold)


    num_anchors = len(anchors)
    classes = tf.cast(tf.expand_dims(classes, axis = -1), boxes.dtype)

    boxes = tf.tile(tf.expand_dims(boxes, axis = -2), [1, num_anchors, 1])
    classes = tf.tile(tf.expand_dims(classes, axis = -2), [1, num_anchors, 1])
    indexes = tf.expand_dims(iou_index, axis = -1)
    ious = tf.expand_dims(ious, axis = -1)

    boxes_and_anchors = tf.concat([boxes, classes, ious, indexes], axis = -1)
    boxes_and_anchors = tf.reshape(boxes_and_anchors, [-1, 7])
    _, anchors_ids =  tf.split(boxes_and_anchors, [6, 1], axis = -1)

    anchors_ids = tf.squeeze(anchors_ids, axis = -1)
    select = tf.where(anchors_ids >= 0)
    boxes_and_anchors = tf.gather_nd(boxes_and_anchors, select)

    (boxes, classes, 
    ious, anchors) = tf.split(boxes_and_anchors, [4, 1, 1, 1], axis = -1)
    return boxes, classes, ious, anchors, num_anchors

  def _get_centers(self, boxes, classes, anchors, width, height, offset):
    grid_xy, wh = tf.split(boxes, 2, axis = -1)
    wh_scale = tf.cast(tf.convert_to_tensor([width, height]), boxes.dtype)

    grid_xy = grid_xy * wh_scale
    centers = tf.math.floor(grid_xy)

    if offset != 0.0:   
      clamp = lambda x, ma: tf.maximum(
          tf.minimum(x, tf.cast(ma, x.dtype)), tf.zeros_like(x))

      grid_xy_index = grid_xy - centers
      positive_shift = ((grid_xy_index < offset) & (grid_xy > 1.))
      negative_shift = (
        (grid_xy_index > (1 - offset)) & (grid_xy < (wh_scale - 1.)))

      zero , _ = tf.split(tf.ones_like(positive_shift), 2, axis = -1)
      shift_mask = tf.concat(
        [zero, positive_shift, negative_shift], axis = -1)
      offset = tf.cast([[0, 0], [1, 0], 
                        [0, 1], [-1, 0], 
                        [0, -1]], offset.dtype) * offset

      num_shifts = tf.shape(shift_mask)
      num_shifts = num_shifts[-1]
      boxes = tf.tile(tf.expand_dims(boxes, axis = -2), [1, num_shifts, 1])
      classes = tf.tile(tf.expand_dims(classes, axis = -2), [1, num_shifts, 1])
      anchors = tf.tile(tf.expand_dims(anchors, axis = -2), [1, num_shifts, 1])

      shift_mask = tf.cast(shift_mask, boxes.dtype)
      shift_ind = shift_mask * tf.range(0, num_shifts, dtype = boxes.dtype) 
      shift_ind = shift_ind - (1 - shift_mask)
      shift_ind = tf.expand_dims(shift_ind, axis = -1)

      boxes_and_centers = tf.concat(
        [boxes, classes, anchors, shift_ind], axis = -1)
      boxes_and_centers = tf.reshape(boxes_and_centers, [-1, 7])
      _, center_ids =  tf.split(boxes_and_centers, [6, 1], axis = -1)

      #center_ids = tf.squeeze(center_ids, axis = -1)
      select = tf.where(center_ids >= 0)
      select, _ = tf.split(select, 2, axis = -1)

      boxes_and_centers = tf.gather_nd(boxes_and_centers, select)

      # center_ids = tf.cast(center_ids, tf.int32)
      center_ids = tf.gather_nd(center_ids, select)
      center_ids = tf.cast(center_ids, tf.int32)
      shifts = tf.gather_nd(offset, center_ids)

      boxes, classes, anchors, _ = tf.split(boxes_and_centers, 
                                                [4, 1, 1, 1], axis = -1)
      grid_xy, _ = tf.split(boxes, 2, axis = -1)
      centers = tf.math.floor(grid_xy * wh_scale - shifts)
      centers = clamp(centers, wh_scale - 1)
    
    x, y = tf.split(centers, 2, axis = -1)
    centers = tf.cast(tf.concat([y, x, anchors], axis = -1), tf.int32)
    return boxes, classes, centers

  def __call__(self, 
               boxes, 
               classes, 
               anchors, 
               width, 
               height, 
               stride, 
               scale_xy, 
               num_instances, 
               fpn_limits = None):
    boxes = box_ops.yxyx_to_xcycwh(boxes)

    width //= stride
    height //= stride
    width = tf.cast(width, boxes.dtype)
    height = tf.cast(height, boxes.dtype)

    if fpn_limits is None:
      offset = tf.cast(0.5 * (scale_xy - 1), boxes.dtype)
      (boxes, classes, ious, 
       anchors, num_anchors) = self._get_anchor_id(boxes, classes, anchors, 
                                                   width, height, stride)
      boxes, classes, centers = self._get_centers(boxes, classes, anchors, 
                                                  width, height, offset)
      ind_mask = tf.ones_like(classes)
      updates = tf.concat(
        [boxes, ind_mask, classes, ind_mask, ind_mask], axis = -1)
    else:
      (centers, updates) = _write_anchor_free_grid(boxes, classes, height, 
                                                   width, stride, fpn_limits)
      boxes, ind_mask, classes, _ = tf.split(updates, [4, 1, 1, 2], axis = -1)
      num_anchors = 1.0 


    width = tf.cast(width, tf.int32)
    height = tf.cast(height, tf.int32)
    full = tf.zeros([height, width, num_anchors, 1], dtype=classes.dtype)
    full = tf.tensor_scatter_nd_add(full, centers, ind_mask)
    centers = preprocessing_ops.pad_max_instances(
      centers, int(num_instances), pad_value=0, pad_axis=0)
    updates = preprocessing_ops.pad_max_instances(
      updates, int(num_instances), pad_value=0, pad_axis=0)
    return centers, updates, full

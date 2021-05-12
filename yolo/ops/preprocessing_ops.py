import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras.backend as K
from yolo.ops import box_ops
from official.vision.beta.ops import preprocess_ops
from official.vision.beta.ops import box_ops as bbox_ops


def rand_uniform_strong(minval, maxval, dtype=tf.float32):
  if minval > maxval:
    minval, maxval = maxval, minval
  return tf.random.uniform([], minval=minval, maxval=maxval, dtype=dtype)


def rand_scale(val, dtype=tf.float32):
  scale = rand_uniform_strong(1.0, val, dtype=dtype)
  do_ret = tf.random.uniform([], minval=0, maxval=2, dtype=tf.int32)
  if (do_ret == 1):
    return scale
  return 1.0 / scale


def _pad_max_instances(value, instances, pad_value=0, pad_axis=0):
  shape = tf.shape(value)
  if pad_axis < 0:
    pad_axis = tf.rank(value) + pad_axis
  dim1 = shape[pad_axis]
  take = tf.math.reduce_min([instances, dim1])
  value, _ = tf.split(value, [take, -1], axis=pad_axis)
  pad = tf.convert_to_tensor([tf.math.reduce_max([instances - dim1, 0])])
  nshape = tf.concat([shape[:pad_axis], pad, shape[(pad_axis + 1):]], axis=0)
  pad_tensor = tf.fill(nshape, tf.cast(pad_value, dtype=value.dtype))
  value = tf.concat([value, pad_tensor], axis=pad_axis)
  return value


def get_non_empty_box_indices(boxes, output_size=None):
  """Get indices for non-empty boxes."""
  # Selects indices if box height or width is 0.
  if output_size is not None:
    width = tf.cast(output_size[1], boxes.dtype)
    height = tf.cast(output_size[0], boxes.dtype)
    boxes = box_ops.yxyx_to_xcycwh(boxes)
    x, y, w, h = tf.split(boxes, 4, axis=-1)

    indices = tf.where(
        tf.logical_and(
            tf.logical_and(
                tf.logical_and(tf.greater(x, 0), tf.greater(y, 0)),
                tf.logical_and(tf.less(x, width), tf.less(y, height))),
            tf.logical_and(tf.greater(h, 0), tf.greater(w, 0))))
  else:
    height = boxes[:, 2] - boxes[:, 0]
    width = boxes[:, 3] - boxes[:, 1]
    indices = tf.where(
        tf.logical_and(tf.greater(height, 0), tf.greater(width, 0)))
  return indices[:, 0]


def get_image_shape(image):
  shape = tf.shape(image)
  if tf.shape(shape)[0] == 4:
    width = shape[2]
    height = shape[1]
  else:
    width = shape[1]
    height = shape[0]
  return height, width


def random_translate(image, t):
  with tf.name_scope('random_translate'):
    if t != 0:
      t_x = tf.random.uniform(minval=-t, maxval=t, shape=(), dtype=tf.float32)
      t_y = tf.random.uniform(minval=-t, maxval=t, shape=(), dtype=tf.float32)
      height, width = get_image_shape(image)
      image_jitter = tf.convert_to_tensor([t_x, t_y])
      image_dims = tf.cast(tf.convert_to_tensor([width, height]), tf.float32)
      image_jitter.set_shape([2])
      image = tfa.image.translate(image, image_jitter * image_dims)
    else:
      t_x = 0.0
      t_y = 0.0
  return image, t_x, t_y


def translate_boxes(box, classes, translate_x, translate_y):
  with tf.name_scope('translate_boxes'):
    box = box_ops.yxyx_to_xcycwh(box)
    x, y, w, h = tf.split(box, 4, axis=-1)
    x = x + translate_x
    y = y + translate_y
    box = tf.cast(tf.concat([x, y, w, h], axis=-1), box.dtype)
    box = box_ops.xcycwh_to_yxyx(box)
  return box, classes


def translate_image(image, t_y, t_x):
  with tf.name_scope('translate_boxes'):
    height, width = get_image_shape(image)
    image_jitter = tf.convert_to_tensor([t_y, t_x])
    image_dims = tf.cast(tf.convert_to_tensor([height, width]), tf.float32)
    image_jitter.set_shape([2])
    jitter = image_jitter * image_dims
    image = tfa.image.translate(image, jitter)
  return image, jitter[0], jitter[1]

def estimate_shape(image, output_size):
  height, width = get_image_shape(image)
  oheight, owidth = output_size[0], output_size[1]

  if height > width:
    scale = oheight / height
    nwidth = tf.cast(scale * tf.cast(width, scale.dtype), width.dtype)
    if nwidth > owidth:
      scale = owidth / width
      oheight = tf.cast(scale * tf.cast(height, scale.dtype), height.dtype)
    else:
      owidth = nwidth
  else:
    scale = owidth / width
    nheight = tf.cast(scale * tf.cast(height, scale.dtype), height.dtype)
    if nheight > oheight:
      scale = oheight / height
      owidth = tf.cast(scale * tf.cast(width, scale.dtype), width.dtype)
    else:
      oheight = nheight
  return oheight, owidth

def letter_box(image, boxes=None, xs=0.0, ys=0.0, output_size = None):
  # the actual height
  if output_size is None:
    height, width = get_image_shape(image)
    clipper = tf.math.maximum(width, height)
    output_size = [clipper, clipper]

  height, width = estimate_shape(image, output_size)
  # desired height 
  hclipper, wclipper = output_size[0], output_size[1]

  xs = tf.convert_to_tensor(xs)
  ys = tf.convert_to_tensor(ys)
  pad_width_p = wclipper - width
  pad_height_p = hclipper - height
  pad_height = tf.cast(tf.cast(pad_height_p, ys.dtype) * ys, tf.int32)
  pad_width = tf.cast(tf.cast(pad_width_p, xs.dtype) * xs, tf.int32)

  image = tf.image.resize(image, (height, width))
  image = tf.image.pad_to_bounding_box(image, pad_height, pad_width, hclipper,
                                        wclipper)
  info = tf.convert_to_tensor([pad_height, pad_width, height, width])

  if boxes is not None:
    boxes = box_ops.yxyx_to_xcycwh(boxes)
    x, y, w, h = tf.split(boxes, 4, axis=-1)

    y *= tf.cast(height / hclipper, y.dtype)
    x *= tf.cast(width / wclipper, x.dtype)

    y += tf.cast((pad_height / hclipper), y.dtype)
    x += tf.cast((pad_width / wclipper), x.dtype)

    h *= tf.cast(height / hclipper, h.dtype)
    w *= tf.cast(width / wclipper, w.dtype)

    boxes = tf.concat([x, y, w, h], axis=-1)

    boxes = box_ops.xcycwh_to_yxyx(boxes)
  return image, boxes, info


def get_best_anchor2(y_true, anchors, width=1, height=1, iou_thresh=0.20):
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

    width = tf.cast(width, dtype=y_true.dtype)
    height = tf.cast(height, dtype=y_true.dtype)
    # split the boxes into center and width height
    anchor_xy = y_true[..., 0:2]
    true_wh = y_true[..., 2:4]

    # scale thhe boxes
    anchors = tf.convert_to_tensor(anchors, dtype=y_true.dtype)
    anchors_x = anchors[..., 0] / height
    anchors_y = anchors[..., 1] / width
    anchors = tf.stack([anchors_x, anchors_y], axis=-1)
    k = tf.shape(anchors)[0]
    # build a matrix of anchor boxes of shape [num_anchors, num_boxes, 4]
    anchors = tf.transpose(anchors, perm=[1, 0])

    anchor_xy = tf.tile(
        tf.expand_dims(anchor_xy, axis=-1),
        [1, 1, 1, tf.shape(anchors)[-1]])
    anchors = tf.tile(
        tf.expand_dims(anchors, axis=0), [tf.shape(anchor_xy)[1], 1, 1])
    anchors = tf.tile(
        tf.expand_dims(anchors, axis=0), [tf.shape(anchor_xy)[0], 1, 1, 1])

    # stack the xy so, each anchor is asscoaited once with each center from
    # the ground truth input
    anchors = K.concatenate([anchor_xy, anchors], axis=2)
    anchors = tf.transpose(anchors, perm=[0, 3, 1, 2])

    # # copy the gt n times so that each anchor from above can be compared to
    # # input ground truth to shape: [num_anchors, num_boxes, 4]
    truth_comp = tf.tile(
        tf.expand_dims(y_true[..., 0:4], axis=-1),
        [1, 1, 1, tf.shape(anchors)[1]])
    truth_comp = tf.transpose(truth_comp, perm=[0, 3, 1, 2])
    # compute intersection over union of the boxes, and take the argmax of
    # comuted iou for each box. thus each box is associated with the
    # largest interection over union

    if iou_thresh >= 1.0:
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
      iou_raw = box_ops.compute_iou(truth_comp, anchors)
      values, indexes = tf.math.top_k(
          tf.transpose(iou_raw, perm=[0, 2, 1]),
          k=tf.cast(k, dtype=tf.int32),
          sorted=True)
      ind_mask = tf.cast(values >= iou_thresh, dtype=indexes.dtype)

    # pad the indexs such that all values less than the thresh are -1
    # add one, multiply the mask to zeros all the bad locations
    # subtract 1 makeing all the bad locations 0.
    iou_index = tf.concat([
        K.expand_dims(indexes[..., 0], axis=-1),
        ((indexes[..., 1:] + 1) * ind_mask[..., 1:]) - 1
    ],
                          axis=-1)

    true_prod = tf.reduce_prod(true_wh, axis=-1, keepdims=True)
    iou_index = tf.where(true_prod > 0, iou_index, tf.zeros_like(iou_index) - 1)

    # tf.print(iou_index, summarize = -1)
    if not is_batch:
      iou_index = tf.squeeze(iou_index, axis=0)
      values = tf.squeeze(values, axis=0)
  return tf.cast(iou_index, dtype=tf.float32), tf.cast(values, dtype=tf.float32)


def get_best_anchor(y_true, anchors, width=1, height=1, iou_thresh=0.20):
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
    width = tf.cast(width, dtype=y_true.dtype)
    height = tf.cast(height, dtype=y_true.dtype)

    true_wh = y_true[..., 2:4]
    hold = tf.zeros_like(true_wh)
    y_true = tf.concat([hold, true_wh], axis=-1)

    # tf.print(tf.shape(true_wh), tf.shape(anchors))
  return get_best_anchor2(
      y_true, anchors, width=width, height=height, iou_thresh=iou_thresh)


def _get_num_reps(anchors, mask, box_mask):
  mask = tf.expand_dims(mask, 0)
  mask = tf.expand_dims(mask, 0)
  mask = tf.expand_dims(mask, 0)
  box_mask = tf.expand_dims(box_mask, -1)
  box_mask = tf.expand_dims(box_mask, -1)

  anchors = tf.expand_dims(anchors, axis=-1)
  anchors_primary, anchors_alternate = tf.split(anchors, [1, -1], axis=-2)
  fillin = tf.zeros_like(anchors_primary) - 1
  anchors_alternate = tf.concat([fillin, anchors_alternate], axis=-2)

  viable_primary = tf.squeeze(
      tf.logical_and(box_mask, anchors_primary == mask), axis=0)
  viable_alternate = tf.squeeze(
      tf.logical_and(box_mask, anchors_alternate == mask), axis=0)

  viable_primary = tf.where(viable_primary)
  viable_alternate = tf.where(viable_alternate)

  viable = anchors == mask
  acheck = tf.reduce_any(viable, axis=-1)
  reps = tf.squeeze(tf.reduce_sum(tf.cast(acheck, mask.dtype), axis=-1), axis=0)
  return reps, viable_primary, viable_alternate


def _gen_utility(boxes):
  eq0 = tf.reduce_all(tf.math.less_equal(boxes[..., 2:4], 0), axis=-1)
  gtlb = tf.reduce_any(tf.math.less(boxes[..., 0:2], 0.0), axis=-1)
  ltub = tf.reduce_any(tf.math.greater_equal(boxes[..., 0:2], 1.0), axis=-1)

  a = tf.logical_or(eq0, gtlb)
  b = tf.logical_or(a, ltub)
  return tf.logical_not(b)


def _gen_offsets(scale_xy, dtype):
  return tf.cast(0.5 * (scale_xy - 1), dtype)


def build_grided_gt_ind(y_true, mask, size, num_classes, dtype, scale_xy,
                        scale_num_inst, use_tie_breaker):
  """
    convert ground truth for use in loss functions
    Args:
      y_true: tf.Tensor[] ground truth
        [batch, box coords[0:4], classes_onehot[0:-1], best_fit_anchor_box]
      mask: list of the anchor boxes choresponding to the output,
        ex. [1, 2, 3] tells this layer to predict only the first 3 anchors
        in the total.
      size: the dimensions of this output, for regular, it progresses from
        13, to 26, to 52
      num_classes: `integer` for the number of classes
      dtype: expected output datatype
      use_tie_breaker: boolean value for wether or not to use the tie
        breaker

    Return:
      tf.Tensor[] of shape [batch, size, size, #of_anchors, 4, 1, num_classes]
  """
  # unpack required components from the input ground truth
  boxes = tf.cast(y_true['bbox'], dtype)
  classes = tf.expand_dims(tf.cast(y_true['classes'], dtype=dtype), axis=-1)
  anchors = tf.cast(y_true['best_anchors'], dtype)
  ious = tf.cast(y_true['best_iou_match'], dtype)

  width = tf.cast(size, boxes.dtype)
  height = tf.cast(size, boxes.dtype)
  # get the number of boxes in the ground truth boxs
  num_boxes = tf.shape(boxes)[-2]
  # get the number of anchor boxes used for this anchor scale
  len_masks = tf.shape(mask)[0]
  # number of anchors
  num_anchors = tf.shape(anchors)[-1]
  num_instances = num_boxes * scale_num_inst

  pull_in = _gen_offsets(scale_xy, boxes.dtype)

  # rescale the x and y centers to the size of the grid [size, size]
  mask = tf.cast(mask, dtype=dtype)
  box_mask = _gen_utility(boxes)
  num_reps, viable_primary, viable_alternate = _get_num_reps(
      anchors, mask, box_mask)
  viable_primary = tf.cast(viable_primary, tf.int32)
  viable_alternate = tf.cast(viable_alternate, tf.int32)

  num_written = 0
  ind_val = tf.TensorArray(tf.int32, size=0, dynamic_size=True)
  ind_sample = tf.TensorArray(dtype, size=0, dynamic_size=True)

  (ind_val, ind_sample,
   num_written) = write_grid(viable_primary, num_reps, boxes, classes, ious,
                             ind_val, ind_sample, height, width, num_written,
                             num_instances, 0.0)

  if use_tie_breaker:
    # tf.print("alternate")
    (ind_val, ind_sample,
     num_written) = write_grid(viable_alternate, num_reps, boxes, classes, ious,
                               ind_val, ind_sample, height, width, num_written,
                               num_instances, 0.0)

  if pull_in > 0.0:
    (ind_val, ind_sample,
     num_written) = write_grid(viable_primary, num_reps, boxes, classes, ious,
                               ind_val, ind_sample, height, width, num_written,
                               num_instances, pull_in)

    if use_tie_breaker:
      # tf.print("alternate")
      (ind_val, ind_sample,
       num_written) = write_grid(viable_alternate, num_reps, boxes, classes,
                                 ious, ind_val, ind_sample, height, width,
                                 num_written, num_instances, pull_in)

  indexs = ind_val.stack()
  samples = ind_sample.stack()

  (true_box, ind_mask, true_class, best_iou_match, num_reps) = tf.split(
      samples, [4, 1, 1, 1, 1], axis=-1)

  full = tf.zeros([size, size, len_masks, 1], dtype=dtype)
  full = tf.tensor_scatter_nd_add(full, indexs, ind_mask)

  if num_written >= num_instances:
    tf.print("clipped")

  indexs = _pad_max_instances(indexs, num_instances, pad_value=0, pad_axis=0)
  samples = _pad_max_instances(samples, num_instances, pad_value=0, pad_axis=0)
  return indexs, samples, full


def write_sample(box, anchor_id, offset, sample, ind_val, ind_sample, height,
                 width, num_written):
  a_ = tf.convert_to_tensor([tf.cast(anchor_id, tf.int32)])

  y = box[1] * height
  x = box[0] * width

  # idk if this is right!!! just testing it now
  if offset > 0:
    y_ = tf.math.floor(y + offset)
    x_ = x
    if y_ >= 0 and y_ < height and y_ != tf.floor(y):
      y_ = tf.convert_to_tensor([tf.cast(y_, tf.int32)])
      x_ = tf.convert_to_tensor([tf.cast(x_, tf.int32)])
      grid_idx = tf.concat([y_, x_, a_], axis=-1)
      ind_val = ind_val.write(num_written, grid_idx)
      ind_sample = ind_sample.write(num_written, sample)
      num_written += 1

    y_ = tf.math.floor(y - offset)
    x_ = x
    if y_ >= 0 and y_ < height and y_ != tf.floor(y):
      y_ = tf.convert_to_tensor([tf.cast(y_, tf.int32)])
      x_ = tf.convert_to_tensor([tf.cast(x_, tf.int32)])
      grid_idx = tf.concat([y_, x_, a_], axis=-1)
      ind_val = ind_val.write(num_written, grid_idx)
      ind_sample = ind_sample.write(num_written, sample)
      num_written += 1

    y_ = y
    x_ = tf.math.floor(x + offset)
    if x_ >= 0 and x_ < height and x_ != tf.floor(x):
      y_ = tf.convert_to_tensor([tf.cast(y_, tf.int32)])
      x_ = tf.convert_to_tensor([tf.cast(x_, tf.int32)])
      grid_idx = tf.concat([y_, x_, a_], axis=-1)
      ind_val = ind_val.write(num_written, grid_idx)
      ind_sample = ind_sample.write(num_written, sample)
      num_written += 1

    y_ = y
    x_ = tf.math.floor(x - offset)
    if x_ >= 0 and x_ < height and x_ != tf.floor(x):
      y_ = tf.convert_to_tensor([tf.cast(y_, tf.int32)])
      x_ = tf.convert_to_tensor([tf.cast(x_, tf.int32)])
      grid_idx = tf.concat([y_, x_, a_], axis=-1)
      ind_val = ind_val.write(num_written, grid_idx)
      ind_sample = ind_sample.write(num_written, sample)
      num_written += 1
  else:
    y_ = tf.convert_to_tensor([tf.cast(y, tf.int32)])
    x_ = tf.convert_to_tensor([tf.cast(x, tf.int32)])
    grid_idx = tf.concat([y_, x_, a_], axis=-1)
    ind_val = ind_val.write(num_written, grid_idx)
    ind_sample = ind_sample.write(num_written, sample)
    num_written += 1
  return ind_val, ind_sample, num_written


def write_grid(viable, num_reps, boxes, classes, ious, ind_val, ind_sample,
               height, width, num_written, num_instances, offset):

  const = tf.cast(tf.convert_to_tensor([1.]), dtype=boxes.dtype)
  num_viable = tf.shape(viable)[0]
  for val in range(num_viable):
    idx = viable[val]
    obj_id, anchor, anchor_idx = idx[0], idx[1], idx[2]
    if num_written >= num_instances:
      break

    reps = tf.convert_to_tensor([num_reps[obj_id]])
    box = boxes[obj_id]
    cls_ = classes[obj_id]
    iou = tf.convert_to_tensor([ious[obj_id, anchor]])
    sample = tf.concat([box, const, cls_, iou, reps], axis=-1)

    ind_val, ind_sample, num_written = write_sample(box, anchor_idx, offset,
                                                    sample, ind_val, ind_sample,
                                                    height, width, num_written)
  return ind_val, ind_sample, num_written


def random_crop_image(image,
                      aspect_ratio_range=(3. / 4., 4. / 3.),
                      area_range=(0.08, 1.0),
                      max_attempts=10,
                      seed=1):
  """Randomly crop an arbitrary shaped slice from the input image.
  Args:
    image: a Tensor of shape [height, width, 3] representing the input image.
    aspect_ratio_range: a list of floats. The cropped area of the image must
      have an aspect ratio = width / height within this range.
    area_range: a list of floats. The cropped reas of the image must contain
      a fraction of the input image within this range.
    max_attempts: the number of attempts at generating a cropped region of the
      image of the specified constraints. After max_attempts failures, return
      the entire image.
    seed: the seed of the random generator.
  Returns:
    cropped_image: a Tensor representing the random cropped image. Can be the
      original image if max_attempts is exhausted.
  """

  with tf.name_scope('random_crop_image'):
    ishape = tf.shape(image)
    crop_offset, crop_size, _ = tf.image.sample_distorted_bounding_box(
        ishape,
        tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4]),
        seed=seed,
        min_object_covered=area_range[0],
        aspect_ratio_range=aspect_ratio_range,
        area_range=area_range,
        max_attempts=max_attempts)

    cropped_image = tf.slice(image, crop_offset, crop_size)

    scale = tf.cast(ishape[:2] / ishape[:2], tf.float32)
    offset = tf.cast(crop_offset[:2], tf.float32)

    info = tf.stack([
        tf.cast(ishape[:2], tf.float32),
        tf.cast(crop_size[:2], tf.float32), scale, offset
    ],
                    axis=0)
    return cropped_image, info


def random_crop_mosaic(image,
                       aspect_ratio_range=(3. / 4., 4. / 3.),
                       area_range=(0.08, 1.0),
                       max_attempts=10,
                       seed=1):
  """Randomly crop an arbitrary shaped slice from the input image.
  Args:
    image: a Tensor of shape [height, width, 3] representing the input image.
    aspect_ratio_range: a list of floats. The cropped area of the image must
      have an aspect ratio = width / height within this range.
    area_range: a list of floats. The cropped reas of the image must contain
      a fraction of the input image within this range.
    max_attempts: the number of attempts at generating a cropped region of the
      image of the specified constraints. After max_attempts failures, return
      the entire image.
    seed: the seed of the random generator.
  Returns:
    cropped_image: a Tensor representing the random cropped image. Can be the
      original image if max_attempts is exhausted.
  """

  with tf.name_scope('random_crop_image'):
    ishape = tf.shape(image)
    crop_offset, crop_size, _ = tf.image.sample_distorted_bounding_box(
        ishape,
        tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4]),
        seed=seed,
        min_object_covered=area_range[0],
        aspect_ratio_range=aspect_ratio_range,
        area_range=area_range,
        max_attempts=max_attempts)

    area = -tf.reduce_prod(crop_size) / tf.reduce_prod(ishape[:2])

    delta = ishape[:2] - crop_size[:2]
    dx = rand_uniform_strong(0, 1 + 1, tf.int32)
    dy = rand_uniform_strong(0, 1 + 1, tf.int32)
    ds = rand_uniform_strong(1, 3, tf.int32)

    oh = tf.maximum(dy * delta[0] - 1, 0) // ds
    ow = tf.maximum(dx * delta[1] - 1, 0) // ds
    crop_offset = tf.convert_to_tensor([oh, ow, 0])
    cropped_image = tf.slice(image, crop_offset, crop_size)

    scale = tf.cast(ishape[:2] / ishape[:2], tf.float32)
    offset = tf.cast(crop_offset[:2], tf.float32)

    info = tf.stack([
        tf.cast(ishape[:2], tf.float32),
        tf.cast(crop_size[:2], tf.float32), scale, offset
    ],
                    axis=0)
    return cropped_image, info


def random_pad(image, area):
  with tf.name_scope('pad_to_bbox'):
    height, width = get_image_shape(image)

    rand_area = tf.cast(area, tf.float32)  #tf.random.uniform([], 1.0, area)
    target_height = tf.cast(
        tf.sqrt(rand_area) * tf.cast(height, rand_area.dtype), tf.int32)
    target_width = tf.cast(
        tf.sqrt(rand_area) * tf.cast(width, rand_area.dtype), tf.int32)

    offset_height = tf.random.uniform([],
                                      0,
                                      target_height - height + 1,
                                      dtype=tf.int32)
    offset_width = tf.random.uniform([],
                                     0,
                                     target_width - width + 1,
                                     dtype=tf.int32)

    image = tf.image.pad_to_bounding_box(image, offset_height, offset_width,
                                         target_height, target_width)

  ishape = tf.convert_to_tensor([height, width])
  oshape = tf.convert_to_tensor([target_height, target_width])
  offset = tf.convert_to_tensor([offset_height, offset_width])

  scale = tf.cast(ishape[:2] / ishape[:2], tf.float32)
  offset = tf.cast(-offset, tf.float32)  # * scale

  info = tf.stack([
      tf.cast(ishape[:2], tf.float32),
      tf.cast(oshape[:2], tf.float32), scale, offset
  ],
                  axis=0)

  return image, info


def random_rotate_image(image, max_angle):
  angle = tf.random.uniform([],
                            minval=-max_angle,
                            maxval=max_angle,
                            dtype=tf.float32)
  deg = angle * 3.14 / 360.
  deg.set_shape(())
  image = tfa.image.rotate(image, deg, interpolation='BILINEAR')
  return image, deg


def get_corners(boxes):
  ymi, xmi, yma, xma = tf.split(boxes, 4, axis=-1)

  tl = tf.concat([xmi, ymi], axis=-1)
  bl = tf.concat([xmi, yma], axis=-1)
  tr = tf.concat([xma, ymi], axis=-1)
  br = tf.concat([xma, yma], axis=-1)

  corners = tf.concat([tl, bl, tr, br], axis=-1)
  return corners


def rotate_points(x_, y_, angle):
  sx = 0.5 - x_
  sy = 0.5 - y_

  r = tf.sqrt(sx**2 + sy**2)
  curr_theta = tf.atan2(sy, sx)

  cos = tf.math.cos(curr_theta - angle)
  sin = tf.math.sin(curr_theta - angle)

  x = r * cos
  y = r * sin

  x = -x + 0.5
  y = -y + 0.5
  return x, y


def rotate_boxes(boxes, height, width, angle):
  corners = get_corners(boxes)
  # boxes = box_ops.yxyx_to_xcycwh(boxes)
  ymin, xmin, ymax, xmax = tf.split(boxes, 4, axis=-1)

  tlx, tly = rotate_points(xmin, ymin, angle)
  blx, bly = rotate_points(xmin, ymax, angle)
  trx, try_ = rotate_points(xmax, ymin, angle)
  brx, bry = rotate_points(xmax, ymax, angle)

  xs = tf.concat([tlx, blx, trx, brx], axis=-1)
  ys = tf.concat([tly, bly, try_, bry], axis=-1)

  xmin = tf.reduce_min(xs, axis=-1, keepdims=True)
  ymin = tf.reduce_min(ys, axis=-1, keepdims=True)
  xmax = tf.reduce_max(xs, axis=-1, keepdims=True)
  ymax = tf.reduce_max(ys, axis=-1, keepdims=True)

  boxes = tf.concat([ymin, xmin, ymax, xmax], axis=-1)
  return boxes


def resize_and_crop_image(image,
                          desired_size,
                          padded_size,
                          aug_scale_min=1.0,
                          aug_scale_max=1.0,
                          seed=1,
                          method=tf.image.ResizeMethod.BILINEAR):
  """Resizes the input image to output size (RetinaNet style).
  Resize and pad images given the desired output size of the image and
  stride size.
  Here are the preprocessing steps.
  1. For a given image, keep its aspect ratio and rescale the image to make it
     the largest rectangle to be bounded by the rectangle specified by the
     `desired_size`.
  2. Pad the rescaled image to the padded_size.
  Args:
    image: a `Tensor` of shape [height, width, 3] representing an image.
    desired_size: a `Tensor` or `int` list/tuple of two elements representing
      [height, width] of the desired actual output image size.
    padded_size: a `Tensor` or `int` list/tuple of two elements representing
      [height, width] of the padded output image size. Padding will be applied
      after scaling the image to the desired_size.
    aug_scale_min: a `float` with range between [0, 1.0] representing minimum
      random scale applied to desired_size for training scale jittering.
    aug_scale_max: a `float` with range between [1.0, inf] representing maximum
      random scale applied to desired_size for training scale jittering.
    seed: seed for random scale jittering.
    method: function to resize input image to scaled image.
  Returns:
    output_image: `Tensor` of shape [height, width, 3] where [height, width]
      equals to `output_size`.
    image_info: a 2D `Tensor` that encodes the information of the image and the
      applied preprocessing. It is in the format of
      [[original_height, original_width], [desired_height, desired_width],
       [y_scale, x_scale], [y_offset, x_offset]], where [desired_height,
      desired_width] is the actual scaled image size, and [y_scale, x_scale] is
      the scaling factor, which is the ratio of
      scaled dimension / original dimension.
  """
  with tf.name_scope('resize_and_crop_image'):
    image_size = tf.cast(tf.shape(image)[0:2], tf.float32)

    random_jittering = (aug_scale_min != 1.0 or aug_scale_max != 1.0)

    if random_jittering:
      random_scale = tf.random.uniform([],
                                       aug_scale_min,
                                       aug_scale_max,
                                       seed=seed)
      scaled_size = tf.round(random_scale * desired_size)
    else:
      scaled_size = desired_size

    scale = tf.minimum(scaled_size[0] / image_size[0],
                       scaled_size[1] / image_size[1])
    scaled_size = tf.round(image_size * scale)

    # Computes 2D image_scale.
    image_scale = scaled_size / image_size

    # Selects non-zero random offset (x, y) if scaled image is larger than
    # desired_size.
    if random_jittering:
      max_offset_ = scaled_size - desired_size

      max_offset = tf.where(
          tf.less(max_offset_, 0), tf.zeros_like(max_offset_), max_offset_)
      offset = max_offset * tf.random.uniform([
          2,
      ], 0, 1, seed=seed)
      offset = tf.cast(offset, tf.int32)
    else:
      offset = tf.zeros((2,), tf.int32)

    scaled_image = tf.image.resize(
        image, tf.cast(scaled_size, tf.int32), method=method)

    if random_jittering:
      scaled_image = scaled_image[offset[0]:offset[0] + desired_size[0],
                                  offset[1]:offset[1] + desired_size[1], :]

    scaled_size = tf.cast(tf.shape(scaled_image)[0:2], tf.int32)

    if random_jittering:
      dy = rand_uniform_strong(0, padded_size[0] - scaled_size[0] + 1, tf.int32)
      dx = rand_uniform_strong(0, padded_size[1] - scaled_size[1] + 1, tf.int32)
    else:
      dy = 0
      dx = 0
    output_image = tf.image.pad_to_bounding_box(scaled_image, dy, dx,
                                                padded_size[0], padded_size[1])

    offset -= tf.convert_to_tensor([dy, dx])

    image_info = tf.stack([
        image_size,
        tf.constant(desired_size, dtype=tf.float32), image_scale,
        tf.cast(offset, tf.float32)
    ])
    return output_image, image_info
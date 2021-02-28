import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras.backend as K
from yolo.ops import box_ops
from official.vision.beta.ops import preprocess_ops

def rand_uniform_strong(minval, maxval, dtype=tf.float32):
  if minval > maxval:
    minval, maxval = maxval, minval
  return tf.random.uniform([], minval=minval, maxval=maxval, dtype=dtype)

def rand_scale(val, dtype=tf.float32):
  scale = rand_uniform_strong(1, val, dtype=dtype)
  do_ret = tf.random.uniform([], minval=0, maxval=2, dtype=tf.int32)
  if (do_ret == 1):
    return scale
  return 1.0 / scale

def _boolean_mask(data, mask):
  data = data * tf.cast(mask, data.dtype)
  if tf.shape(tf.shape(data)) == 2: 
    data = tf.reshape(data, [-1])
  else:
    data = tf.reshape(data, [tf.shape(data)[0], -1])
  return data


def shift_zeros(data, mask, axis=-2, fill=0):
  zeros = tf.zeros_like(data) + fill
  data_flat = tf.boolean_mask(data, mask)
  
  # tf.print(tf.shape(data_flat), tf.shape(data))
  nonzero_lens = tf.reduce_sum(tf.cast(mask, dtype=tf.int32), axis=-2)
  nonzero_mask = tf.sequence_mask(nonzero_lens, maxlen=tf.shape(mask)[-2])
  perm1 = tf.range(0, tf.shape(tf.shape(data))[0] - 2)
  perm2 = tf.roll(
      tf.range(tf.shape(tf.shape(data))[0] - 2,
               tf.shape(tf.shape(data))[0]),
      1,
      axis=-1)

  perm = tf.concat([perm1, perm2], axis=-1)
  nonzero_mask = tf.transpose(nonzero_mask, perm=perm)
  inds = tf.cast(tf.where(nonzero_mask), dtype=tf.int32)
  nonzero_data = tf.tensor_scatter_nd_update(
      zeros, tf.cast(tf.where(nonzero_mask), dtype=tf.int32), data_flat)
  return nonzero_data

def shift_zeros2(mask, squeeze = True, fill = 0):
  mask = tf.cast(mask, tf.float32)
  if squeeze: 
    mask = tf.squeeze(mask, axis = -1)

  k = tf.shape(mask)[-1]
  mask, ind = tf.math.top_k(mask, k=k, sorted = True)
  return mask, ind


def _pad_max_instances(value, instances, pad_value=0, pad_axis=0):
  shape = tf.shape(value)
  if pad_axis < 0:
    pad_axis = tf.shape(shape)[0] + pad_axis
  dim1 = shape[pad_axis]
  take = tf.math.reduce_min([instances, dim1])
  value, _ = tf.split(
      value, [take, -1], axis=pad_axis)  # value[:instances, ...]
  pad = tf.convert_to_tensor([tf.math.reduce_max([instances - dim1, 0])])
  nshape = tf.concat([shape[:pad_axis], pad, shape[(pad_axis + 1):]], axis=0)
  pad_tensor = tf.fill(nshape, tf.cast(pad_value, dtype=value.dtype))
  value = tf.concat([value, pad_tensor], axis=pad_axis)
  return value

def _shift_zeros_full(boxes, classes, num_instances, yxyx = True):
  if yxyx:
    boxes = box_ops.yxyx_to_xcycwh(boxes)
  x, y, w, h = tf.split(boxes, 4, axis=-1)
  mask = w > 0
  # tf.print(tf.shape(x), tf.shape(mask))
  mask, ind = shift_zeros2(mask)
  ind_m = tf.ones_like(ind) * tf.expand_dims(
      tf.range(0,
               tf.shape(ind)[0]), axis=-1)
  ind = tf.stack([tf.reshape(ind_m, [-1]), tf.reshape(ind, [-1])], axis=-1)
  

  classes_shape = tf.shape(classes)
  classes_ = tf.gather_nd(classes, ind)
  classes = (tf.reshape(classes_, classes_shape) * tf.cast(mask, classes.dtype)) - (1 - tf.cast(mask, x.dtype))

  mask = tf.expand_dims(tf.cast(mask, x.dtype), axis = -1)
  x_shape = tf.shape(x)
  x_ = tf.gather_nd(tf.squeeze(x, axis = -1), ind)
  x = tf.reshape(x_, x_shape) * mask

  y_shape = tf.shape(y)
  y_ = tf.gather_nd(tf.squeeze(y, axis = -1), ind)
  y = tf.reshape(y_, y_shape) * mask

  w_shape = tf.shape(w)
  w_ = tf.gather_nd(tf.squeeze(w, axis = -1), ind)
  w = tf.reshape(w_, w_shape) * mask

  h_shape = tf.shape(h)
  h_ = tf.gather_nd(tf.squeeze(h, axis = -1), ind)
  h = tf.reshape(h_, h_shape) * mask

  boxes = tf.cast(tf.concat([x, y, w, h], axis=-1), boxes.dtype)
  boxes = _pad_max_instances(boxes, num_instances, pad_axis=-2, pad_value=0)
  classes = _pad_max_instances(classes, num_instances, pad_axis=-1, pad_value=-1)
  if yxyx:
    boxes = box_ops.xcycwh_to_yxyx(boxes)
  return boxes, classes 

def near_edge_adjustment(boxes, y_lower_bound, x_lower_bound, y_upper_bound, x_upper_bound, keep_thresh = 0.25):
  x_lower_bound = tf.clip_by_value(x_lower_bound, 0.0, 1.0 - K.epsilon())
  y_lower_bound = tf.clip_by_value(y_lower_bound, 0.0, 1.0 - K.epsilon())
  x_upper_bound = tf.clip_by_value(x_upper_bound, 0.0, 1.0 - K.epsilon())
  y_upper_bound = tf.clip_by_value(y_upper_bound, 0.0, 1.0 - K.epsilon())

  x_lower_bound = tf.cast(x_lower_bound, boxes.dtype)
  y_lower_bound = tf.cast(y_lower_bound, boxes.dtype) 
  x_upper_bound = tf.cast(x_upper_bound, boxes.dtype)
  y_upper_bound = tf.cast(y_upper_bound, boxes.dtype)
  keep_thresh = tf.cast(keep_thresh, boxes.dtype)

  y_min, x_min, y_max, x_max = tf.split(tf.cast(boxes, x_lower_bound.dtype), 4, axis=-1)
  boxes = box_ops.yxyx_to_xcycwh(boxes)
  x, y, w, h = tf.split(tf.cast(boxes, x_lower_bound.dtype), 4, axis=-1)

  # locations where atleast 25% of the image is in frame but the certer is not 
  y_mask1 = tf.math.logical_and((y_upper_bound-y_min > tf.cast(h * keep_thresh, y_min.dtype)), (y > y_upper_bound))
  x_mask1 = tf.math.logical_and((x_upper_bound-x_min > tf.cast(w * keep_thresh, x_min.dtype)), (x > x_upper_bound))

  # tf.print(tf.reduce_sum(tf.cast(x_mask1, tf.int32)))
  # tf.print(tf.reduce_sum(tf.cast(y_mask1, tf.int32)))

  #y_new = tf.where(y_mask1, y_upper_bound - 1, y)
  y_new = tf.where(y_mask1, y_upper_bound, y)
  h_new = tf.where(y_mask1, (y_new - y_min) * 2, h)
  # h_new = tf.where(y_mask1, (y_max - y_new) * 2, h)
  # x_new = tf.where(x_mask1, x_upper_bound - 1, x)
  x_new = tf.where(x_mask1, x_upper_bound, x)
  w_new = tf.where(x_mask1, (x_new - x_min) * 2, w)
  #w_new = tf.where(x_mask1, (x_max - x_new) * 2, w)
  
  boxes = tf.cast(tf.concat([x_new, y_new, w_new, h_new], axis=-1), boxes.dtype)
  x, y, w, h = tf.split(tf.cast(boxes, x_lower_bound.dtype), 4, axis=-1)
  boxes = box_ops.xcycwh_to_yxyx(boxes)
  y_min, x_min, y_max, x_max = tf.split(tf.cast(boxes, x_lower_bound.dtype), 4, axis=-1)

  y_mask1 = tf.math.logical_and((y_max-y_lower_bound > tf.cast(h * keep_thresh, y_max.dtype)), (y < y_lower_bound))
  x_mask1 = tf.math.logical_and((x_max-x_lower_bound > tf.cast(w * keep_thresh, x_max.dtype)), (x < x_lower_bound))

  # tf.print(tf.reduce_sum(tf.cast(x_mask1, tf.int32)))
  # tf.print(tf.reduce_sum(tf.cast(y_mask1, tf.int32)))

  # y_new = tf.where(y_mask1, y_lower_bound + 1, y)
  y_new = tf.where(y_mask1, y_lower_bound, y)
  h_new = tf.where(y_mask1, (y_max - y_new) * 2, h)
  # h_new = tf.where(y_mask1, (y_new - y_min) * 2, h)
  # x_new = tf.where(x_mask1, x_lower_bound + 1, x)
  x_new = tf.where(x_mask1, x_lower_bound, x)
  w_new = tf.where(x_mask1, (x_max - x_new) * 2, w)
  # w_new = tf.where(x_mask1, (x_new - x_min) * 2, w)
  
  boxes = tf.cast(tf.concat([x_new, y_new, w_new, h_new], axis=-1), boxes.dtype)
  boxes = box_ops.xcycwh_to_yxyx(boxes)
  return boxes

def get_image_shape(image):
  shape = tf.shape(image)
  if tf.shape(shape)[0] == 4:
    width = shape[2]
    height = shape[1]
  else:
    width = shape[1]
    height = shape[0]
  return height, width 



# do all the ops needed
def random_op_image(image, jfactor, zfactor, tfactor, letter_box = True):
  image, jitter_info = random_jitter(image, jfactor)
  image, translate_info = random_translate(image, tfactor)
  image, crop_info = random_zoom(image, zfactor)
  crop_info.update(translate_info)
  return image, crop_info

def random_translate(image, t):
  with tf.name_scope('translate_image'):
    if t != 0:
      t_x = tf.random.uniform(minval=-t, maxval=t, shape=(), dtype=tf.float32)
      t_y = tf.random.uniform(minval=-t, maxval=t, shape=(), dtype=tf.float32)
      height, width = get_image_shape(image)
      image_jitter = tf.convert_to_tensor([t_x , t_y])
      image_dims = tf.cast(tf.convert_to_tensor([width, height]), tf.float32)
      image_jitter.set_shape([2])
      image = tfa.image.translate(
          image,  image_jitter * image_dims)
    else:
      t_x = 0.0
      t_y = 0.0
  return image, {'translate_offset': [t_y, t_x]}

def translate_boxes(box, classes, translate_x, translate_y):
  with tf.name_scope('translate_boxs'):
    box = box_ops.yxyx_to_xcycwh(box)
    x, y, w, h = tf.split(box, 4, axis=-1)
    x = x + translate_x
    y = y + translate_y
    box = tf.cast(tf.concat([x, y, w, h], axis=-1), box.dtype)
    box = box_ops.xcycwh_to_yxyx(box)
  return box, classes

def random_zoom(image, zfactor):
  zx = 1 + tf.random.uniform(
      minval=-zfactor, maxval=zfactor, shape=(), dtype=tf.float32)
  zy = zx 
  height, width = get_image_shape(image)  
  width = tf.cast(tf.cast(width, zx.dtype) * zx, width.dtype)
  height = tf.cast(tf.cast(height, zy.dtype) * zy, height.dtype)
  return random_crop_or_pad(image, width, height, random_patch = True)

def random_jitter(image, jfactor):
  jx = 1 + tf.random.uniform(minval= -2 * jfactor, maxval=2 * jfactor, shape=(), dtype=tf.float32)
  jy = 1 + tf.random.uniform(minval= -2 * jfactor, maxval=2 * jfactor, shape=(), dtype=tf.float32)

  height, width = get_image_shape(image)
  width = tf.cast(width, jx.dtype) * jx
  height = tf.cast(height, jy.dtype) * jy
  image = tf.image.resize(image, (height, width))
  jitter_info = {'jitter_dims': [jy, jx]}
  return image, jitter_info

def random_crop_or_pad(image,
                      target_width,
                      target_height,
                      random_patch=False):
  with tf.name_scope('resize_crop_filter'):
    default_height, default_width = get_image_shape(image)
    dx = (tf.math.maximum(default_width, target_width) -
          tf.math.minimum(default_width, target_width)) // 2
    dy = (tf.math.maximum(default_height, target_height) -
          tf.math.minimum(default_height, target_height)) // 2

    if random_patch:
      dx = tf.random.uniform([], minval=0, maxval=dx *
                             2 + 1, dtype=tf.int32) if dx != 0 else 0
      dy = tf.random.uniform([], minval=0, maxval=dy *
                             2 + 1, dtype=tf.int32) if dy != 0 else 0

    if target_width > default_width:
      image, _ = pad_to_bbox(image, target_width, default_height, dx, 0)
      dx = -dx
    elif target_width < default_width:
      image, _ = crop_to_bbox(image, target_width, default_height, dx, 0, fix=False)    

    if target_height > default_height:
      image, _ = pad_to_bbox(image, target_width, target_height, 0, dy)
      dy = -dy
    elif target_height < default_height:
      image, _ = crop_to_bbox(image, target_width, target_height, 0, dy, fix=False)      

  crop_info = {'original_dims': [default_height, default_width], 
               'new_dims':[target_height, target_width], 
               'offset': [dy, dx], 
               'fixed': False}
  return image, crop_info

def crop_to_bbox(image,
                target_width,
                target_height,
                offset_width,
                offset_height,
                fix=False):
  with tf.name_scope('resize_crop_filter'):
    height, width = get_image_shape(image)
    image = tf.image.crop_to_bounding_box(image, offset_height, offset_width,
                                          target_height, target_width)
    if fix:
      image = tf.image.pad_to_bounding_box(image, offset_height, offset_width,
                                           height, width)

  crop_info = {'original_dims': [height, width], 
               'new_dims':[target_height, target_width], 
               'offset': [offset_height, offset_width], 
               'fixed': fix}
  return image, crop_info


def pad_to_bbox(image, target_width, target_height,
                       offset_width, offset_height):
  with tf.name_scope('resize_crop_filter'):
    height, width = get_image_shape(image)
    image = tf.image.pad_to_bounding_box(image, offset_height, offset_width,
                                         target_height, target_width)
  crop_info = {'original_dims': [height, width], 
               'new_dims':[target_height, target_width], 
               'offset': [offset_height, offset_width], 
               'fixed': False}
  return image, crop_info


def filter_boxes_and_classes(boxes, classes, image_info, keep_thresh = 0.01):
  height = image_info['original_dims'][0]
  target_height = image_info['new_dims'][0]
  offset_height = image_info['offset'][0]
  

  width = image_info['original_dims'][1]
  target_width = image_info['new_dims'][1]
  offset_width = image_info['offset'][1]

  fixed = image_info['fixed']

  xscaler = width / target_width
  yscaler = height / target_height

  if 'translate_offset' in image_info.keys():
    theight = image_info['translate_offset'][0]
    twidth = image_info['translate_offset'][1]
    boxes, classes = translate_boxes(boxes, classes, twidth, theight)

  # tf.print(xscaler, yscaler, image_info)
   
  x_lower_bound = offset_width / width
  x_upper_bound = (offset_width + target_width) / width if xscaler > 1.0 else tf.cast(1.0, tf.float64)

  y_lower_bound = offset_height / height
  y_upper_bound = (offset_height + target_height) / height if yscaler > 1.0 else tf.cast(1.0, tf.float64)

  boxes = near_edge_adjustment(boxes, y_lower_bound, x_lower_bound, y_upper_bound, x_upper_bound, keep_thresh = keep_thresh)
  boxes = box_ops.yxyx_to_xcycwh(boxes)
  x, y, w, h = tf.split(tf.cast(boxes, x_lower_bound.dtype), 4, axis=-1)

  x_mask_lower = tf.math.logical_and(x > x_lower_bound, x >= 0)
  y_mask_lower = tf.math.logical_and(y > y_lower_bound, y >= 0)
  x_mask_upper = tf.math.logical_and(x < x_upper_bound, x < 1)
  y_mask_upper = tf.math.logical_and(y < y_upper_bound, y < 1) 

  x_mask = tf.math.logical_and(x_mask_lower, x_mask_upper)
  y_mask = tf.math.logical_and(y_mask_lower, y_mask_upper)
  mask = tf.math.logical_and(x_mask, y_mask)

  x = shift_zeros(x, mask)  # tf.boolean_mask(x, mask)
  y = shift_zeros(y, mask)  # tf.boolean_mask(y, mask)
  w = shift_zeros(w, mask)  # tf.boolean_mask(w, mask)
  h = shift_zeros(h, mask)  # tf.boolean_mask(h, mask)
  classes = shift_zeros(tf.expand_dims(classes, axis=-1), mask, fill=-1)
  classes = tf.squeeze(classes, axis=-1)

  if not fixed:
    x = (x - x_lower_bound) * xscaler
    y = (y - y_lower_bound) * yscaler
    w = w * xscaler
    h = h * yscaler

  boxes = tf.cast(tf.concat([x, y, w, h], axis=-1), boxes.dtype)
  boxes = box_ops.xcycwh_to_yxyx(boxes)
  return boxes, classes

def letter_box(image,
               boxes,
               target_dim=None):
  height, width = get_image_shape(image)
  clipper = tf.math.maximum(width, height)
  if target_dim is None:
    target_dim = clipper

  pad_width = clipper - width
  pad_height = clipper - height
  image = tf.image.pad_to_bounding_box(image, pad_height // 2, pad_width // 2,
                                       clipper, clipper)

  boxes = box_ops.yxyx_to_xcycwh(boxes)
  x, y, w, h = tf.split(boxes, 4, axis=-1)

  y *= tf.cast(height / clipper, tf.float32)
  x *= tf.cast(width / clipper, tf.float32)

  y += tf.cast((pad_height / clipper) / 2, tf.float32)
  x += tf.cast((pad_width / clipper) / 2, tf.float32)

  h *= tf.cast(height / clipper, tf.float32)
  w *= tf.cast(width / clipper, tf.float32)

  boxes = tf.concat([x, y, w, h], axis=-1)

  boxes = box_ops.xcycwh_to_yxyx(boxes)
  image = tf.image.resize(image, (target_dim, target_dim))
  return image, boxes

def patch_four_images(images):
  image1, image2, image3, image4 = tf.split(images, 4, axis=0)
  patch1 = tf.concat([image1, image2], axis=-2)
  patch2 = tf.concat([image3, image4], axis=-2)
  full_image = tf.concat([patch1, patch2], axis=-3)
  return full_image

def mosaic(images, boxes, classes, output_size, masks = None, crop_delta=0.6):
  full_image = patch_four_images(images)
  height, width = get_image_shape(full_image)
  num_instances = tf.shape(boxes)[-2]

  box1, box2, box3, box4 = tf.split(boxes * 0.5, 4, axis=0)
  class1, class2, class3, class4 = tf.split(classes, 4, axis=0)
  #translate boxes
  box2, class2 = translate_boxes(box2, class2, .5, 0)
  box3, class3 = translate_boxes(box3, class3, 0, .5)
  box4, class4 = translate_boxes(box4, class4, .5, .5)
  full_boxes = tf.concat([box1, box2, box3, box4], axis=-2)
  full_classes = tf.concat([class1, class2, class3, class4], axis=-1)
  full_boxes, full_classes = _shift_zeros_full(full_boxes,
                                             full_classes,
                                             num_instances,
                                             yxyx=True)

  crop_delta = rand_uniform_strong(crop_delta, 1.0) #1 + crop_delta * 3/5)
  full_image, image_info = random_crop_or_pad(full_image,
                                                  target_width=tf.cast(
          tf.cast(height, tf.float32) * crop_delta, tf.int32),
                                                  target_height=tf.cast(
          tf.cast(width, tf.float32) * crop_delta, tf.int32),
                                                  random_patch=True)
  full_boxes, full_classes = filter_boxes_and_classes(full_boxes, full_classes, image_info, keep_thresh = 0.1)
  full_image = tf.image.resize(full_image, [output_size, output_size])
  return tf.cast(full_image, images.dtype), tf.cast(full_boxes, boxes.dtype), tf.cast(full_classes, classes.dtype)























# ops for building batches in GT format
def get_best_anchor(y_true, anchors, width=1, height=1):
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
  with tf.name_scope('get_anchor'):
    width = tf.cast(width, dtype=y_true.dtype)
    height = tf.cast(height, dtype=y_true.dtype)
    # split the boxes into center and width height
    anchor_xy = y_true[..., 0:2]
    true_wh = y_true[..., 2:4]

    # scale thhe boxes
    anchors = tf.convert_to_tensor(anchors, dtype=y_true.dtype)
    anchors_x = anchors[..., 0] / width
    anchors_y = anchors[..., 1] / height
    anchors = tf.stack([anchors_x, anchors_y], axis=-1)
    k = tf.shape(anchors)[0]
    # build a matrix of anchor boxes of shape [num_anchors, num_boxes, 4]
    anchors = tf.transpose(anchors, perm=[1, 0])

    anchor_xy = tf.tile(
        tf.expand_dims(anchor_xy, axis=-1), [1, 1, tf.shape(anchors)[-1]])
    anchors = tf.tile(
        tf.expand_dims(anchors, axis=0), [tf.shape(anchor_xy)[0], 1, 1])

    # stack the xy so, each anchor is asscoaited once with each center from
    # the ground truth input
    anchors = K.concatenate([anchor_xy, anchors], axis=1)
    anchors = tf.transpose(anchors, perm=[2, 0, 1])

    # copy the gt n times so that each anchor from above can be compared to
    # input ground truth to shape: [num_anchors, num_boxes, 4]
    truth_comp = tf.tile(
        tf.expand_dims(y_true[..., 0:4], axis=-1),
        [1, 1, tf.shape(anchors)[0]])
    truth_comp = tf.transpose(truth_comp, perm=[2, 0, 1])

    # compute intersection over union of the boxes, and take the argmax of
    # comuted iou for each box. thus each box is associated with the
    # largest interection over union
    iou_raw = box_ops.compute_iou(truth_comp, anchors)
    values, indexes = tf.math.top_k(
        tf.transpose(iou_raw, perm=[1, 0]),
        k=tf.cast(k, dtype=tf.int32),
        sorted=True)
    ind_mask = tf.cast(values > 0.213, dtype=indexes.dtype)

    # pad the indexs such that all values less than the thresh are -1
    # add one, multiply the mask to zeros all the bad locations
    # subtract 1 makeing all the bad locations 0.
    iou_index = tf.concat([
        K.expand_dims(indexes[..., 0], axis=-1),
        ((indexes[..., 1:] + 1) * ind_mask[..., 1:]) - 1
    ],
                          axis=-1)
    iou_index = iou_index[..., :6]

  return tf.cast(iou_index, dtype=tf.float32)


def get_best_anchor_batch(y_true, anchors, width=1, height=1):
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
  with tf.name_scope('get_anchor'):
    width = tf.cast(width, dtype=y_true.dtype)
    height = tf.cast(height, dtype=y_true.dtype)
    # split the boxes into center and width height
    anchor_xy = y_true[..., 0:2]
    true_wh = y_true[..., 2:4]

    # scale thhe boxes
    anchors = tf.convert_to_tensor(anchors, dtype=y_true.dtype)
    anchors_x = anchors[..., 0] / width
    anchors_y = anchors[..., 1] / height
    anchors = tf.stack([anchors_x, anchors_y], axis=-1)
    k = tf.shape(anchors)[0]
    # build a matrix of anchor boxes of shape [num_anchors, num_boxes, 4]
    anchors = tf.transpose(anchors, perm=[1, 0])

    anchor_xy = tf.repeat(
        tf.expand_dims(anchor_xy, axis=-1), tf.shape(anchors)[-1], axis=-1)
    anchors = tf.repeat(
        tf.expand_dims(anchors, axis=0), tf.shape(anchor_xy)[1], axis=0)
    anchors = tf.repeat(
        tf.expand_dims(anchors, axis=0), tf.shape(anchor_xy)[0], axis=0)

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
    iou_raw = box_ops.compute_iou(truth_comp, anchors)
    values, indexes = tf.math.top_k(
        tf.transpose(iou_raw, perm=[0, 2, 1]),
        k=tf.cast(k, dtype=tf.int32),
        sorted=True)
    ind_mask = tf.cast(values > 0.213, dtype=indexes.dtype)

    # pad the indexs such that all values less than the thresh are -1
    # add one, multiply the mask to zeros all the bad locations
    # subtract 1 makeing all the bad locations 0.
    iou_index = tf.concat([
        K.expand_dims(indexes[..., 0], axis=-1),
        ((indexes[..., 1:] + 1) * ind_mask[..., 1:]) - 1
    ],
                          axis=-1)
    iou_index = iou_index[..., :6]

  return tf.cast(iou_index, dtype=tf.float32)


def build_grided_gt(y_true, mask, size, num_classes, dtype, use_tie_breaker):
  """
    convert ground truth for use in loss functions
    Args:
      y_true: tf.Tensor[] ground truth
        [box coords[0:4], classes_onehot[0:-1], best_fit_anchor_box]
      mask: list of the anchor boxes choresponding to the output,
        ex. [1, 2, 3] tells this layer to predict only the first 3
        anchors in the total.
      size: the dimensions of this output, for regular, it progresses
        from 13, to 26, to 52
      num_classes: `integer` for the number of classes
      dtype: expected output datatype
      use_tie_breaker: boolean value for wether or not to use
        the tie_breaker

    Return:
      tf.Tensor[] of shape [size, size, #of_anchors, 4, 1, num_classes]
  """
  # unpack required components from the input ground truth
  boxes = tf.cast(y_true['bbox'], dtype)
  classes = tf.expand_dims(tf.cast(y_true['classes'], dtype=dtype), axis=-1)
  anchors = tf.cast(y_true['best_anchors'], dtype)

  # get the number of boxes in the ground truth boxs
  num_boxes = tf.shape(boxes)[0]
  # get the number of anchor boxes used for this anchor scale
  len_masks = tf.shape(mask)[0]

  # init a fixed memeory size grid for this prediction scale
  # [size, size, # of anchors, 1 + 1 + number of anchors per scale]
  full = tf.zeros([size, size, len_masks, 6], dtype=dtype)
  # init a grid to use to track which locations have already
  # been used before (for the tie breaker)
  depth_track = tf.zeros((size, size, len_masks), dtype=tf.int32)

  # rescale the x and y centers to the size of the grid [size, size]
  x = tf.cast(boxes[..., 0] * tf.cast(size, dtype=dtype), dtype=tf.int32)
  y = tf.cast(boxes[..., 1] * tf.cast(size, dtype=dtype), dtype=tf.int32)

  # init all the tensorArrays to be used in storeing the index
  # and the values to be used to update both depth_track and full
  update_index = tf.TensorArray(tf.int32, size=0, dynamic_size=True)
  update = tf.TensorArray(dtype, size=0, dynamic_size=True)

  # init constants and match data types before entering loop
  i = 0
  anchor_id = 0
  const = tf.cast(tf.convert_to_tensor([1.]), dtype=dtype)
  mask = tf.cast(mask, dtype=dtype)
  rand_update = 0.0

  for box_id in range(num_boxes):
    # if the width or height of the box is zero, skip it
    if K.all(tf.math.equal(boxes[box_id, 2:4], 0)):
      continue
    # after pre processing, if the box is not in the i
    # image bounds anymore skip the box
    if K.any(tf.math.less(boxes[box_id, 0:2], 0.0)) or K.any(
        tf.math.greater_equal(boxes[box_id, 0:2], 1.0)):
      continue
    if use_tie_breaker:
      for anchor_id in range(tf.shape(anchors)[-1]):
        index = tf.math.equal(anchors[box_id, anchor_id], mask)
        if K.any(index):
          # using the boolean index mask to determine exactly which
          # anchor box was used
          p = tf.cast(K.argmax(tf.cast(index, dtype=tf.int32)), dtype=tf.int32)
          # determine if the index was used or not
          used = depth_track[y[box_id], x[box_id], p]
          # defualt used upadte value
          uid = 1

          # if anchor_id is 0, this is the best matched anchor for this box
          # with the highest IOU
          if anchor_id == 0:
            # write the box to the update list
            # create random numbr to trigger a replacment if the cell
            # is used already
            if tf.math.equal(used, 1):
              rand_update = tf.random.uniform([], maxval=1)
            else:
              rand_update = 1.0

            if rand_update > 0.5:
              # write the box to the update list
              update_index = update_index.write(i, [y[box_id], x[box_id], p])
              value = K.concatenate([boxes[box_id], const, classes[box_id]])
              update = update.write(i, value)

          # if used is 2, this cell is filled with a non-optimal box
          # if used is 0, the cell in the ground truth is not yet consumed
          # in either case you can replace that cell with a new box, as long
          # as it is not consumed by an optimal box with anchor_id = 0
          elif tf.math.equal(used, 2) or tf.math.equal(used, 0):
            uid = 2
            # write the box to the update list
            update_index = update_index.write(i, [y[box_id], x[box_id], p])
            value = K.concatenate([boxes[box_id], const, classes[box_id]])
            update = update.write(i, value)

          depth_track = tf.tensor_scatter_nd_update(depth_track,
                                                    [(y[box_id], x[box_id], p)],
                                                    [uid])
          i += 1
    else:
      index = tf.math.equal(anchors[box_id, 0], mask)
      # if any there is an index match
      if K.any(index):
        # find the index
        p = tf.cast(K.argmax(tf.cast(index, dtype=tf.int32)), dtype=tf.int32)
        # update the list of used boxes
        update_index = update_index.write(i, [y[box_id], x[box_id], p])
        value = K.concatenate([boxes[box_id], const, classes[box_id]])
        update = update.write(i, value)
        i += 1

  # if the size of the update list is not 0, do an update, other wise,
  # no boxes and pass an empty grid
  if tf.math.greater(update_index.size(), 0):
    update_index = update_index.stack()
    update = update.stack()
    full = tf.tensor_scatter_nd_update(full, update_index, update)
  return full


def build_batch_grided_gt(y_true, mask, size, num_classes, dtype,
                          use_tie_breaker):
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

  # get the batch size
  batches = tf.shape(boxes)[0]
  # get the number of boxes in the ground truth boxs
  num_boxes = tf.shape(boxes)[1]
  # get the number of anchor boxes used for this anchor scale
  len_masks = tf.shape(mask)[0]

  # init a fixed memeory size grid for this prediction scale
  # [batch, size, size, # of anchors, 1 + 1 + number of anchors per scale]
  full = tf.zeros([batches, size, size, len_masks, 1 + 4 + 1], dtype=dtype)
  # init a grid to use to track which locations have already
  # been used before (for the tie breaker)
  depth_track = tf.zeros((batches, size, size, len_masks), dtype=tf.int32)

  # rescale the x and y centers to the size of the grid [size, size]
  x = tf.cast(boxes[..., 0] * tf.cast(size, dtype=dtype), dtype=tf.int32)
  y = tf.cast(boxes[..., 1] * tf.cast(size, dtype=dtype), dtype=tf.int32)

  # init all the tensorArrays to be used in storeing the index and the values
  # to be used to update both depth_track and full
  update_index = tf.TensorArray(tf.int32, size=0, dynamic_size=True)
  update = tf.TensorArray(dtype, size=0, dynamic_size=True)

  # init constants and match data types before entering loop
  i = 0
  anchor_id = 0
  const = tf.cast(tf.convert_to_tensor([1.]), dtype=dtype)
  mask = tf.cast(mask, dtype=dtype)
  rand_update = 0.0

  for batch in range(batches):
    for box_id in range(num_boxes):
      # if the width or height of the box is zero, skip it
      if K.all(tf.math.equal(boxes[batch, box_id, 2:4], 0)):
        continue
      # after pre processing, if the box is not in the image bounds anymore
      # skip the box
      if K.any(tf.math.less(boxes[batch, box_id, 0:2], 0.0)) or K.any(
          tf.math.greater_equal(boxes[batch, box_id, 0:2], 1.0)):
        continue
      if use_tie_breaker:
        for anchor_id in range(tf.shape(anchors)[-1]):
          index = tf.math.equal(anchors[batch, box_id, anchor_id], mask)
          if K.any(index):
            # using the boolean index mask to determine exactly which anchor
            #  box was used
            p = tf.cast(
                K.argmax(tf.cast(index, dtype=tf.int32)), dtype=tf.int32)
            # determine if the index was used or not
            used = depth_track[batch, y[batch, box_id], x[batch, box_id], p]
            # defualt used upadte value
            uid = 1

            # if anchor_id is 0, this is the best matched anchor for this box
            # with the highest IOU
            if anchor_id == 0:
              # create random number to trigger a replacment if the cell
              # is used already
              if tf.math.equal(used, 1):
                rand_update = tf.random.uniform([], maxval=1)
              else:
                rand_update = 1.0

              if rand_update > 0.5:
                # write the box to the update list
                update_index = update_index.write(
                    i, [batch, y[batch, box_id], x[batch, box_id], p])
                value = K.concatenate(
                    [boxes[batch, box_id], const, classes[batch, box_id]])
                update = update.write(i, value)

            # if used is 2, this cell is filled with a non-optimal box
            # if used is 0, the cell in the ground truth is not yet consumed
            # in either case you can replace that cell with a new box, as long
            # as it is not consumed by an optimal box with anchor_id = 0
            elif tf.math.equal(used, 2) or tf.math.equal(used, 0):
              uid = 2
              # write the box to the update list
              update_index = update_index.write(
                  i, [batch, y[batch, box_id], x[batch, box_id], p])
              value = K.concatenate(
                  [boxes[batch, box_id], const, classes[batch, box_id]])
              update = update.write(i, value)

            # update the used index for where and how the box was placed
            depth_track = tf.tensor_scatter_nd_update(
                depth_track, [(batch, y[batch, box_id], x[batch, box_id], p)],
                [uid])
            i += 1
      else:
        index = tf.math.equal(anchors[batch, box_id, 0], mask)
        if K.any(index):
          # if any there is an index match
          p = tf.cast(K.argmax(tf.cast(index, dtype=tf.int32)), dtype=tf.int32)
          # write the box to the update list
          update_index = update_index.write(
              i, [batch, y[batch, box_id], x[batch, box_id], p])
          value = K.concatenate(
              [boxes[batch, box_id], const, classes[batch, box_id]])
          update = update.write(i, value)
          i += 1

  # if the size of the update list is not 0, do an update, other wise,
  # no boxes and pass an empty grid
  if tf.math.greater(update_index.size(), 0):
    update_index = update_index.stack()
    update = update.stack()
    full = tf.tensor_scatter_nd_update(full, update_index, update)
  return full
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


def shift_zeros2(mask, squeeze=True, fill=0):
  mask = tf.cast(mask, tf.float32)
  if squeeze:
    mask = tf.squeeze(mask, axis=-1)

  k = tf.shape(mask)[-1]
  mask, ind = tf.math.top_k(mask, k=k, sorted=True)
  return mask, ind


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


def _shift_zeros_full(boxes, classes, num_instances, mask=None, yxyx=True):
  is_batch = True
  boxes_shape = boxes.get_shape()
  if boxes_shape.ndims == 2:
    is_batch = False
    boxes = tf.expand_dims(boxes, 0)
    classes = tf.expand_dims(classes, 0)
  elif boxes_shape.ndims is None:
    is_batch = False
    boxes = tf.expand_dims(image, 0)
    classes = tf.expand_dims(classes, 0)
    boxes.set_shape([None] * 3)
    classes.set_shape([None] * 2)
  elif boxes_shape.ndims != 3:
    raise ValueError('\'box\' (shape %s) must have either 3 or 4 dimensions.')

  if yxyx:
    boxes = box_ops.yxyx_to_xcycwh(boxes)
  x, y, w, h = tf.split(boxes, 4, axis=-1)

  if mask is None:
    mask = tf.logical_and(w > 0, h > 0)
  elif not is_batch:
    mask = tf.expand_dims(mask, 0)

  # tf.print(tf.shape(x), tf.shape(mask))
  mask, ind = shift_zeros2(mask)
  ind_m = tf.ones_like(ind) * tf.expand_dims(
      tf.range(0,
               tf.shape(ind)[0]), axis=-1)
  ind = tf.stack([tf.reshape(ind_m, [-1]), tf.reshape(ind, [-1])], axis=-1)

  classes_shape = tf.shape(classes)
  classes_ = tf.gather_nd(classes, ind)
  classes = (tf.reshape(classes_, classes_shape) *
             tf.cast(mask, classes.dtype)) - (1 - tf.cast(mask, classes.dtype))

  mask = tf.expand_dims(tf.cast(mask, x.dtype), axis=-1)
  x_shape = tf.shape(x)
  x_ = tf.gather_nd(tf.squeeze(x, axis=-1), ind)
  x = tf.reshape(x_, x_shape) * mask  #- (1 - tf.cast(mask, x.dtype))

  y_shape = tf.shape(y)
  y_ = tf.gather_nd(tf.squeeze(y, axis=-1), ind)
  y = tf.reshape(y_, y_shape) * mask  #- (1 - tf.cast(mask, y.dtype))

  w_shape = tf.shape(w)
  w_ = tf.gather_nd(tf.squeeze(w, axis=-1), ind)
  w = tf.reshape(w_, w_shape) * mask  #- (1 - tf.cast(mask, w.dtype))

  h_shape = tf.shape(h)
  h_ = tf.gather_nd(tf.squeeze(h, axis=-1), ind)
  h = tf.reshape(h_, h_shape) * mask  #- (1 - tf.cast(mask, h.dtype))

  boxes = tf.cast(tf.concat([x, y, w, h], axis=-1), boxes.dtype)
  boxes = _pad_max_instances(boxes, num_instances, pad_axis=-2, pad_value=0)
  classes = _pad_max_instances(
      classes, num_instances, pad_axis=-1, pad_value=-1)
  if yxyx:
    boxes = box_ops.xcycwh_to_yxyx(boxes)

  if not is_batch:
    boxes = tf.squeeze(boxes, axis=0)
    classes = tf.squeeze(classes, axis=0)
  return boxes, classes


def near_edge_adjustment(boxes,
                         y_lower_bound,
                         x_lower_bound,
                         y_upper_bound,
                         x_upper_bound,
                         keep_thresh=0.25,
                         aggressive=False):
  x_lower_bound = tf.clip_by_value(x_lower_bound, 0.0, 1.0 - K.epsilon())
  y_lower_bound = tf.clip_by_value(y_lower_bound, 0.0, 1.0 - K.epsilon())
  x_upper_bound = tf.clip_by_value(x_upper_bound, 0.0, 1.0 - K.epsilon())
  y_upper_bound = tf.clip_by_value(y_upper_bound, 0.0, 1.0 - K.epsilon())

  x_lower_bound = tf.cast(x_lower_bound, boxes.dtype)
  y_lower_bound = tf.cast(y_lower_bound, boxes.dtype)
  x_upper_bound = tf.cast(x_upper_bound, boxes.dtype)
  y_upper_bound = tf.cast(y_upper_bound, boxes.dtype)
  keep_thresh = tf.cast(keep_thresh, boxes.dtype)

  y_min, x_min, y_max, x_max = tf.split(
      tf.cast(boxes, x_lower_bound.dtype), 4, axis=-1)

  # locations where atleast 25% of the image is in frame but the certer is not
  if keep_thresh == 0:
    y_mask1 = tf.math.logical_and(y_upper_bound > y_min, y_max > y_upper_bound)
    x_mask1 = tf.math.logical_and(x_upper_bound > x_min, x_max > x_upper_bound)
    y_max = tf.where(y_mask1, y_upper_bound, y_max)
    x_max = tf.where(x_mask1, x_upper_bound, x_max)
    y_mask1 = tf.math.logical_and(x_max > x_lower_bound, y_min < y_lower_bound)
    x_mask1 = tf.math.logical_and(x_max > x_lower_bound, x_min < x_lower_bound)
    y_min = tf.where(y_mask1, y_lower_bound, y_min)
    x_min = tf.where(x_mask1, x_lower_bound, x_min)
    boxes = tf.cast(
        tf.concat([y_min, x_min, y_max, x_max], axis=-1), boxes.dtype)
  elif aggressive:
    boxes = box_ops.yxyx_to_xcycwh(boxes)
    x, y, w, h = tf.split(tf.cast(boxes, x_lower_bound.dtype), 4, axis=-1)
    y_mask1 = tf.math.logical_and(
        (y_upper_bound - y_min > tf.cast(h * 0.5 * keep_thresh, y_min.dtype)),
        (y > y_upper_bound))
    x_mask1 = tf.math.logical_and(
        (x_upper_bound - x_min > tf.cast(w * 0.5 * keep_thresh, x_min.dtype)),
        (x > x_upper_bound))
    y_max = tf.where(y_mask1, y_upper_bound, y_max)
    x_max = tf.where(x_mask1, x_upper_bound, x_max)
    boxes = tf.cast(
        tf.concat([y_min, x_min, y_max, x_max], axis=-1), boxes.dtype)
    boxes = box_ops.yxyx_to_xcycwh(boxes)
    x, y, w, h = tf.split(tf.cast(boxes, x_lower_bound.dtype), 4, axis=-1)
    y_mask1 = tf.math.logical_and(
        (y_max - y_lower_bound > tf.cast(h * 0.5 * keep_thresh, y_max.dtype)),
        (y < y_lower_bound))
    x_mask1 = tf.math.logical_and(
        (x_max - x_lower_bound > tf.cast(w * 0.5 * keep_thresh, x_max.dtype)),
        (x < x_lower_bound))
    y_min = tf.where(y_mask1, y_lower_bound, y_min)
    x_min = tf.where(x_mask1, x_lower_bound, x_min)
    boxes = tf.cast(
        tf.concat([y_min, x_min, y_max, x_max], axis=-1), boxes.dtype)
  else:
    # locations where atleast 25% of the image is in frame but the certer is not
    boxes = box_ops.yxyx_to_xcycwh(boxes)
    x, y, w, h = tf.split(tf.cast(boxes, x_lower_bound.dtype), 4, axis=-1)
    y_mask1 = tf.math.logical_and(
        (y_upper_bound - y_min > tf.cast(h * 0.5 * keep_thresh, y_min.dtype)),
        (y > y_upper_bound))
    x_mask1 = tf.math.logical_and(
        (x_upper_bound - x_min > tf.cast(w * 0.5 * keep_thresh, x_min.dtype)),
        (x > x_upper_bound))

    y_new = tf.where(y_mask1, y_upper_bound, y)
    h_new = tf.where(y_mask1, (y_new - y_min) * 2, h)
    x_new = tf.where(x_mask1, x_upper_bound, x)
    w_new = tf.where(x_mask1, (x_new - x_min) * 2, w)

    boxes = tf.cast(
        tf.concat([x_new, y_new, w_new, h_new], axis=-1), boxes.dtype)
    x, y, w, h = tf.split(tf.cast(boxes, x_lower_bound.dtype), 4, axis=-1)
    boxes = box_ops.xcycwh_to_yxyx(boxes)
    y_min, x_min, y_max, x_max = tf.split(
        tf.cast(boxes, x_lower_bound.dtype), 4, axis=-1)

    y_mask1 = tf.math.logical_and(
        (y_max - y_lower_bound > tf.cast(h * 0.5 * keep_thresh, y_max.dtype)),
        (y < y_lower_bound))
    x_mask1 = tf.math.logical_and(
        (x_max - x_lower_bound > tf.cast(w * 0.5 * keep_thresh, x_max.dtype)),
        (x < x_lower_bound))

    y_new = tf.where(y_mask1, y_lower_bound, y)
    h_new = tf.where(y_mask1, (y_max - y_new) * 2, h)
    x_new = tf.where(x_mask1, x_lower_bound, x)
    w_new = tf.where(x_mask1, (x_max - x_new) * 2, w)

    boxes = tf.cast(
        tf.concat([x_new, y_new, w_new, h_new], axis=-1), boxes.dtype)
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
def random_op_image(image, jfactor, zx, zy, tfactor):
  image, jitter_info = random_jitter(image, jfactor)
  image, translate_info = random_translate(image, tfactor)
  image, crop_info = random_zoom(image, zx, zy)
  crop_info.update(translate_info)
  return image, crop_info


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
  return image, {'translate_offset': [t_y, t_x]}


def translate_boxes(box, classes, translate_x, translate_y):
  with tf.name_scope('translate_boxes'):
    box = box_ops.yxyx_to_xcycwh(box)
    x, y, w, h = tf.split(box, 4, axis=-1)
    x = x + translate_x
    y = y + translate_y
    box = tf.cast(tf.concat([x, y, w, h], axis=-1), box.dtype)
    box = box_ops.xcycwh_to_yxyx(box)
  return box, classes


def random_zoom(image, zx, zy):
  height, width = get_image_shape(image)
  width = tf.cast(tf.cast(width, zx.dtype) * zx, width.dtype)
  height = tf.cast(tf.cast(height, zy.dtype) * zy, height.dtype)
  return random_crop_or_pad(image, width, height, random_patch=True)


def random_jitter(image, jfactor):
  jx = 1 + tf.random.uniform(
      minval=-jfactor, maxval=jfactor, shape=(), dtype=tf.float32)
  jy =  1 + tf.random.uniform(
      minval=-jfactor, maxval=jfactor, shape=(), dtype=tf.float32)

  height, width = get_image_shape(image)
  width = tf.cast(width, jx.dtype) * jx
  height = tf.cast(height, jy.dtype) * jy
  image = tf.image.resize(image, (height, width))
  jitter_info = {'jitter_dims': [jy, jx]}
  return image, jitter_info


def random_crop_or_pad(image, target_width, target_height, random_patch=False):
  with tf.name_scope('random_crop_or_pad'):
    default_height, default_width = get_image_shape(image)
    dx = (tf.math.maximum(default_width, target_width) -
          tf.math.minimum(default_width, target_width)) // 2
    dy = (tf.math.maximum(default_height, target_height) -
          tf.math.minimum(default_height, target_height)) // 2

    if random_patch:
      dx = tf.random.uniform([], minval=0, maxval=dx * 2 +
                             1, dtype=tf.int32) if dx != 0 else 0
      dy = tf.random.uniform([], minval=0, maxval=dy * 2 +
                             1, dtype=tf.int32) if dy != 0 else 0

    if target_width > default_width:
      image, _ = pad_to_bbox(image, target_width, default_height, dx, 0)
      dx = -dx
    elif target_width < default_width:
      image, _ = crop_to_bbox(
          image, target_width, default_height, dx, 0, fix=False)

    if target_height > default_height:
      image, _ = pad_to_bbox(image, target_width, target_height, 0, dy)
      dy = -dy
    elif target_height < default_height:
      image, _ = crop_to_bbox(
          image, target_width, target_height, 0, dy, fix=False)

  crop_info = {
      'original_dims': [default_height, default_width],
      'new_dims': [target_height, target_width],
      'offset': [dy, dx],
      'fixed': False
  }
  return image, crop_info


def crop_to_bbox(image,
                 target_width,
                 target_height,
                 offset_width,
                 offset_height,
                 fix=False):
  with tf.name_scope('crop_to_bbox'):
    height, width = get_image_shape(image)
    image = tf.image.crop_to_bounding_box(image, offset_height, offset_width,
                                          target_height, target_width)
    if fix:
      image = tf.image.pad_to_bounding_box(image, offset_height, offset_width,
                                           height, width)

  crop_info = {
      'original_dims': [height, width],
      'new_dims': [target_height, target_width],
      'offset': [offset_height, offset_width],
      'fixed': fix
  }
  return image, crop_info


def filter_boxes_and_classes(boxes, classes, image_info, keep_thresh=0.01):
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

  x_lower_bound = offset_width / width
  x_upper_bound = (offset_width +
                   target_width) / width if xscaler > 1.0 else tf.cast(
                       1.0, tf.float64)

  y_lower_bound = offset_height / height
  y_upper_bound = (offset_height +
                   target_height) / height if yscaler > 1.0 else tf.cast(
                       1.0, tf.float64)

  ymin, xmin, ymax, xmax = tf.split(
      tf.cast(boxes, x_lower_bound.dtype), 4, axis=-1)
  w_mask = (xmax - xmin) > 0
  h_mask = (ymax - ymin) > 0

  boxes = near_edge_adjustment(
      boxes,
      y_lower_bound,
      x_lower_bound,
      y_upper_bound,
      x_upper_bound,
      keep_thresh=keep_thresh)
  boxes = box_ops.yxyx_to_xcycwh(boxes)
  x, y, w, h = tf.split(tf.cast(boxes, x_lower_bound.dtype), 4, axis=-1)

  x_mask_lower = tf.math.logical_and(x > x_lower_bound, x >= 0)
  y_mask_lower = tf.math.logical_and(y > y_lower_bound, y >= 0)
  x_mask_upper = tf.math.logical_and(x < x_upper_bound, x < 1)
  y_mask_upper = tf.math.logical_and(y < y_upper_bound, y < 1)

  x_mask = tf.math.logical_and(x_mask_lower, x_mask_upper)
  y_mask = tf.math.logical_and(y_mask_lower, y_mask_upper)
  mask = tf.logical_and(
      tf.math.logical_and(x_mask, y_mask), tf.logical_and(w_mask, h_mask))

  if not fixed:
    x = (x - x_lower_bound) * xscaler
    y = (y - y_lower_bound) * yscaler
    w = w * xscaler
    h = h * yscaler

  boxes = tf.cast(tf.concat([x, y, w, h], axis=-1), boxes.dtype)
  boxes = box_ops.xcycwh_to_yxyx(boxes)
  boxes, classes = _shift_zeros_full(
      boxes, classes, mask=mask, num_instances=tf.shape(boxes)[-2])
  return boxes, classes


def letter_box(image, boxes, xs = 0.5, ys = 0.5, target_dim=None):
  height, width = get_image_shape(image)
  clipper = tf.math.maximum(width, height)
  if target_dim is None:
    target_dim = clipper

  xs = tf.convert_to_tensor(xs)
  ys = tf.convert_to_tensor(ys)
  pad_width_p = clipper - width
  pad_height_p = clipper - height
  pad_height = tf.cast(tf.cast(pad_height_p, ys.dtype) * ys, tf.int32)
  pad_width = tf.cast(tf.cast(pad_width_p, xs.dtype) * xs, tf.int32)
  image = tf.image.pad_to_bounding_box(image, pad_height, pad_width,
                                       clipper, clipper)

  boxes = box_ops.yxyx_to_xcycwh(boxes)
  x, y, w, h = tf.split(boxes, 4, axis=-1)

  y *= tf.cast(height / clipper, y.dtype)
  x *= tf.cast(width / clipper, x.dtype)

  y += tf.cast((pad_height / clipper), y.dtype)
  x += tf.cast((pad_width / clipper), x.dtype)

  h *= tf.cast(height / clipper, h.dtype)
  w *= tf.cast(width / clipper, w.dtype)

  boxes = tf.concat([x, y, w, h], axis=-1)

  boxes = box_ops.xcycwh_to_yxyx(boxes)
  boxes = tf.where(h == 0, tf.zeros_like(boxes), boxes)

  image = tf.image.resize(image, (target_dim, target_dim))

  scale = target_dim/clipper
  pt_width = tf.cast(tf.cast(pad_width, scale.dtype) * scale, tf.int32)
  pt_height = tf.cast(tf.cast(pad_height, scale.dtype) * scale, tf.int32)
  pt_width_p = tf.cast(tf.cast(pad_width_p, scale.dtype) * scale, tf.int32)
  pt_height_p = tf.cast(tf.cast(pad_height_p, scale.dtype) * scale, tf.int32)
  return image, boxes, [pt_height, pt_width, target_dim - pt_height_p, target_dim - pt_width_p]

def _unletter_shift(images, infos, boxes, classes, xs = 0.5, ys = 0.5, target_shape = 416):
  num_items = tf.shape(images)[0]
  for i in range(num_items):
    image, info = crop_to_bbox(images[i], infos[i, 3], infos[i, 2], infos[i, 1], infos[i, 0])
    box, class_ = filter_boxes_and_classes(boxes[i], classes[i], info)
    image, box, _ = letter_box(image, box, xs = xs, ys = ys, target_dim=target_shape)
    images = tf.tensor_scatter_nd_update(images, [[i]], tf.expand_dims(tf.cast(image, images.dtype), axis = 0))
    boxes = tf.tensor_scatter_nd_update(boxes, [[i]], tf.expand_dims(tf.cast(box, boxes.dtype), axis = 0))
    classes = tf.tensor_scatter_nd_update(classes, [[i]], tf.expand_dims(tf.cast(class_, classes.dtype), axis = 0))
  return images, boxes, classes

def max_shape(image1, image2, image3, image4):
  height1, width1 = get_image_shape(image1)
  height2, width2 = get_image_shape(image2)
  height3, width3 = get_image_shape(image3)
  height4, width4 = get_image_shape(image4)

  maxim = tf.reduce_max([height1, width1, height2, width2, height3, width3, height4, width4])
  return maxim

def patch_four_images(images, boxes, classes, info):
  num_instances = tf.shape(boxes)[-2]
  image1, image2, image3, image4 = tf.split(images, 4, axis=0)
  info1, info2, info3, info4 = tf.split(info, 4, axis=0)
  box1, box2, box3, box4 = tf.split(boxes, 4, axis=0)
  class1, class2, class3, class4 = tf.split(classes, 4, axis=0)

  maxim = max_shape(image1, image2, image3, image4)
  image1, box1, class1 = _unletter_shift(image1, info1, box1, class1, xs = 1.0, ys = 1.0, target_shape=maxim)
  image2, box2, class2 = _unletter_shift(image2, info2, box2, class2, xs = 0.0, ys = 1.0, target_shape=maxim)
  image3, box3, class3 = _unletter_shift(image3, info3, box3, class3, xs = 1.0, ys = 0.0, target_shape=maxim)
  image4, box4, class4 = _unletter_shift(image4, info4, box4, class4, xs = 0.0, ys = 0.0, target_shape=maxim)

  patch1 = tf.concat([image1, image2], axis=-2)
  patch2 = tf.concat([image3, image4], axis=-2)
  full_image = tf.concat([patch1, patch2], axis=-3)

  box1 = box1 * 0.5
  box2, class2 = translate_boxes(box2 * 0.5, class2, .5, 0)
  box3, class3 = translate_boxes(box3 * 0.5, class3, 0, .5)
  box4, class4 = translate_boxes(box4 * 0.5, class4, .5, .5)

  boxes = tf.concat([box1, box2, box3, box4], axis = -2)
  classes = tf.concat([class1, class2, class3, class4], axis = -1)
  boxes, classes = _shift_zeros_full(boxes, classes, num_instances, yxyx=True)
  return full_image, boxes, classes

def mosaic(images,
           boxes,
           classes,
           image_info,
           output_size,
           masks=None,
           crop_delta=0.6,
           keep_thresh=0.25):
  crop_delta = tf.convert_to_tensor(crop_delta)
  full_image, full_boxes, full_classes = patch_four_images(images, boxes, classes, image_info)
  height, width = get_image_shape(full_image)
  # num_instances = tf.shape(boxes)[-2]

  # box1, box2, box3, box4 = tf.split(boxes * 0.5, 4, axis=0)
  # class1, class2, class3, class4 = tf.split(classes, 4, axis=0)
  # #translate boxes
  # box2, class2 = translate_boxes(box2, class2, .5, 0)
  # box3, class3 = translate_boxes(box3, class3, 0, .5)
  # box4, class4 = translate_boxes(box4, class4, .5, .5)
  # full_boxes = tf.concat([box1, box2, box3, box4], axis=-2)
  # full_classes = tf.concat([class1, class2, class3, class4], axis=-1)
  # full_boxes, full_classes = _shift_zeros_full(
  #     full_boxes, full_classes, num_instances, yxyx=True)

  crop_delta = rand_uniform_strong(crop_delta, 1.0)  #1 + crop_delta * 3/5)
  #crop_delta = rand_scale(crop_delta) 
  full_image, image_info = random_crop_or_pad(
      full_image,
      target_width=tf.cast(tf.cast(height, tf.float32) * crop_delta, tf.int32),
      target_height=tf.cast(tf.cast(width, tf.float32) * crop_delta, tf.int32),
      random_patch=True)
  full_boxes, full_classes = filter_boxes_and_classes(
      full_boxes, full_classes, image_info, keep_thresh=keep_thresh)
  full_image = tf.image.resize(full_image, [output_size, output_size])

  num_items = tf.shape(full_image)[0]
  info = tf.convert_to_tensor([[0, 0, output_size, output_size]])
  return (tf.cast(full_image, images.dtype), tf.cast(full_boxes, boxes.dtype),
          tf.cast(full_classes, classes.dtype), info)


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
    anchors_x = anchors[..., 0] / width
    anchors_y = anchors[..., 1] / height
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

    true_prod = tf.reduce_prod(true_wh, axis = -1, keepdims = True)
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
    y_true = tf.concat([hold, true_wh], axis = -1)

    # tf.print(tf.shape(true_wh), tf.shape(anchors))
  return get_best_anchor2(y_true, anchors, width=width, height=height, iou_thresh=iou_thresh)


def _get_num_reps(anchors, mask, box_mask):
  mask = tf.expand_dims(mask, 0)
  mask = tf.expand_dims(mask, 0)
  mask = tf.expand_dims(mask, 0)
  box_mask = tf.expand_dims(box_mask, -1)
  box_mask = tf.expand_dims(box_mask, -1)

  anchors = tf.expand_dims(anchors, axis = -1)
  anchors_primary, anchors_alternate = tf.split(anchors, [1, -1], axis = -2)
  fillin = tf.zeros_like(anchors_primary) - 1
  anchors_alternate = tf.concat([fillin, anchors_alternate], axis = -2)

  viable_primary = tf.logical_and(box_mask, anchors_primary == mask)
  viable_alternate = tf.logical_and(box_mask, anchors_alternate == mask)

  viable_primary = tf.where(viable_primary)
  viable_alternate = tf.where(viable_alternate)

  # tf.print(viable_primary.shape)
  # tf.print(viable_alternate)

  viable = anchors == mask
  acheck = tf.reduce_any(viable, axis = -1)
  reps = tf.reduce_sum(tf.cast(acheck, mask.dtype), axis = -1)
  return reps, viable_primary, viable_alternate

def _gen_utility(boxes):
  eq0 = tf.reduce_all(tf.math.less_equal(boxes[..., 2:4], 0), axis = -1)
  gtlb = tf.reduce_any(tf.math.less(boxes[..., 0:2], 0.0), axis = -1)
  ltub = tf.reduce_any(tf.math.greater_equal(boxes[..., 0:2], 1.0), axis = -1)
  # rep_mask = reps <= 0

  a = tf.logical_or(eq0, gtlb)
  b = tf.logical_or(a, ltub)
  # b = tf.logical_or(b, rep_mask)
  return tf.logical_not(b)

def build_grided_gt_ind(y_true, mask, size, num_classes, dtype, use_tie_breaker):
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

  is_batch = True
  boxes_shape = boxes.get_shape()
  if boxes_shape.ndims == 2:
    is_batch = False
    boxes = tf.expand_dims(boxes, 0)
    classes = tf.expand_dims(classes, 0)
    anchors = tf.expand_dims(anchors, 0)
    ious = tf.expand_dims(ious, 0)
  elif boxes_shape.ndims is None:
    is_batch = False
    boxes = tf.expand_dims(image, 0)
    classes = tf.expand_dims(classes, 0)
    anchors = tf.expand_dims(anchors, 0)
    ious = tf.expand_dims(ious, 0)
    boxes.set_shape([None] * 3)
    classes.set_shape([None] * 2)
    anchors.set_shape([None] * 3)
    ious.set_shape([None] * 3)
  elif boxes_shape.ndims != 3:
    raise ValueError('\'box\' (shape %s) must have either 3 or 4 dimensions.')

  # get the batch size
  batches = tf.shape(boxes)[0]
  # get the number of boxes in the ground truth boxs
  num_boxes = tf.shape(boxes)[-2]
  # get the number of anchor boxes used for this anchor scale
  len_masks = tf.shape(mask)[0]
  # number of anchors
  num_anchors = tf.shape(anchors)[-1]
  num_instances = num_boxes



  # rescale the x and y centers to the size of the grid [size, size]
  mask = tf.cast(mask, dtype=dtype)
  const = tf.cast(tf.convert_to_tensor([1.]), dtype=boxes.dtype)
  x = tf.cast(boxes[..., 0] * tf.cast(size, dtype=dtype), dtype=tf.int32)
  y = tf.cast(boxes[..., 1] * tf.cast(size, dtype=dtype), dtype=tf.int32)
  box_mask = _gen_utility(boxes)
  num_reps, viable_primary, viable_alternate = _get_num_reps(anchors, mask, box_mask)
  viable_primary = tf.cast(viable_primary, tf.int32)
  viable_alternate = tf.cast(viable_alternate, tf.int32)

  i = 0
  num_boxes_written = tf.zeros([batches], dtype = tf.int32)
  list_indx = tf.TensorArray(tf.int32, size=0, dynamic_size=True)
  list_ind_val = tf.TensorArray(tf.int32, size=0, dynamic_size=True)
  list_ind_sample = tf.TensorArray(dtype, size=0, dynamic_size=True)

  update_index = tf.TensorArray(tf.int32, size=0, dynamic_size=True)
  update = tf.TensorArray(dtype, size=0, dynamic_size=True)

  num_primary = tf.shape(viable_primary)[0]
  for val in range(num_primary):
    idx = viable_primary[val]
    batch, obj_id, anchor, anchor_idx = idx[0], idx[1], idx[2], idx[3]

    count = num_boxes_written[batch]
    if count >= num_instances:
      continue
    batch_ind = batch * num_instances

    reps = num_reps[batch, obj_id]
    box = boxes[batch, obj_id]
    classif = classes[batch, obj_id]
    iou = tf.convert_to_tensor([ious[batch, obj_id, anchor]])
    # reps_ = tf.convert_to_tensor([reps])
    sample = tf.concat([box, const, classif, iou], axis = -1)
    y_, x_ = y[batch, obj_id], x[batch, obj_id]
    list_indx = list_indx.write(i, [batch_ind + count])
    list_ind_val = list_ind_val.write(i, [y_, x_, anchor_idx])
    list_ind_sample = list_ind_sample.write(i, sample)

    update_index = update_index.write(i, [batch, y_, x_, anchor_idx])
    update = update.write(i, const)
    count += 1
    i += 1
    num_boxes_written = tf.tensor_scatter_nd_update(num_boxes_written, [[batch]], [count])

  if use_tie_breaker:
    num_alternate = tf.shape(viable_alternate)[0]
    for val in range(num_alternate):
      idx = viable_alternate[val]
      batch, obj_id, anchor, anchor_idx = idx[0], idx[1], idx[2], idx[3]
      count = num_boxes_written[batch]
      if count >= num_instances:
        continue
      batch_ind = batch * num_instances

      reps = num_reps[batch, obj_id]
      box = boxes[batch, obj_id]
      classif = classes[batch, obj_id]
      iou = tf.convert_to_tensor([ious[batch, obj_id, anchor]])
      # reps_ = tf.convert_to_tensor([reps])
      sample = tf.concat([box, const, classif, iou], axis = -1)
      y_, x_ = y[batch, obj_id], x[batch, obj_id]
      list_indx = list_indx.write(i, [batch_ind + count])
      list_ind_val = list_ind_val.write(i, [y_, x_, anchor_idx])
      list_ind_sample = list_ind_sample.write(i, sample)

      update_index = update_index.write(i, [batch, y_, x_, anchor_idx])
      update = update.write(i, const)
      count += 1
      i += 1
      num_boxes_written = tf.tensor_scatter_nd_update(num_boxes_written, [[batch]], [count])

  num_coords = 4 + 1 + 1 + 1
  indexes = tf.zeros([batches * num_instances, 3], tf.int32)
  gridvals = tf.zeros([batches * num_instances, num_coords], boxes.dtype)
  full = tf.zeros([batches, size, size, len_masks, 1], dtype=dtype)

  if tf.math.greater(list_indx.size(), 0):
    list_indx = list_indx.stack()
    list_ind_val = list_ind_val.stack()
    list_ind_sample = list_ind_sample.stack()
    indexes = tf.tensor_scatter_nd_update(indexes, list_indx, list_ind_val)
    gridvals = tf.tensor_scatter_nd_update(gridvals, list_indx, list_ind_sample)

  if tf.math.greater(update_index.size(), 0):
    update_index = update_index.stack()
    update = update.stack()
    full = tf.tensor_scatter_nd_add(full, update_index, update)

  if is_batch:
    indexes = tf.reshape(indexes, [batches, -1, 3])
    gridvals = tf.reshape(gridvals, [batches, -1,  num_coords])

  if not is_batch:
    full = tf.squeeze(full, axis=0)

  # if is_batch: 
  #   reps = tf.gather_nd(full, indexes, batch_dims = 1)
  # else:
  #   reps = tf.gather_nd(full, indexes, batch_dims = 0)
  
  # reps = reps * tf.expand_dims(gridvals[..., 4], axis = -1)
  # reps = tf.where(reps == 0.0, tf.ones_like(reps), reps)
  # gridvals = tf.concat([gridvals, reps], axis = -1)

  # full = tf.clip_by_value(full, 0.0, 1.0)

  # tf.print(tf.reduce_max(full))
  return indexes, gridvals, full


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
    
    
    scale = tf.cast(ishape[:2]/ishape[:2], tf.float32)
    offset = tf.cast(crop_offset[:2], tf.float32)

    info = tf.stack([tf.cast(ishape[:2], tf.float32), 
                     tf.cast(crop_size[:2], tf.float32),
                     scale,
                     offset], axis = 0)
    return cropped_image, info


def random_pad(image, area):
  with tf.name_scope('pad_to_bbox'):
    height, width = get_image_shape(image)

    rand_area = tf.cast(area, tf.float32) #tf.random.uniform([], 1.0, area)
    target_height = tf.cast(tf.sqrt(rand_area) * tf.cast(height, rand_area.dtype), tf.int32)
    target_width = tf.cast(tf.sqrt(rand_area) * tf.cast(width, rand_area.dtype), tf.int32)

    offset_height = tf.random.uniform([], 0 , target_height - height + 1, dtype = tf.int32)
    offset_width = tf.random.uniform([], 0 , target_width - width + 1, dtype = tf.int32)

    image = tf.image.pad_to_bounding_box(image, offset_height, offset_width,
                                         target_height, target_width)
  
  ishape = tf.convert_to_tensor([height, width])
  oshape = tf.convert_to_tensor([target_height, target_width])
  offset = tf.convert_to_tensor([offset_height, offset_width])

  scale = tf.cast(ishape[:2]/ishape[:2], tf.float32)
  offset = tf.cast(-offset, tf.float32) # * scale

  info = tf.stack([tf.cast(ishape[:2], tf.float32), 
                    tf.cast(oshape[:2], tf.float32),
                    scale,
                    offset], axis = 0)

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
  ymi, xmi, yma, xma = tf.split(boxes, 4, axis = -1)
  
  tl = tf.concat([xmi, ymi], axis = -1) 
  bl = tf.concat([xmi, yma], axis = -1)
  tr = tf.concat([xma, ymi], axis = -1) 
  br = tf.concat([xma, yma], axis = -1)

  corners = tf.concat([tl, bl, tr, br], axis = -1) 
  return corners

def rotate_points(x_, y_, angle):
  sx = 0.5 - x_
  sy = 0.5 - y_

  r = tf.sqrt(sx**2 + sy **2)
  curr_theta = tf.atan2(sy,sx)

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
  ymin, xmin, ymax, xmax = tf.split(boxes, 4, axis = -1)


  tlx, tly = rotate_points(xmin, ymin, angle)
  blx, bly = rotate_points(xmin, ymax, angle)
  trx, try_ = rotate_points(xmax, ymin, angle)
  brx, bry = rotate_points(xmax, ymax, angle)

  xs = tf.concat([tlx, blx, trx, brx], axis = -1)
  ys = tf.concat([tly, bly, try_, bry], axis = -1)

  xmin = tf.reduce_min(xs, axis = -1, keepdims = True)
  ymin = tf.reduce_min(ys, axis = -1, keepdims = True)
  xmax = tf.reduce_max(xs, axis = -1, keepdims = True)
  ymax = tf.reduce_max(ys, axis = -1, keepdims = True)


  boxes = tf.concat([ymin, xmin, ymax, xmax], axis = -1)  
  return boxes
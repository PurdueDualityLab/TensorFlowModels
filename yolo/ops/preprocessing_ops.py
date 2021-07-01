from random import random
from numpy import float32
from numpy.core.fromnumeric import resize
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras.backend as K
from yolo.ops import box_ops
from official.vision.beta.ops import preprocess_ops
from official.vision.beta.ops import box_ops as bbox_ops


def rand_uniform_strong(minval, 
                        maxval, 
                        precalc_val = None, 
                        dtype=tf.float32):
  """
  Equivalent to tf.random.uniform, except that minval and maxval are flipped if
  minval is greater than maxval.

  Args:
    minval: An `int` for a lower or upper endpoint of the interval from which to
      choose the random number.
    maxval: An `int` for the other endpoint.
    dtype: The output type of the tensor.
  
  Returns:
    A random tensor of type dtype that falls between minval and maxval excluding
    the bigger one.
  """
  if minval > maxval:
    minval, maxval = maxval, minval
  
  if precalc_val is None:
    precalc_val = tf.random.uniform([], minval=0, maxval=1, dtype=dtype) 
    val = (precalc_val * (maxval - minval)) + minval;
  else:
    val = tf.random.uniform([], minval=minval, maxval=maxval, dtype=dtype)
  return val 

def rand_strong(minval, maxval, dtype=tf.float32):
  """
  Equivalent to tf.random.uniform, except that minval and maxval are flipped if
  minval is greater than maxval.

  Args:
    minval: An `int` for a lower or upper endpoint of the interval from which to
      choose the random number.
    maxval: An `int` for the other endpoint.
    dtype: The output type of the tensor.
  
  Returns:
    A random tensor of type dtype that falls between minval and maxval excluding
    the bigger one.
  """
  if minval > maxval:
    minval, maxval = maxval, minval
  return tf.random.uniform([], minval=minval, maxval=maxval, dtype=dtype)

def rand_scale(val, dtype=tf.float32):
  """
  Generates a random number for the scale. Half the time, the value is between
  [1.0, val) with uniformly distributed probability. The other half, the value
  is the reciprocal of this value.
  
  The function is identical to the one in the original implementation:
  https://github.com/AlexeyAB/darknet/blob/a3714d0a/src/utils.c#L708-L713
  
  Args:
    val: A float representing the maximum scaling allowed.
    dtype: The output type of the tensor.

  Returns:
    The random scale.
  """
  scale = rand_uniform_strong(1.0, val, dtype=dtype)
  do_ret = tf.random.uniform([], minval=0, maxval=2, dtype=tf.int32)
  if (do_ret == 1):
    return scale
  return 1.0 / scale


def _pad_max_instances(value, instances, pad_value=0, pad_axis=0):
  """
  Pad a dimension of the tensor to have a maximum number of instances filling
  additional entries with the `pad_value`.
   
  Args:
    value: An input tensor.
    instances: An int representing the maximum number of instances.
    pad_value: An int representing the value used for padding until the maximum
      number of instances is obtained.
    pad_axis: An int representing the axis index to pad.

  Returns:
    The output tensor whose dimensions match the input tensor except with the
    size along the `pad_axis` replaced by `instances`.
  """
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


def get_image_shape(image):
  """
  Get the shape of the image regardless of if the image is in the
  (batch_size, x, y, c) format or the (x, y, c) format.
  
  Args:
    image: A int tensor who has either 3 or 4 dimensions.
  
  Returns:
    A tuple representing the (height, width) of the image.
  """
  shape = tf.shape(image)
  if tf.shape(shape)[0] == 4:
    width = shape[2]
    height = shape[1]
  else:
    width = shape[1]
    height = shape[0]
  return height, width


def random_translate(image, t):
  """
  Translate the boxes by a random fraction of the image width and height between
  [-t, t).

  Args:
    image: An int tensor of either 3 or 4 dimensions representing the image or a
      batch of images.
    t: A float representing the fraction of the width and height by which to
      jitter the image.
  
  Returns:
    A tuple with three elements: a tensor representing the translated image with
    the same dimensions as `box`, and the number of pixels that the image was
    translated in both the x and y directions.
  """
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
  """
  Translate the boxes by a fixed number of pixels.
  
  Args:
    box: An tensor representing the bounding boxes in the (x, y, w, h) format
      whose last dimension is of size 4.
    classes: A tensor representing the classes of each bounding box. The shape
      of `classes` lacks the final dimension of the shape `box` which is of size
      4.
  
  Returns:
    A tuple with two elements: a tensor representing the translated boxes with
    the same dimensions as `box` and the `classes` tensor.
  """
  with tf.name_scope('translate_boxes'):
    box = box_ops.yxyx_to_xcycwh(box)
    x, y, w, h = tf.split(box, 4, axis=-1)
    x = x + translate_x
    y = y + translate_y
    box = tf.cast(tf.concat([x, y, w, h], axis=-1), box.dtype)
    box = box_ops.xcycwh_to_yxyx(box)
  return box, classes


def translate_image(image, t_y, t_x):
  """
  Translate the boxes by a fix fraction of the image width and height
  represented by (t_y, t_x).

  Args:
    image: An int tensor of either 3 or 4 dimensions representing the image or a
      batch of images.
    t_y: A float representing the fraction of the height by which to translate
      the image vertically.
    t_x: A float representing the fraction of the width by which to translate
      the image horizontally.

  Returns:
    A tuple with three elements: a tensor representing the translated image with
    the same dimensions as `box`, and the number of pixels that the image was
    translated in both the y and x directions.
  """
  with tf.name_scope('translate_boxes'):
    height, width = get_image_shape(image)
    image_jitter = tf.convert_to_tensor([t_y, t_x])
    image_dims = tf.cast(tf.convert_to_tensor([height, width]), tf.float32)
    image_jitter.set_shape([2])
    jitter = image_jitter * image_dims
    image = tfa.image.translate(image, jitter)
  return image, jitter[0], jitter[1]


def random_rotate_image(image, max_angle):
  """
  Rotate an image by a random angle that ranges from [-max_angle, max_angle).
  
  Args:
    x_: A float tensor for the x-coordinates of the points.
    y_: A float tensor for the y-coordinates of the points.
    angle: A float representing the counter-clockwise rotation in radians.
  
  Returns:
    The rotated points.
  """
  angle = tf.random.uniform([],
                            minval=-max_angle,
                            maxval=max_angle,
                            dtype=tf.float32)
  deg = angle * 3.14 / 360.
  deg.set_shape(())
  image = tfa.image.rotate(image, deg, interpolation='BILINEAR')
  return image, deg

def get_corners(boxes):
  """
  Get the coordinates of the 4 courners of a set of bounding boxes.
  
  Args:
    boxes: A tensor whose last axis has a length of 4 and is in the format
      (ymin, xmin, ymax, xmax)
  
  Returns:
    A tensor whose dimensions are the same as `boxes` but has an additional axis
    of size 2 representing the 4 courners of the bounding boxes in the order
    (tl, bl, tr, br).
  """

  ymi, xmi, yma, xma = tf.split(boxes, 4, axis=-1)

  tl = tf.concat([xmi, ymi], axis=-1)
  bl = tf.concat([xmi, yma], axis=-1)
  tr = tf.concat([xma, ymi], axis=-1)
  br = tf.concat([xma, yma], axis=-1)

  corners = tf.concat([tl, bl, tr, br], axis=-1)
  return corners


def rotate_points(x_, y_, angle):
  """
  Rotate points by an angle about the center of the image.
  
  Args:
    x_: A float tensor for the x-coordinates of the points.
    y_: A float tensor for the y-coordinates of the points.
    angle: A float representing the counter-clockwise rotation in radians.
  
  Returns:
    The rotated points.
  """
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


def rotate_boxes(boxes, angle):
  """
  Rotate bounding boxes by an angle about the center of the image.
  
  Args:
    boxes: A float tensor whose last axis represents (ymin, xmin, ymax, xmax).
    angle: A float representing the counter-clockwise rotation in radians.
  
  Returns:
    The rotated boxes.
  """

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




def random_window_crop(image, 
                       target_height, 
                       target_width, 
                       skip_zero = False, 
                       round = True):
  ishape = tf.shape(image)

  th = target_height if target_height < ishape[0] else ishape[0]
  tw = target_width if target_width < ishape[1] else ishape[1]
  if round:
    oh = tf.random.uniform([], -1, 1 + 1, tf.int32)
    ow = tf.random.uniform([], -1, 1 + 1, tf.int32)

    if skip_zero:
      if oh == 0:
        value = tf.random.uniform([], 0, 2, tf.int32)
        if value == 0: 
          oh += 1
        else:
          oh -= 1
      if ow == 0:
        value = tf.random.uniform([], 0, 2, tf.int32)
        if value == 0: 
          ow += 1
        else:
          ow -= 1
    oh = tf.cast(oh, tf.float32)
    ow = tf.cast(ow, tf.float32)
  else:
    oh = tf.random.uniform([], -1, 1)
    ow = tf.random.uniform([], -1, 1)

  crop_size = tf.convert_to_tensor([th, tw, -1])

  # tf.print(oh, ow)
  crop_offset = ishape - crop_size
  crop_offset = tf.convert_to_tensor([crop_offset[0]//2, crop_offset[1]//2, 0])

  oh = tf.cast(tf.cast(crop_offset[0], oh.dtype) * oh, tf.int32)
  ow = tf.cast(tf.cast(crop_offset[1], ow.dtype) * ow, tf.int32) 
  crop_offset += tf.convert_to_tensor([oh, ow, 0])

  cropped_image = tf.slice(image, crop_offset, crop_size)

  scale = tf.cast(ishape[:2] / ishape[:2], tf.float32)
  offset = tf.cast(crop_offset[:2], tf.float32)

  info = tf.stack([
      tf.cast(ishape[:2], tf.float32),
      tf.cast(crop_size[:2], tf.float32), scale, offset
  ],
                  axis=0)

  return cropped_image, info



def resize_and_crop_image(image,
                          desired_size,
                          padded_size,
                          letter_box=None,
                          aug_scale_min=1.0,
                          aug_scale_max=1.0,
                          random_pad=False,
                          shiftx=0.5,
                          shifty=0.5,
                          sheer=0.0,
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

    if letter_box == False:
      height, width = get_image_shape(image)
      clipper = tf.reduce_max((height, width))
      image = image = tf.image.resize(
          image, (clipper, clipper), preserve_aspect_ratio=False)
    elif letter_box == True:
      height, width = get_image_shape(image)
      clipper = tf.reduce_max((height, width))
      w_scale = width / clipper
      h_scale = height / clipper

      height_, width_ = desired_size[0], desired_size[1]
      height_ = tf.cast(h_scale * tf.cast(height_, h_scale.dtype), tf.int32)
      width_ = tf.cast(w_scale * tf.cast(width_, w_scale.dtype), tf.int32)

      image = image = tf.image.resize(
        image, (height_, width_), preserve_aspect_ratio=False)

    if sheer > 0.0: 
      h_, w_ = get_image_shape(image)

      hr = tf.cast(sheer * tf.cast(h_, tf.float32), tf.int32)
      wr = tf.cast(sheer * tf.cast(w_, tf.float32), tf.int32)

      h_ += rand_uniform_strong(-hr, hr + 1, tf.int32)
      w_ += rand_uniform_strong(-wr, wr + 1, tf.int32)
      image = tf.image.resize(image, (h_, w_))

    image_size = tf.cast(tf.shape(image)[0:2], tf.float32)
    random_jittering = (aug_scale_min != 1.0 or aug_scale_max != 1.0)

    if random_jittering:
      random_scale = tf.random.uniform([],
                                       aug_scale_min,
                                       aug_scale_max,
                                       seed=seed)
      scaled_size = tf.round(random_scale * desired_size)
    else:
      random_scale = 1.0
      scaled_size = desired_size

    scale = tf.minimum(scaled_size[0] / image_size[0],
                       scaled_size[1] / image_size[1])
    scaled_size = tf.round(image_size * scale)

    # Computes 2D image_scale.
    image_scale = scaled_size / image_size

    # Selects non-zero random offset (x, y) if scaled image is larger than
    # desired_size.
    random_pad = tf.cast(random_pad, tf.float32)
    if random_pad == 1.0:
      random_pad = tf.cast(0.5, tf.float32)
    if random_jittering:
      max_offset_ = scaled_size - desired_size

      max_offset = tf.where(
          tf.less(max_offset_, 0), tf.zeros_like(max_offset_), max_offset_)
      offset = max_offset * 0.5 + max_offset * tf.random.uniform([2,], 
                                                -random_pad, random_pad, seed=seed)
      offset = tf.cast(offset, tf.int32)
    else:
      offset = tf.zeros((2,), tf.int32)

    scaled_image = tf.image.resize(
        image, tf.cast(scaled_size, tf.int32), method=method)

    if random_jittering:
      scaled_image = scaled_image[offset[0]:offset[0] + desired_size[0],
                                  offset[1]:offset[1] + desired_size[1], :]

    scaled_size = tf.cast(tf.shape(scaled_image)[0:2], tf.int32)

    if random_pad > 0.0:
      shifty = 0.5 + rand_uniform_strong(-random_pad, random_pad, tf.float32)
      shiftx = 0.5 + rand_uniform_strong(-random_pad, random_pad, tf.float32)

    dy = tf.cast(
        tf.cast(padded_size[0] - scaled_size[0], tf.float32) * shifty,
        tf.int32)
    dx = tf.cast(
        tf.cast(padded_size[1] - scaled_size[1], tf.float32) * shiftx,
        tf.int32)

    output_image = tf.image.pad_to_bounding_box(scaled_image, dy, dx,
                                                padded_size[0], padded_size[1])

    offset -= tf.convert_to_tensor([dy, dx])

    image_info = tf.stack([
        image_size,
        tf.constant(desired_size, dtype=tf.float32), image_scale,
        tf.cast(offset, tf.float32)
    ])

    infos = [image_info]
    return output_image, infos

def intersection(a, b):
  minx = tf.maximum(a[0], b[0])
  miny = tf.maximum(a[1], b[1])

  maxx = tf.minimum(a[2], b[2])
  maxy = tf.minimum(a[3], b[3])
  return tf.convert_to_tensor([minx, miny, maxx, maxy])

def cast(values, dtype):
  return [tf.cast(value, dtype) for value in values]

def mosaic_cut(image, ow, oh, w, h, cut, ptop, pleft, pbottom, pright, shiftx, shifty):
    ow = tf.cast(ow, tf.int32)
    oh = tf.cast(oh, tf.int32)
    w = tf.cast(w, tf.int32)
    h = tf.cast(h, tf.int32)
    cut_x = tf.cast(cut[1], tf.int32)
    cut_y = tf.cast(cut[0], tf.int32)

    left_shift = tf.minimum(
                  tf.minimum(cut_x, 
                    tf.maximum(0, tf.cast(-pleft * w/ow, tf.int32))), 
                      w - cut_x)
    top_shift = tf.minimum(
                  tf.minimum(cut_y, 
                    tf.maximum(0, tf.cast(-ptop * h/oh, tf.int32))), 
                      h - cut_y)

    right_shift = tf.minimum(
                    tf.minimum(w - cut_x, 
                      tf.maximum(0, tf.cast(-pright * w/ow, tf.int32))), 
                        cut_x)
    bot_shift = tf.minimum(
                  tf.minimum(h - cut_y, 
                    tf.maximum(0, tf.cast(-pbottom * h/oh, tf.int32))), 
                      cut_y)
    
    if shiftx == 0.0 and shifty == 0.0: 
      crop_offset = [top_shift, left_shift, 0]
      crop_size = [cut_y, cut_x, -1]
    elif shiftx == 1.0 and shifty == 0.0:
      crop_offset = [top_shift, cut_x - right_shift, 0]
      crop_size = [cut_y, w - cut_x, -1]
    elif shiftx == 0.0 and shifty == 1.0:
      crop_offset = [cut_y - bot_shift, left_shift, 0]
      crop_size = [h - cut_y, cut_x, -1]
    elif shiftx == 1.0 and shifty == 1.0:
      crop_offset = [cut_y - bot_shift, cut_x - right_shift, 0]
      crop_size = [h - cut_y, w - cut_x, -1]
    else:
      raise Exception("crop mosaic not supported for floating shifts")

    ishape = tf.shape(image)[:2]
    image = tf.slice(image, crop_offset, crop_size)
    crop_info = tf.stack([
        tf.cast(ishape, tf.float32),
        tf.cast(tf.shape(image)[:2], dtype=tf.float32),
        tf.ones_like(ishape, dtype = tf.float32),
        tf.cast(crop_offset[:2], tf.float32)
    ])

    return image, crop_info


def resize_and_jitter_image(image,
                            desired_size,
                            jitter=0.0,
                            letter_box=None,
                            resize = 1.0, 
                            random_pad=True,
                            shiftx=0.0,
                            shifty=0.0,
                            cut = None,
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

    if jitter > 1 or jitter < 0:
      raise Exception("maximum change in aspect ratio must be between 0 and 1")

    if tf.cast(random_pad, tf.float32) > 0.0:
      random_pad = True 
    else:
      random_pad = False

    original_dims = tf.shape(image)[:2]
    jitter = tf.cast(jitter, tf.float32)
    ow = tf.cast(original_dims[1], tf.float32)
    oh = tf.cast(original_dims[0], tf.float32)
    w = tf.cast(desired_size[1], tf.float32)
    h = tf.cast(desired_size[0], tf.float32)

    dw = ow * jitter
    dh = oh * jitter
    
    if resize != 1:
      if resize > 1.0:
        resize_up = resize if resize > 1.0 else 1/resize
        max_rdw = ow * (1 - (1 / resize_up)) / 2
        max_rdh = oh * (1 - (1 / resize_up)) / 2
      else:
        max_rdw = 0.0
        max_rdh = 0.0 

      resize_down = resize if resize < 1.0 else 1/resize
      min_rdw = ow * (1 - (1 / resize_down)) / 2
      min_rdh = oh * (1 - (1 / resize_down)) / 2

    pleft = rand_uniform_strong(-dw, dw, dw.dtype)
    pright = rand_uniform_strong(-dw, dw, dw.dtype)
    ptop = rand_uniform_strong(-dh, dh, dh.dtype)
    pbottom = rand_uniform_strong(-dh, dh, dh.dtype)

    if resize != 1:
      pleft += rand_uniform_strong(min_rdw, max_rdw)
      pright += rand_uniform_strong(min_rdw, max_rdw)
      ptop += rand_uniform_strong(min_rdh, max_rdh)
      pbottom += rand_uniform_strong(min_rdh, max_rdh)

    if letter_box:
      image_aspect_ratio = ow/oh 
      input_aspect_ratio = w/h 

      distorted_aspect = image_aspect_ratio/input_aspect_ratio

      if distorted_aspect > 1: 
        delta_h = ((ow/input_aspect_ratio) - oh)/2
        ptop = ptop - delta_h
        pbottom = pbottom - delta_h
      else:
        delta_w = ((oh * input_aspect_ratio) - ow)/2
        pright = pright - delta_w
        pleft = pleft - delta_w

    swidth = tf.cast(ow - pleft - pright, tf.int32)
    sheight = tf.cast(oh - ptop - pbottom, tf.int32)

    pleft, pright, ptop, pbottom = cast([pleft,pright,ptop,pbottom], tf.int32)
    src_crop = intersection([ptop, pleft, sheight + ptop, swidth + pleft], 
                            [0, 0, original_dims[0], original_dims[1]])

    h_ = (src_crop[2] - src_crop[0])
    w_ = (src_crop[3] - src_crop[1])
    if random_pad:
      rmh = tf.maximum(0, -ptop)
      rmw = tf.maximum(0, -pleft)
    else:
      rmw = tf.cast(tf.cast(swidth - w_, tf.float32) * shiftx, w_.dtype)
      rmh = tf.cast(tf.cast(sheight - h_, tf.float32) * shifty, h_.dtype)
    dst_shape = [rmh,rmw,rmh + h_,rmw + w_]
    ptop, pleft, pbottom, pright = dst_shape

    pad = dst_shape * tf.convert_to_tensor([1, 1, -1, -1]) 
    pad += tf.convert_to_tensor([0, 0, sheight, swidth])

    cropped_image = tf.slice(image, 
                             [src_crop[0], src_crop[1], 0], 
                             [src_crop[2] - src_crop[0], 
                              src_crop[3] - src_crop[1], -1])

    crop_info = tf.stack([
          tf.cast(original_dims, tf.float32),
          tf.cast(tf.shape(cropped_image)[:2], dtype=tf.float32),
          tf.ones_like(original_dims, dtype = tf.float32),
          tf.cast(src_crop[:2], tf.float32)
      ])

    image_ = tf.pad(cropped_image, [[pad[0], pad[2]], 
                                    [pad[1], pad[3]], 
                                    [0, 0]])
    pad_info = tf.stack([
          tf.cast(tf.shape(cropped_image)[:2], tf.float32),
          tf.cast(tf.shape(image_)[:2], dtype=tf.float32),
          tf.ones_like(original_dims, dtype = tf.float32),
          -tf.cast(pad[:2], tf.float32)
      ])
   
    image_ = tf.image.resize(image_, (desired_size[0], desired_size[1]))
    infos = [crop_info, pad_info]

    if cut is not None:
      image_, crop_info = mosaic_cut(image_, ow, oh, w, h, cut, 
                                    ptop, pleft, pbottom, pright, 
                                    shiftx, shifty)
      infos.append(crop_info)
    
    return image_, infos 


# ops for box clipping and cleaning
def boxes_candidates(clipped_boxes,  
                     box_history, 
                     wh_thr=2, 
                     ar_thr=20, 
                     area_thr=0.0):
  
  # area thesh can be negative if darknet clipping is used 
  # area_thresh < 0.0 = darknet clipping 
  # area_thresh >= 0.0 = scaled model clipping
  area_thr = tf.math.abs(area_thr)
  
  # get the scaled and shifted heights of the original 
  # unclipped boxes 
  og_height = box_history[:, 2] - box_history[:, 0]
  og_width = box_history[:, 3] - box_history[:, 1]
  
  # get the scaled and shifted heights of the clipped boxes 
  clipped_height = clipped_boxes[:, 2] - clipped_boxes[:, 0]
  clipped_width = clipped_boxes[:, 3] - clipped_boxes[:, 1]

  # determine the aspect ratio of the clipped boxes
  ar = tf.maximum(clipped_width/(clipped_height + 1e-16), 
                  clipped_height/(clipped_width + 1e-16))
  
  # ensure the clipped width adn height are larger than a preset threshold 
  conda = clipped_width > wh_thr
  condb = clipped_height > wh_thr

  # ensure the area of the clipped box is larger than the area threshold 
  condc = ((clipped_height * clipped_width)/(og_width * og_height + 1e-16)) > area_thr
  
  # ensure the aspect ratio is not too extreme
  condd = ar < ar_thr
  
  cond = tf.expand_dims(tf.logical_and(
    tf.logical_and(conda, condb), 
    tf.logical_and(condc, condd)
  ), axis = -1)
  
  # set all the boxes that fail the test to be equal to zero 
  boxes = tf.where(cond, clipped_boxes, tf.zeros_like(clipped_boxes))
  return boxes

def resize_and_crop_boxes(boxes,
                          image_scale,
                          output_size,
                          offset,
                          box_history = None):
  # shift and scale the input boxes
  boxes *= tf.tile(tf.expand_dims(image_scale, axis=0), [1, 2])
  boxes -= tf.tile(tf.expand_dims(offset, axis=0), [1, 2])
  
  # check the hitory of the boxes
  if box_history is None:
    box_history = boxes
  else:
    box_history *= tf.tile(tf.expand_dims(image_scale, axis=0), [1, 2])
    box_history -= tf.tile(tf.expand_dims(offset, axis=0), [1, 2])
  
  # clip the shifted and scaled boxes 
  clipped_boxes = bbox_ops.clip_boxes(boxes, output_size)
  return clipped_boxes, boxes, box_history

def apply_infos(boxes, infos, shuffle_boxes = True, area_thresh = 0.1):
  # clip and clean boxes
  def get_valid_boxes(boxes, unclipped_boxes = None):
    """Get indices for non-empty boxes."""
    # convert the boxes to center width height formatting 
    boxes = box_ops.yxyx_to_xcycwh(boxes)
    (_, _, width, height) = (boxes[..., 0], 
                             boxes[..., 1], 
                             boxes[..., 2], 
                             boxes[..., 3])
    
    # if the width or height is less than zero the box is removed
    base = tf.logical_and(tf.greater(height, 0), tf.greater(width, 0))
    
    if area_thresh < 0.0 and unclipped_boxes is not None:
      # if the area theshold is lower than 0.0 we clip boxes 
      # using the original darknet method. if the center of the 
      # box is not in the frame, it gets removed. this filtering 
      # must be done on boxes prior to box clipping. Clipping the
      # box enusres that it is in the frame by moving the center
      unclipped_boxes = box_ops.yxyx_to_xcycwh(unclipped_boxes)
      x = unclipped_boxes[..., 0]
      y = unclipped_boxes[..., 1]
      
      # check that the center of the unclipped box is within the 
      # frame. 
      condx = tf.logical_and(x >= 0, x < 1)
      condy = tf.logical_and(y >= 0, y < 1)
      base = tf.logical_and(base, tf.logical_and(condx, condy))
    return base 
  
  # this is no history to start
  box_history = None
  
  # make sure all boxes are valid to start, clip to [0, 1]
  # then check all boxes meet the conditions
  boxes = tf.math.maximum(tf.math.minimum(boxes, 1.0), 0.0)
  cond = get_valid_boxes(boxes)
  
  for info in infos:
    # rescale the boxes
    boxes = bbox_ops.denormalize_boxes(boxes, info[0, :])
    if box_history is not None:
      box_history = bbox_ops.denormalize_boxes(box_history, info[0, :])
     
    # shift and scale all boxes, and keep track of box history with no 
    # box clipping, history is used for clipping boxes that becom too small
    # or exit the image
    (boxes,           # clipped final boxes 
     unclipped_boxes, # unclipped final boxes 
     box_history) = resize_and_crop_boxes(boxes, 
                                          info[2, :],
                                          info[1, :], info[3, :], 
                                          box_history = box_history)
    
    # normalize the boxes to [0, 1] 
    boxes = bbox_ops.normalize_boxes(boxes, info[1, :])
    unclipped_boxes = bbox_ops.normalize_boxes(unclipped_boxes, info[1, :])
    box_history = bbox_ops.normalize_boxes(box_history, info[1, :])
    
    # get all the boxes that still remain in the image and store
    # in a bit vector for later use
    cond = tf.logical_and(
              get_valid_boxes(boxes, 
                              unclipped_boxes = unclipped_boxes), cond)
    
    # delete the unclipped boxes, they do not need to be tracked
    del unclipped_boxes

  # remove the bad boxes
  boxes *= tf.cast(tf.expand_dims(cond, axis = -1), boxes.dtype)
  
  # threshold the existing boxes
  boxes = bbox_ops.denormalize_boxes(boxes, info[1, :])
  box_history = bbox_ops.denormalize_boxes(box_history, info[1, :])
  boxes = boxes_candidates(boxes, box_history, area_thr = area_thresh)
  boxes = bbox_ops.normalize_boxes(boxes, info[1, :])
  
  # select and gather the good boxes 
  inds = bbox_ops.get_non_empty_box_indices(boxes)
  if shuffle_boxes:
    inds = tf.random.shuffle(inds)
  boxes = tf.gather(boxes, inds)
  return boxes, inds 


# write the boxes to the anchor grid
def _get_num_reps(anchors, mask, box_mask):
  """
  Calculate the number of anchor boxes that an object is repeated in.
  
  Args:
    anchors: A list or Tensor representing the anchor boxes
    mask: A `list` of `int`s representing the mask for the anchor boxes that
      will be considered when creating the gridded ground truth.
    box_mask: A mask for all of the boxes that are in the bounds of the image
      and have positive height and width.
  
  Returns:
    A tuple with three elements: the number of repetitions, the box indices
    where the primary anchors are usable according to the mask, and the box
    indices where the alternate anchors are usable according to the mask.
  """
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
  """
  Generate a mask to filter the boxes whose x and y coordinates are in
  [0.0, 1.0) and have width and height are greater than 0.
  
  Args:
    boxes: An tensor representing the bounding boxes in the (x, y, w, h) format
      whose last dimension is of size 4.
  
  Returns:
    A mask for all of the boxes that are in the bounds of the image and have
    positive height and width.
  """
  eq0 = tf.reduce_all(tf.math.less_equal(boxes[..., 2:4], 0), axis=-1)
  gtlb = tf.reduce_any(tf.math.less(boxes[..., 0:2], 0.0), axis=-1)
  ltub = tf.reduce_any(tf.math.greater_equal(boxes[..., 0:2], 1.0), axis=-1)

  a = tf.logical_or(eq0, gtlb)
  b = tf.logical_or(a, ltub)
  return tf.logical_not(b)


def _gen_offsets(scale_xy, dtype):
  """
  Create offset Tensor for a `scale_xy` for use in `build_grided_gt_ind`.
  
  Args:
    scale_xy: A `float` to represent the amount the boxes are scaled in the
      loss function.
    dtype: The type of the Tensor to create.
  
  Returns:
    Returns the Tensor for the offsets.
  """
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
    scale_xy: A `float` to represent the amount the boxes are scaled in the
      loss function.
    scale_num_inst: A `float` to represent the scale at which to multiply the
      number of predicted boxes by to get the number of instances to write
      to the grid.tf.print(self._mosaic_crop_mode is None or self._mosaic_crop_mode == "crop") the tie breaker.

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
  len_masks = len(mask)  #mask is a python object tf.shape(mask)[0]
  # number of anchors
  num_anchors = tf.shape(anchors)[-1]
  num_instances = num_boxes * scale_num_inst

  pull_in = _gen_offsets(scale_xy, boxes.dtype)
  # x + 0.5
  # x - 0.5
  # y + 0.5
  # y - 0.5

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
  """
  Write out the y and x grid cell location and the anchor into the `ind_val`
  TensorArray and write the `sample` to the `ind_sample` TensorArray.
  
  Args:
    box: An `int` Tensor with two elements for the relative position of the
      box in the image.
    anchor_id: An `int` representing the anchor box.
    offset: A `float` representing the offset from the grid cell.
    sample: A `float` Tensor representing the data to write out.
    ind_val: A TensorArray representing the indices for the detected boxes.
    ind_sample: A TensorArray representing the indices for the detected boxes.
    height: An `int` representing the height of the image.
    width: An `int` representing the width of the image.
    num_written: A `int` representing the number of samples that have been
      written.
  
  Returns:
    Modified `ind_val`, `ind_sample`, and `num_written`.
  """

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
  """
  Iterate through all viable anchor boxes and write the sample information to
  the `ind_val` and `ind_sample` TensorArrays.
  
  Args:
    num_reps: An `int` Tensor representing the number of repetitions.
    boxes: A 2D `int` Tensor with two elements for the relative position of the
      box in the image for each object.
    classes: An `int` Tensor representing the class for each object.
    ious: A 2D `int` Tensor representing the IOU of object with each anchor box.
    ind_val: A TensorArray representing the indices for the detected boxes.
    ind_sample: A TensorArray representing the indices for the detected boxes.
    height: An `int` representing the height of the image.
    width: An `int` representing the width of the image.
    num_written: A `int` representing the number of samples that have been
      written.
    num_instances: A `int` representing the number of instances that can be
      written at the maximum.
    offset: A `float` representing the offset from the grid cell.
  
  Returns:
    Modified `ind_val`, `ind_sample`, and `num_written`.
  """

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

def get_best_anchor(y_true, anchors, width=1, height=1, iou_thresh=0.25):
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

    true_wh = y_true[..., 2:4]
    anchors = tf.convert_to_tensor(anchors, dtype=y_true.dtype)
    anchors_x = anchors[..., 0] / height
    anchors_y = anchors[..., 1] / width
    anchors = tf.stack([anchors_x, anchors_y], axis=-1)
    k = tf.shape(anchors)[0]

    anchors = tf.expand_dims(
      tf.concat([tf.zeros_like(anchors), anchors], axis = -1), axis = 0)
    truth_comp = tf.concat([tf.zeros_like(true_wh), true_wh], axis = -1)

    if iou_thresh >= 1.0:
      anchors = tf.expand_dims(anchors, axis = -2)
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
        anchors, 
        truth_comp, 
        )
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

    if not is_batch:
      iou_index = tf.squeeze(iou_index, axis=0)
      values = tf.squeeze(values, axis=0)
  return tf.cast(iou_index, dtype=tf.float32), tf.cast(values, dtype=tf.float32)
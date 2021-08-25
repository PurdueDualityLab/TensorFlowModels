import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras.backend as K
from yolo.ops import box_ops
from official.vision.beta.ops import box_ops as bbox_ops
import numpy as np

PAD_VALUE = 114


def rand_uniform_strong(minval, maxval, dtype=tf.float32, seed=None):
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

  return tf.random.uniform([],
                           minval=minval,
                           maxval=maxval,
                           seed=seed,
                           dtype=dtype)


def rand_scale(val, dtype=tf.float32, seed=None):
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
  do_ret = tf.random.uniform([], minval=0, maxval=2, dtype=tf.int32, seed=seed)
  if (do_ret == 1):
    return scale
  return 1.0 / scale


def pad_max_instances(value, instances, pad_value=0, pad_axis=0):
  """
  Pad a dimension of the tensor to have a maximum number of instances filling
  additional entries with the `pad_value`. Allows for selection of the padding 
  axis
   
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
    image: A tensor who has either 3 or 4 dimensions.
  
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


def _augment_hsv_darknet(image, rh, rs, rv, seed=None):
  """
  Randomly alter the hue, saturation, and brightness of an image. 

  Args: 
    image: Tensor of shape [None, None, 3] that needs to be altered.
    rh: `float32` used to indicate the maximum delta that can be added to hue.
    rs: `float32` used to indicate the maximum delta that can be multiplied to 
      saturation.
    rv: `float32` used to indicate the maximum delta that can be multiplied to 
      brightness.
    seed: `Optional[int]` for the seed to use in random number generation.
  
  Returns:
    The HSV altered image in the same datatype as the input image
  """
  if rh > 0.0:
    delta = rand_uniform_strong(-rh, rh, seed=seed)
    image = tf.image.adjust_hue(image, delta)
  if rs > 0.0:
    delta = rand_scale(rs, seed=seed)
    image = tf.image.adjust_saturation(image, delta)
  if rv > 0.0:
    delta = rand_scale(rv, seed=seed)
    image *= delta

  # clip the values of the image between 0.0 and 1.0
  image = tf.clip_by_value(image, 0.0, 1.0)
  return image


def _augment_hsv_torch(image, rh, rs, rv, seed=None):
  """
  Randomly alter the hue, saturation, and brightness of an image. 

  Args: 
    image: Tensor of shape [None, None, 3] that needs to be altered.
    rh: `float32` used to indicate the maximum delta that can be  multiplied to 
      hue.
    rs: `float32` used to indicate the maximum delta that can be multiplied to 
      saturation.
    rv: `float32` used to indicate the maximum delta that can be multiplied to 
      brightness.
    seed: `Optional[int]` for the seed to use in random number generation.
  
  Returns:
    The HSV altered image in the same datatype as the input image
  """
  dtype = image.dtype
  image = tf.cast(image, tf.float32)
  image = tf.image.rgb_to_hsv(image)
  gen_range = tf.cast([rh, rs, rv], image.dtype)
  r = tf.random.uniform([3], -1, 1, 
                        dtype=image.dtype, 
                        seed = seed) * gen_range + 1

  image = tf.cast(image, r.dtype) * r
  h, s, v = tf.split(image, 3, axis=-1)
  h = h % 1.0
  s = tf.clip_by_value(s, 0.0, 1.0)
  v = tf.clip_by_value(v, 0.0, 1.0)

  image = tf.concat([h, s, v], axis=-1)
  image = tf.image.hsv_to_rgb(image)
  return tf.cast(image, dtype)

# def _augment_hsv_torch(image, rh, rs, rv, seed=None):
#   """
#   Randomly alter the hue, saturation, and brightness of an image. 

#   Args: 
#     image: Tensor of shape [None, None, 3] that needs to be altered.
#     rh: `float32` used to indicate the maximum delta that can be added to hue.
#     rs: `float32` used to indicate the maximum delta that can be multiplied to 
#       saturation.
#     rv: `float32` used to indicate the maximum delta that can be multiplied to 
#       brightness.
#     seed: `Optional[int]` for the seed to use in random number generation.
  
#   Returns:
#     The HSV altered image in the same datatype as the input image
#   """
#   if rh > 0.0:
#     delta = rand_uniform_strong(-rh, rh, seed=seed)
#     image = tf.image.adjust_hue(image, delta)
#   if rs > 0.0:
#     delta = 1 + rand_uniform_strong(-rs, rs, seed=seed)
#     image = tf.image.adjust_saturation(image, delta)
#   if rv > 0.0:
#     delta = 1 + rand_uniform_strong(-rs, rs, seed=seed)
#     image *= delta

#   # clip the values of the image between 0.0 and 1.0
#   image = tf.clip_by_value(image, 0.0, 1.0)
#   return image

def image_rand_hsv(image, rh, rs, rv, seed=None, darknet=False):
  """
  Randomly alter the hue, saturation, and brightness of an image. 

  Args: 
    image: Tensor of shape [None, None, 3] that needs to be altered.
    rh: `float32` used to indicate the maximum delta that can be  multiplied to 
      hue.
    rs: `float32` used to indicate the maximum delta that can be multiplied to 
      saturation.
    rv: `float32` used to indicate the maximum delta that can be multiplied to 
      brightness.
    seed: `Optional[int]` for the seed to use in random number generation.
    darknet: `bool` indicating wether the model was orignally built in the 
      darknet or the pytorch library.
  
  Returns:
    The HSV altered image in the same datatype as the input image
  """
  if darknet:
    image = _augment_hsv_darknet(image, rh, rs, rv, seed = seed)
  else:
    image = _augment_hsv_torch(image, rh, rs, rv, seed=seed)
  return image

def random_window_crop(image, target_height, target_width, translate=0.0):
  """Takes a random crop of the image to the target height and width
  
  Args: 
    image: Tensor of shape [None, None, 3] that needs to be altered.
    target_height: `int` indicating the espected output height of the image.
    target_width: `int` indicating the espected output width of the image.
    translate: `float` indicating the maximum delta at which you can take a 
      random crop relative to the center of the image. 
  
  Returns:
    image: The cropped image in the same datatype as the input image.
    info: `float` tensor that is applied to the boxes in order to select the 
      boxes still contained within the image.
  """
  ishape = tf.shape(image)
  th = target_height if target_height < ishape[0] else ishape[0]
  tw = target_width if target_width < ishape[1] else ishape[1]
  crop_size = tf.convert_to_tensor([th, tw, -1])

  crop_offset = ishape - crop_size
  crop_offset = tf.convert_to_tensor(
      [crop_offset[0] // 2, crop_offset[1] // 2, 0])
  shift = tf.convert_to_tensor([
      rand_uniform_strong(-translate, translate),
      rand_uniform_strong(-translate, translate), 0
  ])
  crop_offset = crop_offset + tf.cast(shift * tf.cast(crop_offset, shift.dtype),
                                      crop_offset.dtype)
  cropped_image = tf.slice(image, crop_offset, crop_size)

  scale = tf.cast(ishape[:2] / ishape[:2], tf.float32)
  offset = tf.cast(crop_offset[:2], tf.float32)

  info = tf.stack([
      tf.cast(ishape[:2], tf.float32),
      tf.cast(crop_size[:2], tf.float32), scale, offset
  ],
                  axis=0)
  return cropped_image, info


def mosaic_cut(image, ow, oh, w, h, center, ptop, pleft, pbottom, pright, 
               shiftx, shifty):
  """Given a center location, cut the input image into a slice that will be 
  concatnated with other slices with the same center in order to construct 
  a final mosaiced image. 

  
  Args: 
    image: Tensor of shape [None, None, 3] that needs to be altered.
    ow: `float` value indicating the orignal width of the image.
    oh: `float` value indicating the orignal height of the image.
    w: `float` value indicating the final width image.
    h: `float` value indicating the final height image.
    center: `float` value indicating the desired center of the final patched 
      image.
    ptop: `float` value indicating the top of the image without padding.
    pleft: `float` value indicating the left of the image without padding. 
    pbottom: `float` value indicating the bottom of the image without padding. 
    pright: `float` value indicating the right of the image without padding. 
    shiftx: `float` 0.0 or 1.0 value indicating if the image is in the 
      left or right.
    shifty: `float` 0.0 or 1.0 value indicating if the image is in the 
      top or bottom.
  
  Returns:
    image: The cropped image in the same datatype as the input image.
    crop_info: `float` tensor that is applied to the boxes in order to select 
      the boxes still contained within the image.
  """
  with tf.name_scope('mosaic_cut'):
    center = tf.cast(center, w.dtype)
    cut_x, cut_y = center[1], center[0]

    # Select the crop of the image to use
    left_shift = tf.minimum(
        tf.minimum(cut_x, tf.maximum(0.0, -pleft * w / ow)), w - cut_x)
    top_shift = tf.minimum(
        tf.minimum(cut_y, tf.maximum(0.0, -ptop * h / oh)), h - cut_y)
    right_shift = tf.minimum(
        tf.minimum(w - cut_x, tf.maximum(0.0, -pright * w / ow)), cut_x)
    bot_shift = tf.minimum(
        tf.minimum(h - cut_y, tf.maximum(0.0, -pbottom * h / oh)), cut_y)

    # Build a crop offset and a crop size tensor to use for slicing. 
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
      crop_offset = [0, 0, 0]
      crop_size = [-1, -1, -1]

    # Contain and crop the image.
    ishape = tf.cast(tf.shape(image)[:2], crop_size[0].dtype)
    crop_size[0] = tf.minimum(crop_size[0], ishape[0])
    crop_size[1] = tf.minimum(crop_size[1], ishape[1])

    crop_offset = tf.cast(crop_offset, tf.int32)
    crop_size = tf.cast(crop_size, tf.int32)

    image = tf.slice(image, crop_offset, crop_size)
    crop_info = tf.stack([
        tf.cast(ishape, tf.float32),
        tf.cast(tf.shape(image)[:2], dtype=tf.float32),
        tf.ones_like(ishape, dtype=tf.float32),
        tf.cast(crop_offset[:2], tf.float32)
    ])

  return image, crop_info


def resize_and_jitter_image(image,
                            desired_size,
                            jitter=0.0,
                            letter_box=None,
                            resize=1.0,
                            random_pad=True,
                            crop_only=False,
                            shiftx=0.5,
                            shifty=0.5,
                            cut=None,
                            method=tf.image.ResizeMethod.BILINEAR,
                            seed=None):
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
  def intersection(a, b):
    minx = tf.maximum(a[0], b[0])
    miny = tf.maximum(a[1], b[1])
    maxx = tf.minimum(a[2], b[2])
    maxy = tf.minimum(a[3], b[3])
    return tf.convert_to_tensor([minx, miny, maxx, maxy])

  def cast(values, dtype):
    return [tf.cast(value, dtype) for value in values]

  if jitter > 0.5 or jitter < 0:
    raise Exception("maximum change in aspect ratio must be between 0 and 0.5")

  with tf.name_scope('resize_and_jitter_image'):
    # Cast all parameters to a usable float data type.
    jitter = tf.cast(jitter, tf.float32)
    original_dtype, original_dims = image.dtype, tf.shape(image)[:2]
    ow, oh, w, h = cast(
        [original_dims[1], original_dims[0], desired_size[1], desired_size[0]],
        tf.float32)

    # Compute the random delta width and height etc. and randomize the
    # location of the corner points.
    dw = ow * jitter
    dh = oh * jitter
    pleft = rand_uniform_strong(-dw, dw, dw.dtype, seed=seed)
    pright = rand_uniform_strong(-dw, dw, dw.dtype, seed=seed)
    ptop = rand_uniform_strong(-dh, dh, dh.dtype, seed=seed)
    pbottom = rand_uniform_strong(-dh, dh, dh.dtype, seed=seed)

    # Bounded resizing of the images.
    if resize != 1:
      max_rdw, max_rdh = 0.0, 0.0
      if resize > 1.0:
        max_rdw = ow * (1 - (1 / resize)) / 2
        max_rdh = oh * (1 - (1 / resize)) / 2

      resize_down = resize if resize < 1.0 else 1 / resize
      min_rdw = ow * (1 - (1 / resize_down)) / 2
      min_rdh = oh * (1 - (1 / resize_down)) / 2

      pleft += rand_uniform_strong(min_rdw, max_rdw, seed=seed)
      pright += rand_uniform_strong(min_rdw, max_rdw, seed=seed)
      ptop += rand_uniform_strong(min_rdh, max_rdh, seed=seed)
      pbottom += rand_uniform_strong(min_rdh, max_rdh, seed=seed)

    # Letter box the image.
    if letter_box == True or letter_box is None:
      image_aspect_ratio, input_aspect_ratio = ow / oh, w / h
      distorted_aspect = image_aspect_ratio / input_aspect_ratio

      delta_h, delta_w = 0.0, 0.0
      pullin_h, pullin_w = 0.0, 0.0
      if distorted_aspect > 1:
        delta_h = ((ow / input_aspect_ratio) - oh) / 2
      else:
        delta_w = ((oh * input_aspect_ratio) - ow) / 2

      if letter_box is None:
        rwidth = ow + delta_w + delta_w
        rheight = oh + delta_h + delta_h
        if rheight < h and rwidth < w:
          pullin_h = ((h - rheight) * rheight / h) / 2
          pullin_w = ((w - rwidth) * rwidth / w) / 2

      ptop = ptop - delta_h - pullin_h
      pbottom = pbottom - delta_h - pullin_h
      pright = pright - delta_w - pullin_w
      pleft = pleft - delta_w - pullin_w

    # Compute the width and height to crop or pad too, and clip all crops to
    # to be contained within the image.
    swidth = ow - pleft - pright
    sheight = oh - ptop - pbottom
    src_crop = intersection([ptop, pleft, sheight + ptop, swidth + pleft],
                            [0, 0, oh, ow])

    # Random padding used for mosaic.
    h_ = src_crop[2] - src_crop[0]
    w_ = src_crop[3] - src_crop[1]
    if random_pad:
      rmh = tf.maximum(0.0, -ptop)
      rmw = tf.maximum(0.0, -pleft)
    else:
      rmw = (swidth - w_) * shiftx
      rmh = (sheight - h_) * shifty

    # Cast cropping params to usable dtype.
    src_crop = tf.cast(src_crop, tf.int32)

    # Compute padding parmeters.
    dst_shape = [rmh, rmw, rmh + h_, rmw + w_]
    ptop, pleft, pbottom, pright = dst_shape
    pad = dst_shape * tf.cast([1, 1, -1, -1], ptop.dtype)
    pad += tf.cast([0, 0, sheight, swidth], ptop.dtype)
    pad = tf.cast(pad, tf.int32)

    infos = []

    # Crop the image to desired size.
    cropped_image = tf.slice(
        image, [src_crop[0], src_crop[1], 0],
        [src_crop[2] - src_crop[0], src_crop[3] - src_crop[1], -1])
    crop_info = tf.stack([
        tf.cast(original_dims, tf.float32),
        tf.cast(tf.shape(cropped_image)[:2], dtype=tf.float32),
        tf.ones_like(original_dims, dtype=tf.float32),
        tf.cast(src_crop[:2], tf.float32)
    ])
    infos.append(crop_info)

    if crop_only:
      if not letter_box:
        h_, w_ = cast(get_image_shape(cropped_image), w.dtype)
        w = tf.cast(tf.round((w_ * w) / swidth), tf.int32)
        h = tf.cast(tf.round((h_ * h) / sheight), tf.int32)
        cropped_image = tf.image.resize(cropped_image, [h, w], method=method)
        cropped_image = tf.cast(cropped_image, original_dtype)
      return cropped_image, infos, cast(
          [ow, oh, w, h, ptop, pleft, pbottom, pright], tf.int32)

    # Pad the image to desired size.
    image_ = tf.pad(
        cropped_image, [[pad[0], pad[2]], [pad[1], pad[3]], [0, 0]],
        constant_values=PAD_VALUE)
    pad_info = tf.stack([
        tf.cast(tf.shape(cropped_image)[:2], tf.float32),
        tf.cast(tf.shape(image_)[:2], dtype=tf.float32),
        tf.ones_like(original_dims, dtype=tf.float32),
        (-tf.cast(pad[:2], tf.float32))
    ])
    infos.append(pad_info)

    temp = tf.shape(image_)[:2]
    cond = temp > tf.cast(desired_size, temp.dtype) 
    if tf.reduce_any(cond):
      size = tf.cast(desired_size, temp.dtype)
      size = tf.where(cond, size, temp)
      image_ = tf.image.resize(
          image_, (size[0], size[1]),
          method=tf.image.ResizeMethod.AREA)
      image_ = tf.cast(image_, original_dtype)

    image_ = tf.image.resize(
          image_, (desired_size[0], desired_size[1]),
          method=tf.image.ResizeMethod.BILINEAR, 
          antialias=False)

    image_ = tf.cast(image_, original_dtype)
    if cut is not None:
      image_, crop_info = mosaic_cut(image_, ow, oh, w, h, cut, ptop, pleft,
                                     pbottom, pright, shiftx, shifty)
      infos.append(crop_info)
    return image_, infos, cast([ow, oh, w, h, ptop, pleft, pbottom, pright],
                               tf.int32)

def build_transform(image,
                    perspective=0.00,
                    degrees=0.0,
                    scale_min=1.0,
                    scale_max=1.0,
                    translate=0.0,
                    random_pad=False,
                    desired_size=None,
                    seed=None):

  height, width = get_image_shape(image)
  ch = height = tf.cast(height, tf.float32)
  cw = width = tf.cast(width, tf.float32)
  deg_to_rad = lambda x: tf.cast(x, tf.float32) * np.pi / 180.0

  if desired_size is not None:
    desired_size = tf.cast(desired_size, tf.float32)
    ch = desired_size[0]
    cw = desired_size[1]

  # Compute the center of the image in the output resulution.
  C = tf.eye(3, dtype=tf.float32)
  C = tf.tensor_scatter_nd_update(C, [[0, 2], [1, 2]], [-cw / 2, -ch / 2])
  Cb = tf.tensor_scatter_nd_update(C, [[0, 2], [1, 2]], [cw / 2, ch / 2])

  # Compute a random rotation to apply. 
  R = tf.eye(3, dtype=tf.float32)
  a = deg_to_rad(tf.random.uniform([], -degrees, degrees, seed=seed))
  cos = tf.math.cos(a)
  sin = tf.math.sin(a)
  R = tf.tensor_scatter_nd_update(R, [[0, 0], [0, 1], [1, 0], [1, 1]],
                                  [cos, -sin, sin, cos])
  Rb = tf.tensor_scatter_nd_update(R, [[0, 0], [0, 1], [1, 0], [1, 1]],
                                   [cos, sin, -sin, cos])

  # Compute a random prespective change to apply. 
  P = tf.eye(3)
  Px = tf.random.uniform([], -perspective, perspective, seed=seed)
  Py = tf.random.uniform([], -perspective, perspective, seed=seed)
  P = tf.tensor_scatter_nd_update(P, [[2, 0], [2, 1]], [Px, Py])
  Pb = tf.tensor_scatter_nd_update(P, [[2, 0], [2, 1]], [-Px, -Py])

  # Compute a random scaling to apply. 
  S = tf.eye(3, dtype=tf.float32)
  s = tf.random.uniform([], scale_min, scale_max, seed=seed)
  S = tf.tensor_scatter_nd_update(S, [[0, 0], [1, 1]], [1 / s, 1 / s])
  Sb = tf.tensor_scatter_nd_update(S, [[0, 0], [1, 1]], [s, s])

  # Compute a random Translation to apply.
  T = tf.eye(3)
  if ((random_pad and s <= 1.0) or
      (random_pad and s > 1.0 and translate < 0.0)):
    # The image is contained within the image and arbitrarily translated to 
    # locations with in the image.
    C = Cb = tf.eye(3, dtype=tf.float32)
    Tx = tf.random.uniform([], -1, 0, seed=seed) * (cw / s - width)
    Ty = tf.random.uniform([], -1, 0, seed=seed) * (ch / s - height)
  else:
    # The image can be translated outside of the output resolution window
    # but the image is translated relative to the output resolution not the 
    # input image resolution. 
    Tx = tf.random.uniform([], 0.5 - translate, 0.5 + translate, seed=seed)
    Ty = tf.random.uniform([], 0.5 - translate, 0.5 + translate, seed=seed)

    # Center and Scale the image such that the window of translation is 
    # contained to the output resolution. 
    dx, dy = (width - cw / s) / width, (height - ch / s) / height
    sx, sy = 1 - dx, 1 - dy
    bx, by = dx / 2, dy / 2
    Tx, Ty = bx + (sx * Tx), by + (sy * Ty)

    # Scale the translation to width and height of the image.
    Tx *= width
    Ty *= height

  T = tf.tensor_scatter_nd_update(T, [[0, 2], [1, 2]], [Tx, Ty])
  Tb = tf.tensor_scatter_nd_update(T, [[0, 2], [1, 2]], [-Tx, -Ty])

  # Use repeated matric multiplications to combine all the image transforamtions 
  # into a single unified augmentation operation M is applied to the image
  # Mb is to apply to the boxes. The order of matrix multiplication is 
  # important. First, Translate, then Scale, then Rotate, then Center, then 
  # finally alter the Prepsective. 
  M = T @ S @ R @ C @ P
  Mb = Pb @ Cb @ Rb @ Sb @ Tb
  return M, Mb, s


def affine_warp_image(image,
                      desired_size,
                      perspective=0.00,
                      degrees=0.0,
                      scale_min=1.0,
                      scale_max=1.0,
                      translate=0.0,
                      random_pad=False,
                      seed=None):
  
  # Build an image transformation matrix.
  image_size = tf.cast(get_image_shape(image), tf.float32)
  M_, Mb, _ = build_transform(
      image,
      perspective=perspective,
      degrees=degrees,
      scale_min=scale_min,
      scale_max=scale_max,
      translate=translate,
      random_pad=random_pad,
      desired_size=desired_size,
      seed=seed)
  M = tf.reshape(M_, [-1])
  M = tf.cast(M[:-1], tf.float32)

  # Apply the transformation to image.
  image = tfa.image.transform(
      image,
      M,
      fill_value=PAD_VALUE,
      output_shape=desired_size,
      interpolation='bilinear')

  desired_size = tf.cast(desired_size, tf.float32)
  return image, M_, [image_size, desired_size, Mb]

# ops for box clipping and cleaning
def affine_warp_boxes(Mb, boxes, output_size, box_history=None):
  def _get_corners(box):
    """Get the corner of each box as a tuple of (x, y) coordinates"""
    ymi, xmi, yma, xma = tf.split(box, 4, axis=-1)
    tl = tf.concat([xmi, ymi], axis=-1)
    bl = tf.concat([xmi, yma], axis=-1)
    tr = tf.concat([xma, ymi], axis=-1)
    br = tf.concat([xma, yma], axis=-1)
    return tf.concat([tl, bl, tr, br], axis=-1)

  def _corners_to_boxes(corner):
    """Convert (x, y) corner tuples back into boxes in the format
    [ymin, xmin, ymax, xmax]"""
    corner = tf.reshape(corner, [-1, 4, 2])
    y = corner[..., 1]
    x = corner[..., 0]
    y_min = tf.reduce_min(y, axis=-1)
    x_min = tf.reduce_min(x, axis=-1)
    y_max = tf.reduce_max(y, axis=-1)
    x_max = tf.reduce_max(x, axis=-1)
    return tf.stack([y_min, x_min, y_max, x_max], axis=-1)

  def _aug_boxes(M, box):
    """Apply an affine transformation matrix M to the boxes to get the 
    randomly augmented boxes"""
    corners = _get_corners(box)
    corners = tf.reshape(corners, [-1, 4, 2])
    z = tf.expand_dims(tf.ones_like(corners[..., 1]), axis=-1)
    corners = tf.concat([corners, z], axis=-1)

    corners = tf.transpose(
        tf.matmul(M, corners, transpose_b=True), perm=(0, 2, 1))

    corners, p = tf.split(corners, [2, 1], axis=-1)
    corners /= p
    corners = tf.reshape(corners, [-1, 8])
    box = _corners_to_boxes(corners)
    return box

  boxes = _aug_boxes(Mb, boxes)
  if box_history is not None:
    box_history = _aug_boxes(Mb, box_history)
  else:
    box_history = boxes

  clipped_boxes = bbox_ops.clip_boxes(boxes, output_size)
  return clipped_boxes, boxes, box_history

def boxes_candidates(clipped_boxes,
                     box_history,
                     wh_thr=2,
                     ar_thr=20,
                     area_thr=0.0):

  # Area thesh can be negative if darknet clipping is used.
  # Area_thresh < 0.0 = darknet clipping.
  # Area_thresh >= 0.0 = scaled model clipping.
  area_thr = tf.math.abs(area_thr)

  # Get the scaled and shifted heights of the original
  # unclipped boxes.
  og_height = box_history[:, 2] - box_history[:, 0]
  og_width = box_history[:, 3] - box_history[:, 1]

  # Get the scaled and shifted heights of the clipped boxes.
  clipped_height = clipped_boxes[:, 2] - clipped_boxes[:, 0]
  clipped_width = clipped_boxes[:, 3] - clipped_boxes[:, 1]

  # Determine the aspect ratio of the clipped boxes.
  ar = tf.maximum(clipped_width / (clipped_height + 1e-16),
                  clipped_height / (clipped_width + 1e-16))

  # Ensure the clipped width adn height are larger than a preset threshold.
  conda = clipped_width > wh_thr
  condb = clipped_height > wh_thr

  # Ensure the area of the clipped box is larger than the area threshold.
  condc = ((clipped_height * clipped_width) /
           (og_width * og_height + 1e-16)) > area_thr

  # Ensure the aspect ratio is not too extreme.
  condd = ar < ar_thr

  cond = tf.expand_dims(
      tf.logical_and(
          tf.logical_and(conda, condb), tf.logical_and(condc, condd)),
      axis=-1)

  # Set all the boxes that fail the test to be equal to zero.
  boxes = tf.where(cond, clipped_boxes, tf.zeros_like(clipped_boxes))
  return boxes


def resize_and_crop_boxes(boxes,
                          image_scale,
                          output_size,
                          offset,
                          box_history=None):
  # Shift and scale the input boxes.
  boxes *= tf.tile(tf.expand_dims(image_scale, axis=0), [1, 2])
  boxes -= tf.tile(tf.expand_dims(offset, axis=0), [1, 2])

  # Check the hitory of the boxes.
  if box_history is None:
    box_history = boxes
  else:
    box_history *= tf.tile(tf.expand_dims(image_scale, axis=0), [1, 2])
    box_history -= tf.tile(tf.expand_dims(offset, axis=0), [1, 2])

  # Clip the shifted and scaled boxes.
  clipped_boxes = bbox_ops.clip_boxes(boxes, output_size)
  return clipped_boxes, boxes, box_history


def apply_infos(boxes,
                infos,
                affine=None,
                shuffle_boxes=False,
                area_thresh=0.1,
                seed=None):
  # Clip and clean boxes.
  def get_valid_boxes(boxes, unclipped_boxes=None):
    """Get indices for non-empty boxes."""
    # Convert the boxes to center width height formatting.
    boxes = box_ops.yxyx_to_xcycwh(boxes)
    (_, _, width, height) = (boxes[..., 0], boxes[..., 1], boxes[...,
                                                                 2], boxes[...,
                                                                           3])

    # If the width or height is less than zero the box is removed.
    base = tf.logical_and(tf.greater(height, 0), tf.greater(width, 0))

    if area_thresh < 0.0 and unclipped_boxes is not None:
      # If the area theshold is lower than 0.0 we clip boxes
      # using the original darknet method. if the center of the
      # box is not in the frame, it gets removed. this filtering
      # must be done on boxes prior to box clipping. Clipping the
      # box enusres that it is in the frame by moving the center.
      unclipped_boxes = box_ops.yxyx_to_xcycwh(unclipped_boxes)
      x = unclipped_boxes[..., 0]
      y = unclipped_boxes[..., 1]

      # check that the center of the unclipped box is within the
      # frame.
      condx = tf.logical_and(x >= 0, x < 1)
      condy = tf.logical_and(y >= 0, y < 1)
      base = tf.logical_and(base, tf.logical_and(condx, condy))
    return base

  output_size = tf.cast([512, 512], tf.float32)
  if infos is None:
    infos = []

  # Initialize history to None to indicate no operation have been applied to 
  # the boxes yet.
  box_history = None

  # Make sure all boxes are valid to start, clip to [0, 1] and get only the 
  # valid boxes.
  boxes = tf.math.maximum(tf.math.minimum(boxes, 1.0), 0.0)
  cond = get_valid_boxes(boxes)

  for info in infos:
    # Denormalize the boxes.
    boxes = bbox_ops.denormalize_boxes(boxes, info[0])
    if box_history is not None:
      box_history = bbox_ops.denormalize_boxes(box_history, info[0])

    # Shift and scale all boxes, and keep track of box history with no
    # box clipping, history is used for removing boxes that have become 
    # too small or exit the image area.
    (boxes,  # Clipped final boxes. 
     unclipped_boxes,  # Unclipped final boxes. 
     box_history) = resize_and_crop_boxes(
            boxes, info[2, :], info[1, :], info[3, :], box_history=box_history)

    # Normalize the boxes to [0, 1].
    boxes = bbox_ops.normalize_boxes(boxes, info[1])
    unclipped_boxes = bbox_ops.normalize_boxes(unclipped_boxes, info[1])
    box_history = bbox_ops.normalize_boxes(box_history, info[1])

    # Get all the boxes that still remain in the image and store
    # in a bit vector for later use.
    cond = tf.logical_and(
        get_valid_boxes(boxes, unclipped_boxes=unclipped_boxes), cond)

    # Delete the unclipped boxes, they do not need to be tracked.
    del unclipped_boxes
    output_size = info[1]

  if affine is not None:
    # Denormalize the boxes.
    boxes = bbox_ops.denormalize_boxes(boxes, affine[0])
    if box_history is not None:
      # Denormalize the box history.
      box_history = bbox_ops.denormalize_boxes(box_history, affine[0])

      (boxes,  # Clipped final boxes. 
      unclipped_boxes,  # Unclipped final boxes. 
      box_history) = affine_warp_boxes(
            affine[2], boxes, affine[1], box_history=box_history)

    # Normalize the boxes to [0, 1].
    boxes = bbox_ops.normalize_boxes(boxes, affine[1])
    unclipped_boxes = bbox_ops.normalize_boxes(unclipped_boxes, affine[1])
    box_history = bbox_ops.normalize_boxes(box_history, affine[1])

    # Get all the boxes that still remain in the image and store
    # in a bit vector for later use.
    cond = tf.logical_and(
        get_valid_boxes(boxes, unclipped_boxes=unclipped_boxes), cond)

    # Delete the unclipped boxes, they do not need to be tracked.
    del unclipped_boxes
    output_size = affine[1]

  # Remove the bad boxes.
  boxes *= tf.cast(tf.expand_dims(cond, axis=-1), boxes.dtype)

  # Threshold the existing boxes.
  boxes = bbox_ops.denormalize_boxes(boxes, output_size)
  box_history = bbox_ops.denormalize_boxes(box_history, output_size)
  boxes = boxes_candidates(boxes, box_history, area_thr=area_thresh)
  boxes = bbox_ops.normalize_boxes(boxes, output_size)

  # Select and gather the good boxes.
  inds = bbox_ops.get_non_empty_box_indices(boxes)
  if shuffle_boxes:
    inds = tf.random.shuffle(inds, seed=seed)
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


def build_grided_gt_ind(y_true, mask, sizew, sizeh, num_classes, dtype,
                        scale_xy, scale_num_inst, use_tie_breaker):
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

  width = tf.cast(sizew, boxes.dtype)
  height = tf.cast(sizeh, boxes.dtype)
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
  ind_val = tf.TensorArray(
      tf.int32, size=0, dynamic_size=True, element_shape=[
          3,
      ])
  ind_sample = tf.TensorArray(
      dtype, size=0, dynamic_size=True, element_shape=[
          8,
      ])

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

  full = tf.zeros([sizeh, sizew, len_masks, 1], dtype=dtype)
  full = tf.tensor_scatter_nd_add(full, indexs, ind_mask)

  if num_written >= num_instances:
    tf.print("clipped")

  indexs = pad_max_instances(indexs, num_instances, pad_value=0, pad_axis=0)
  samples = pad_max_instances(samples, num_instances, pad_value=0, pad_axis=0)
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
    g = tf.cast(offset, x.dtype)
    gain = tf.cast(tf.convert_to_tensor([width, height]), x.dtype)
    gxy = tf.cast(tf.convert_to_tensor([x, y]), x.dtype)
    clamp = lambda x, ma: tf.maximum(
        tf.minimum(x, tf.cast(ma, x.dtype)), tf.zeros_like(x))

    gxyi = gxy - tf.floor(gxy)
    ps = ((gxyi < g) & (gxy > 1.))
    ns = ((gxyi > (1 - g)) & (gxy < (gain - 1.)))

    shifts = [ps[0], ps[1], ns[0], ns[1]]
    offset = tf.cast([[1, 0], [0, 1], [-1, 0], [0, -1]], g.dtype) * g

    xc = clamp(tf.convert_to_tensor([tf.cast(x, tf.int32)]), width - 1)
    yc = clamp(tf.convert_to_tensor([tf.cast(y, tf.int32)]), height - 1)
    for i in range(4):
      x_ = x - offset[i, 0]
      y_ = y - offset[i, 1]

      x_ = clamp(tf.convert_to_tensor([tf.cast(x_, tf.int32)]), width - 1)
      y_ = clamp(tf.convert_to_tensor([tf.cast(y_, tf.int32)]), height - 1)
      if shifts[i]:  # and (xc != x_ or yc != y_):
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


def get_best_anchor(y_true,
                    anchors,
                    width=1,
                    height=1,
                    iou_thresh=0.25,
                    best_match_only=False):
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
    anchors = tf.cast(anchors, dtype=tf.float32)
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
    else:
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

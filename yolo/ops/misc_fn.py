# def resize_and_crop_image(image,
#                           desired_size,
#                           padded_size,
#                           letter_box=None,
#                           aug_scale_min=1.0,
#                           aug_scale_max=1.0,
#                           random_pad=False,
#                           shiftx=0.5,
#                           shifty=0.5,
#                           sheer=0.0,
#                           seed=1,
#                           method=tf.image.ResizeMethod.BILINEAR):
#   """Resizes the input image to output size (RetinaNet style).
#   Resize and pad images given the desired output size of the image and
#   stride size.
#   Here are the preprocessing steps.
#   1. For a given image, keep its aspect ratio and rescale the image to make it
#      the largest rectangle to be bounded by the rectangle specified by the
#      `desired_size`.
#   2. Pad the rescaled image to the padded_size.
#   Args:
#     image: a `Tensor` of shape [height, width, 3] representing an image.
#     desired_size: a `Tensor` or `int` list/tuple of two elements representing
#       [height, width] of the desired actual output image size.
#     padded_size: a `Tensor` or `int` list/tuple of two elements representing
#       [height, width] of the padded output image size. Padding will be applied
#       after scaling the image to the desired_size.
#     aug_scale_min: a `float` with range between [0, 1.0] representing minimum
#       random scale applied to desired_size for training scale jittering.
#     aug_scale_max: a `float` with range between [1.0, inf] representing maximum
#       random scale applied to desired_size for training scale jittering.
#     seed: seed for random scale jittering.
#     method: function to resize input image to scaled image.
#   Returns:
#     output_image: `Tensor` of shape [height, width, 3] where [height, width]
#       equals to `output_size`.
#     image_info: a 2D `Tensor` that encodes the information of the image and the
#       applied preprocessing. It is in the format of
#       [[original_height, original_width], [desired_height, desired_width],
#        [y_scale, x_scale], [y_offset, x_offset]], where [desired_height,
#       desired_width] is the actual scaled image size, and [y_scale, x_scale] is
#       the scaling factor, which is the ratio of
#       scaled dimension / original dimension.
#   """
#   with tf.name_scope('resize_and_crop_image'):

#     if letter_box == False:
#       height, width = get_image_shape(image)
#       clipper = tf.reduce_max((height, width))
#       image = image = tf.image.resize(
#           image, (clipper, clipper), preserve_aspect_ratio=False)
#     elif letter_box == True:
#       height, width = get_image_shape(image)
#       clipper = tf.reduce_max((height, width))
#       w_scale = width / clipper
#       h_scale = height / clipper

#       height_, width_ = desired_size[0], desired_size[1]
#       height_ = tf.cast(h_scale * tf.cast(height_, h_scale.dtype), tf.int32)
#       width_ = tf.cast(w_scale * tf.cast(width_, w_scale.dtype), tf.int32)

#       image = image = tf.image.resize(
#         image, (height_, width_), preserve_aspect_ratio=False)

#     if sheer > 0.0:
#       h_, w_ = get_image_shape(image)

#       hr = tf.cast(sheer * tf.cast(h_, tf.float32), tf.int32)
#       wr = tf.cast(sheer * tf.cast(w_, tf.float32), tf.int32)

#       h_ += rand_uniform_strong(-hr, hr + 1, tf.int32)
#       w_ += rand_uniform_strong(-wr, wr + 1, tf.int32)
#       image = tf.image.resize(image, (h_, w_))

#     image_size = tf.cast(tf.shape(image)[0:2], tf.float32)
#     random_jittering = (aug_scale_min != 1.0 or aug_scale_max != 1.0)

#     if random_jittering:
#       random_scale = tf.random.uniform([],
#                                        aug_scale_min,
#                                        aug_scale_max,
#                                        seed=seed)
#       scaled_size = tf.round(random_scale * desired_size)
#       tf.print(scaled_size)
#     else:
#       random_scale = 1.0
#       scaled_size = desired_size

#     scale = tf.minimum(scaled_size[0] / image_size[0],
#                        scaled_size[1] / image_size[1])
#     scaled_size = tf.round(image_size * scale)

#     # Computes 2D image_scale.
#     image_scale = scaled_size / image_size

#     # Selects non-zero random offset (x, y) if scaled image is larger than
#     # desired_size.
#     random_pad = tf.cast(random_pad, tf.float32)
#     if random_pad == 1.0:
#       random_pad = tf.cast(0.5, tf.float32)
#     if random_jittering:
#       max_offset_ = scaled_size - desired_size

#       max_offset = tf.where(
#           tf.less(max_offset_, 0), tf.zeros_like(max_offset_), max_offset_)
#       offset = max_offset * 0.5 + max_offset * tf.random.uniform([2,],
#                                                 -random_pad, random_pad, seed=seed)
#       offset = tf.cast(offset, tf.int32)
#     else:
#       offset = tf.zeros((2,), tf.int32)

#     scaled_image = tf.image.resize(
#         image, tf.cast(scaled_size, tf.int32), method=method)

#     if random_jittering:
#       scaled_image = scaled_image[offset[0]:offset[0] + desired_size[0],
#                                   offset[1]:offset[1] + desired_size[1], :]

#     scaled_size = tf.cast(tf.shape(scaled_image)[0:2], tf.int32)

#     if random_pad > 0.0:
#       shifty = 0.5 + rand_uniform_strong(-random_pad, random_pad, tf.float32)
#       shiftx = 0.5 + rand_uniform_strong(-random_pad, random_pad, tf.float32)

#     dy = tf.cast(
#         tf.cast(padded_size[0] - scaled_size[0], tf.float32) * shifty,
#         tf.int32)
#     dx = tf.cast(
#         tf.cast(padded_size[1] - scaled_size[1], tf.float32) * shiftx,
#         tf.int32)

#     output_image = tf.image.pad_to_bounding_box(scaled_image, dy, dx,
#                                                 padded_size[0], padded_size[1])

#     offset -= tf.convert_to_tensor([dy, dx])

#     image_info = tf.stack([
#         image_size,
#         tf.constant(desired_size, dtype=tf.float32), image_scale,
#         tf.cast(offset, tf.float32)
#     ])

#     infos = [image_info]
#     return output_image, infos

# def mean_pad(image, pady, padx, targety, targetx, color = False):
#   shape = tf.shape(image)[:2]
#   pad = [pady, padx, targety - shape[0] - pady, targetx - shape[1] - padx]

#   if color:
#     r, g, b = tf.split(image, 3, axis = -1)
#     r = tf.pad(r, [[pad[0], pad[2]],
#                   [pad[1], pad[3]],
#                   [0, 0]],
#                   constant_values=tf.reduce_mean(r))
#     g = tf.pad(g, [[pad[0], pad[2]],
#                   [pad[1], pad[3]],
#                   [0, 0]],
#                   constant_values=tf.reduce_mean(g))
#     b = tf.pad(b, [[pad[0], pad[2]],
#                   [pad[1], pad[3]],
#                   [0, 0]],
#                   constant_values=tf.reduce_mean(b))
#     image_ = tf.concat([r, g, b], axis = -1)
#   else:
#     image_ = tf.pad(image, [[pad[0], pad[2]],
#                             [pad[1], pad[3]],
#                             [0, 0]])

#   pad_info = tf.stack([
#         tf.cast(tf.shape(image)[:2], tf.float32),
#         tf.cast(tf.shape(image_)[:2], dtype=tf.float32),
#         tf.ones_like(tf.shape(image)[:2], dtype = tf.float32),
#         -tf.cast(pad[:2], tf.float32)
#     ])
#   return image_, pad_info

# def random_jitter_crop(image, jitter=0.2, seed=1):
#   """Randomly crop an arbitrary shaped slice from the input image.

#   Args:
#     image: a Tensor of shape [height, width, 3] representing the input image.
#     aspect_ratio_range: a list of floats. The cropped area of the image must
#       have an aspect ratio = width / height within this range.
#     area_range: a list of floats. The cropped reas of the image must contain
#       a fraction of the input image within this range.
#     max_attempts: the number of attempts at generating a cropped region of the
#       image of the specified constraints. After max_attempts failures, return
#       the entire image.
#     seed: the seed of the random generator.

#   Returns:
#     cropped_image: a Tensor representing the random cropped image. Can be the
#       original image if max_attempts is exhausted.
#   """

#   with tf.name_scope('random_crop_image'):
#     ishape = tf.shape(image)

#     if jitter > 1 or jitter < 0:
#       raise Exception("maximum change in aspect ratio must be between 0 and 1")

#     original_dims = ishape[:2]
#     jitter = tf.cast(jitter, tf.float32)/2
#     ow = tf.cast(original_dims[1], tf.float32)
#     oh = tf.cast(original_dims[0], tf.float32)

#     dw = ow * jitter
#     dh = oh * jitter

#     pleft = rand_uniform_strong(-dw, dw, dw.dtype)
#     pright = rand_uniform_strong(-dw, dw, dw.dtype)
#     ptop = rand_uniform_strong(-dh, dh, dh.dtype)
#     pbottom = rand_uniform_strong(-dh, dh, dh.dtype)

#     crop_top = tf.convert_to_tensor([pleft, ptop])
#     crop_bottom = tf.convert_to_tensor([ow - pright, oh - pbottom])

#     src_top = tf.zeros_like(crop_top)
#     src_bottom = tf.cast(tf.convert_to_tensor([ow, oh]), src_top.dtype)

#     intersect_top = tf.maximum(crop_top, src_top)
#     intersect_bottom = tf.minimum(crop_bottom, src_bottom)

#     intersect_wh = src_bottom - intersect_top - (src_bottom - intersect_bottom)

#     crop_offset = tf.cast(
#         tf.convert_to_tensor([intersect_top[1], intersect_top[0], 0]), tf.int32)
#     crop_size = tf.cast(
#         tf.convert_to_tensor([intersect_wh[1], intersect_wh[0], -1]), tf.int32)

#     cropped_image = tf.slice(image, crop_offset, crop_size)

#     scale = tf.cast(ishape[:2] / ishape[:2], tf.float32)
#     offset = tf.cast(crop_offset[:2], tf.float32)

#     info = tf.stack([
#         tf.cast(ishape[:2], tf.float32),
#         tf.cast(crop_size[:2], tf.float32), scale, offset
#     ],
#                     axis=0)
#     return cropped_image, info

# def resize_and_jitter_image(image,
#                             desired_size,
#                             jitter=0.0,
#                             resize = 1.0,
#                             letter_box=None,
#                             crop_only = False,

#                             random_pad=True,
#                             shiftx=0.0,
#                             shifty=0.0,
#                             cut = None,
#                             method=tf.image.ResizeMethod.BILINEAR):
#   """Resizes the input image to output size (RetinaNet style).
#   Resize and pad images given the desired output size of the image and
#   stride size.
#   Here are the preprocessing steps.
#   1. For a given image, keep its aspect ratio and rescale the image to make it
#      the largest rectangle to be bounded by the rectangle specified by the
#      `desired_size`.
#   2. Pad the rescaled image to the padded_size.
#   Args:
#     image: a `Tensor` of shape [height, width, 3] representing an image.
#     desired_size: a `Tensor` or `int` list/tuple of two elements representing
#       [height, width] of the desired actual output image size.
#     padded_size: a `Tensor` or `int` list/tuple of two elements representing
#       [height, width] of the padded output image size. Padding will be applied
#       after scaling the image to the desired_size.
#     aug_scale_min: a `float` with range between [0, 1.0] representing minimum
#       random scale applied to desired_size for training scale jittering.
#     aug_scale_max: a `float` with range between [1.0, inf] representing maximum
#       random scale applied to desired_size for training scale jittering.
#     seed: seed for random scale jittering.
#     method: function to resize input image to scaled image.
#   Returns:
#     output_image: `Tensor` of shape [height, width, 3] where [height, width]
#       equals to `output_size`.
#     image_info: a 2D `Tensor` that encodes the information of the image and the
#       applied preprocessing. It is in the format of
#       [[original_height, original_width], [desired_height, desired_width],
#        [y_scale, x_scale], [y_offset, x_offset]], where [desired_height,
#       desired_width] is the actual scaled image size, and [y_scale, x_scale] is
#       the scaling factor, which is the ratio of
#       scaled dimension / original dimension.
#   """
#   with tf.name_scope('resize_and_crop_image'):
#     if jitter > 1 or jitter < 0:
#       raise Exception("maximum change in aspect ratio must be between 0 and 1")
#     if tf.cast(random_pad, tf.float32) > 0.0:
#       random_pad = True
#     else:
#       random_pad = False

#     original_dims = tf.shape(image)[:2]
#     jitter = tf.cast(jitter, tf.float32)
#     ow = tf.cast(original_dims[1], tf.float32)
#     oh = tf.cast(original_dims[0], tf.float32)
#     w = tf.cast(desired_size[1], tf.float32)
#     h = tf.cast(desired_size[0], tf.float32)

#     dw = ow * jitter
#     dh = oh * jitter

#     if resize != 1:
#       max_rdw, max_rdh = 0.0, 0.0
#       if resize > 1.0:
#         resize_up = resize if resize > 1.0 else 1/resize
#         max_rdw = ow * (1 - (1 / resize_up)) / 2
#         max_rdh = oh * (1 - (1 / resize_up)) / 2

#       resize_down = resize if resize < 1.0 else 1/resize
#       min_rdw = ow * (1 - (1 / resize_down)) / 2
#       min_rdh = oh * (1 - (1 / resize_down)) / 2

#     pleft = rand_uniform_strong(-dw, dw, dw.dtype)
#     pright = rand_uniform_strong(-dw, dw, dw.dtype)
#     ptop = rand_uniform_strong(-dh, dh, dh.dtype)
#     pbottom = rand_uniform_strong(-dh, dh, dh.dtype)

#     if resize != 1:
#       pleft += rand_uniform_strong(min_rdw, max_rdw)
#       pright += rand_uniform_strong(min_rdw, max_rdw)
#       ptop += rand_uniform_strong(min_rdh, max_rdh)
#       pbottom += rand_uniform_strong(min_rdh, max_rdh)

#     if letter_box:
#       image_aspect_ratio = ow/oh
#       input_aspect_ratio = w/h

#       distorted_aspect = image_aspect_ratio/input_aspect_ratio
#       if distorted_aspect > 1:
#         delta_h = ((ow/input_aspect_ratio) - oh)/2
#         ptop = ptop - delta_h
#         pbottom = pbottom - delta_h
#       else:
#         delta_w = ((oh * input_aspect_ratio) - ow)/2
#         pright = pright - delta_w
#         pleft = pleft - delta_w

#     infos = []
#     swidth = tf.cast(ow - pleft - pright, tf.int32)
#     sheight = tf.cast(oh - ptop - pbottom, tf.int32)

#     pleft, pright, ptop, pbottom = cast([pleft,pright,ptop,pbottom], tf.int32)
#     src_crop = intersection([ptop, pleft, sheight + ptop, swidth + pleft],
#                             [0, 0, original_dims[0], original_dims[1]])
#     h_ = (src_crop[2] - src_crop[0])
#     w_ = (src_crop[3] - src_crop[1])

#     cropped_image = tf.slice(image, [src_crop[0], src_crop[1], 0], [h_, w_, -1])
#     crop_info = tf.stack([
#           tf.cast(original_dims, tf.float32),
#           tf.cast(tf.shape(cropped_image)[:2], dtype=tf.float32),
#           tf.ones_like(original_dims, dtype = tf.float32),
#           tf.cast(src_crop[:2], tf.float32)
#       ])
#     infos.append(crop_info)

#     if not crop_only:
#       if random_pad:
#         rmh = tf.maximum(0, -ptop)
#         rmw = tf.maximum(0, -pleft)
#       else:
#         rmw = tf.cast(tf.cast(swidth - w_, tf.float32) * shiftx, w_.dtype)
#         rmh = tf.cast(tf.cast(sheight - h_, tf.float32) * shifty, h_.dtype)
#       dst_shape = [rmh,rmw,rmh + h_,rmw + w_]
#       ptop, pleft, pbottom, pright = dst_shape

#       padded_image =  tf.image.pad_to_bounding_box(
#         cropped_image, dst_shape[0],dst_shape[1], sheight, swidth)
#       pad_info = tf.stack([
#             tf.cast(tf.shape(cropped_image)[:2], tf.float32),
#             tf.cast(tf.shape(padded_image)[:2], dtype=tf.float32),
#             tf.ones_like(original_dims, dtype = tf.float32),
#             -tf.cast(dst_shape[:2], tf.float32)
#         ])
#       infos.append(pad_info)

#       image = tf.image.resize(padded_image,
#                               (desired_size[0], desired_size[1]),
#                               method = method)
#     else:
#       image = cropped_image

#     if not crop_only and cut is not None:
#       image, crop_info = mosaic_cut(image, ow, oh, w, h, cut,
#                                     ptop, pleft, pbottom, pright,
#                                     shiftx, shifty)
#       infos.append(crop_info)

#     return image, infos

# def resize_and_jitter_image(image,
#                             desired_size,
#                             jitter=0.0,
#                             letter_box=None,
#                             resize = 1.0,
#                             random_pad=True,
#                             crop_only = False,
#                             shiftx=0.0,
#                             shifty=0.0,
#                             cut = None,
#                             method=tf.image.ResizeMethod.BILINEAR,
#                             seed = None):
#   """Resizes the input image to output size (RetinaNet style).
#   Resize and pad images given the desired output size of the image and
#   stride size.
#   Here are the preprocessing steps.
#   1. For a given image, keep its aspect ratio and rescale the image to make it
#      the largest rectangle to be bounded by the rectangle specified by the
#      `desired_size`.
#   2. Pad the rescaled image to the padded_size.
#   Args:
#     image: a `Tensor` of shape [height, width, 3] representing an image.
#     desired_size: a `Tensor` or `int` list/tuple of two elements representing
#       [height, width] of the desired actual output image size.
#     padded_size: a `Tensor` or `int` list/tuple of two elements representing
#       [height, width] of the padded output image size. Padding will be applied
#       after scaling the image to the desired_size.
#     aug_scale_min: a `float` with range between [0, 1.0] representing minimum
#       random scale applied to desired_size for training scale jittering.
#     aug_scale_max: a `float` with range between [1.0, inf] representing maximum
#       random scale applied to desired_size for training scale jittering.
#     seed: seed for random scale jittering.
#     method: function to resize input image to scaled image.
#   Returns:
#     output_image: `Tensor` of shape [height, width, 3] where [height, width]
#       equals to `output_size`.
#     image_info: a 2D `Tensor` that encodes the information of the image and the
#       applied preprocessing. It is in the format of
#       [[original_height, original_width], [desired_height, desired_width],
#        [y_scale, x_scale], [y_offset, x_offset]], where [desired_height,
#       desired_width] is the actual scaled image size, and [y_scale, x_scale] is
#       the scaling factor, which is the ratio of
#       scaled dimension / original dimension.
#   """
#   def cast(values, dtype):
#     return [tf.cast(tf.round(value), dtype) for value in values]

#   with tf.name_scope('resize_and_jitter_image'):
#     if jitter > 1 or jitter < 0:
#       raise Exception("maximum change in aspect ratio must be between 0 and 1")

#     if tf.cast(random_pad, tf.float32) > 0.0:
#       random_pad = True
#     else:
#       random_pad = False

#     original_dims = tf.shape(image)[:2]
#     jitter = tf.cast(jitter, tf.float32)
#     ow = tf.cast(original_dims[1], tf.float32)
#     oh = tf.cast(original_dims[0], tf.float32)
#     w = tf.cast(desired_size[1], tf.float32)
#     h = tf.cast(desired_size[0], tf.float32)

#     dw = ow * jitter
#     dh = oh * jitter

#     if resize != 1:
#       if resize > 1.0:
#         resize_up = resize if resize > 1.0 else 1/resize
#         max_rdw = ow * (1 - (1 / resize_up)) / 2
#         max_rdh = oh * (1 - (1 / resize_up)) / 2
#       else:
#         max_rdw = 0.0
#         max_rdh = 0.0

#       resize_down = resize if resize < 1.0 else 1/resize
#       min_rdw = ow * (1 - (1 / resize_down)) / 2
#       min_rdh = oh * (1 - (1 / resize_down)) / 2

#     pleft = rand_uniform_strong(-dw, dw, dw.dtype, seed = seed)
#     pright = rand_uniform_strong(-dw, dw, dw.dtype, seed = seed)
#     ptop = rand_uniform_strong(-dh, dh, dh.dtype, seed = seed)
#     pbottom = rand_uniform_strong(-dh, dh, dh.dtype, seed = seed)

#     if resize != 1:
#       pleft += rand_uniform_strong(min_rdw, max_rdw, seed = seed)
#       pright += rand_uniform_strong(min_rdw, max_rdw, seed = seed)
#       ptop += rand_uniform_strong(min_rdh, max_rdh, seed = seed)
#       pbottom += rand_uniform_strong(min_rdh, max_rdh, seed = seed)

#     if letter_box == True or letter_box is None:
#       image_aspect_ratio = ow/oh
#       input_aspect_ratio = w/h
#       distorted_aspect = image_aspect_ratio/input_aspect_ratio

#       delta_h, delta_w = 0.0, 0.0
#       pullin_h, pullin_w = 0.0, 0.0
#       if distorted_aspect > 1:
#         delta_h = ((ow/input_aspect_ratio) - oh)/2
#       else:
#         delta_w = ((oh * input_aspect_ratio) - ow)/2

#       if letter_box is None:
#         rwidth = ow + delta_w + delta_w
#         rheight = oh + delta_h + delta_h
#         if rheight < h and rwidth < w:
#           pullin_h = ((h - rheight) * rheight/h)/2
#           pullin_w = ((w - rwidth ) * rwidth/w)/2

#       ptop = ptop - delta_h - pullin_h
#       pbottom = pbottom - delta_h - pullin_h
#       pright = pright - delta_w - pullin_w
#       pleft = pleft - delta_w - pullin_w

#     swidth = tf.cast(ow - pleft - pright, tf.int32)
#     sheight = tf.cast(oh - ptop - pbottom, tf.int32)

#     pleft, pright, ptop, pbottom = cast([pleft,pright,ptop,pbottom], tf.int32)
#     src_crop = intersection([ptop, pleft, sheight + ptop, swidth + pleft],
#                             [0, 0, original_dims[0], original_dims[1]])

#     h_ = (src_crop[2] - src_crop[0])
#     w_ = (src_crop[3] - src_crop[1])
#     if random_pad:
#       rmh = tf.maximum(0, -ptop)
#       rmw = tf.maximum(0, -pleft)
#     else:
#       rmw = tf.cast(tf.cast(swidth - w_, tf.float32) * shiftx, w_.dtype)
#       rmh = tf.cast(tf.cast(sheight - h_, tf.float32) * shifty, h_.dtype)
#     dst_shape = [rmh,rmw,rmh + h_,rmw + w_]
#     ptop, pleft, pbottom, pright = dst_shape

#     pad = dst_shape * tf.convert_to_tensor([1, 1, -1, -1])
#     pad += tf.convert_to_tensor([0, 0, sheight, swidth])

#     infos = []

#     cropped_image = tf.slice(image,
#                              [src_crop[0], src_crop[1], 0],
#                              [src_crop[2] - src_crop[0],
#                               src_crop[3] - src_crop[1], -1])
#     crop_info = tf.stack([
#           tf.cast(original_dims, tf.float32),
#           tf.cast(tf.shape(cropped_image)[:2], dtype=tf.float32),
#           tf.ones_like(original_dims, dtype = tf.float32),
#           tf.cast(src_crop[:2], tf.float32)
#       ])
#     infos.append(crop_info)

#     if crop_only:
#       if not letter_box:
#         h_, w_ = get_image_shape(cropped_image)
#         w = tf.cast((w_* tf.cast(w, tf.int32))/ swidth, tf.int32)
#         h = tf.cast((h_* tf.cast(h, tf.int32))/ sheight, tf.int32)
#         cropped_image = tf.image.resize(cropped_image, [h, w], method = method)
#       return cropped_image, infos, cast([ow,oh,w,h,ptop,pleft,pbottom,pright],tf.int32)

#     image_ = tf.pad(cropped_image, [[pad[0], pad[2]],
#                                     [pad[1], pad[3]],
#                                     [0, 0]])
#     pad_info = tf.stack([
#           tf.cast(tf.shape(cropped_image)[:2], tf.float32),
#           tf.cast(tf.shape(image_)[:2], dtype=tf.float32),
#           tf.ones_like(original_dims, dtype = tf.float32),
#           -tf.cast(pad[:2], tf.float32)
#       ])
#     infos.append(pad_info)

#     image_ = tf.image.resize(image_, (desired_size[0], desired_size[1]))
#     if cut is not None:
#       image_, crop_info = mosaic_cut(image_, *cast([ow, oh, w, h, cut,
#                                     ptop, pleft, pbottom, pright], tf.float32),
#                                     shiftx, shifty)
#       infos.append(crop_info)

#     return image_, infos, cast([ow,oh,w,h,ptop,pleft,pbottom,pright],tf.int32)

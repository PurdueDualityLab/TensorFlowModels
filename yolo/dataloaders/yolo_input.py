""" Detection Data parser and processing for YOLO.
Parse image and ground truths in a dataset to training targets and package them
into (image, labels) tuple for RetinaNet.
"""

# Import libraries
import tensorflow as tf
import tensorflow_addons as tfa

from yolo.ops import preprocessing_ops
from yolo.ops import box_ops as box_utils
from official.vision.beta.ops import box_ops, preprocess_ops
from official.vision.beta.dataloaders import parser, utils
from yolo.ops import loss_utils as loss_ops


def pad_max_instances(value, instances, pad_value=0, pad_axis=0):
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


# image_w=608, image_h=608


class Parser(parser.Parser):
  """Parser to parse an image and its annotations into a dictionary of tensors."""

  def __init__(self,
               output_size,
               min_level=3,
               max_level=5,
               aug_rand_crop=0.0,
               max_num_instances=200,
               aug_rand_transalate=0.0,
               aug_rand_saturation=1.0,
               aug_rand_brightness=1.0,
               aug_rand_hue=1.0,
               aug_scale_aspect=0.0,
               aug_rand_angle=0.0,
               aug_scale_min=1.0,
               aug_scale_max=1.0,
               random_pad=True,
               anchor_t=4.0,
               scale_xy=None,
               use_scale_xy=True,
               masks=None,
               anchors=None,
               letter_box=False,
               random_flip=True,
               use_tie_breaker=True,
               dtype='float32'):
    """Initializes parameters for parsing annotations in the dataset.
    Args:
      output_size: `Tensor` or `list` for [height, width] of output image. The
        output_size should be divided by the largest feature stride 2^max_level.
      min_level: `int` number of minimum level of the output feature pyramid.
      max_level: `int` number of maximum level of the output feature pyramid.
      aug_rand_crop: `float` for the maximum change in aspect ratio expected in 
        each preprocessing step.
      max_num_instances: `int` for the number of boxes to compute loss on.
      aug_rand_transalate: `float` ranging from 0 to 1 indicating the maximum 
        amount to randomly translate an image.
      aug_rand_saturation: `float` indicating the maximum scaling value for 
        saturation. saturation will be scaled between 1/value and value.
      aug_rand_brightness: `float` indicating the maximum scaling value for 
        brightness. brightness will be scaled between 1/value and value.
      aug_rand_hue: `float` indicating the maximum scaling value for 
        hue. saturation will be scaled between 1 - value and 1 + value.
      aug_scale_aspect: `float` indicating the maximum scaling value for 
        aspect. aspect will be scaled between 1 - value and 1 + value
      aug_rand_angle: `float` indicating the maximum angle value for 
        angle. angle will be changes between 0 and value.
      aug_scale_min: `float` indicating the minimum scaling value for image 
        scale jitter. 
      aug_scale_max: `float` indicating the maximum scaling value for image 
        scale jitter.
      random_pad: `bool` indiccating wether to use padding to apply random 
        translation true for darknet yolo false for scaled yolo
      anchor_t: `float` indicating the threshold over which an anchor will be 
        considered for prediction, at zero, all the anchors will be used and at
        1.0 only the best will be used. for anchor thresholds larger than 1.0 
        we stop using the IOU for anchor comparison and resort directly to 
        comparing the width and height, this is used for the scaled models   
      scale_xy: dictionary `float` values inidcating how far each pixel can see 
        outside of its containment of 1.0. a value of 1.2 indicates there is a 
        20% extended radius around each pixel that this specific pixel can 
        predict values for a center at. the center can range from 0 - value/2 
        to 1 + value/2, this value is set in the yolo filter, and resused here. 
        there should be one value for scale_xy for each level from min_level to 
        max_level  
      use_scale_xy: `boolean` indicating weather the scale_xy values shoudl be 
        used or ignored. used if set to True. 
      masks: dictionary of lists of `int` values indicating the indexes in the 
        list of anchor boxes to use an each prediction level between min_level 
        and max_level. each level must have a list of indexes.  
      anchors: list of lists of `float` values for each anchor box.
      letter_box: `boolean` indicating whether upon start of the datapipeline 
        regardless of the preprocessing ops that are used, the aspect ratio of 
        the images should be preserved.  
      random_flip: `boolean` indicating whether or not to randomly flip the 
        image horizontally 
      use_tie_breaker: `boolean` indicating whether to use the anchor threshold 
        value
      dtype: `str` indicating the output datatype of the datapipeline selecting 
        from {"float32", "float16", "bfloat16"}
    """

    # base initialization
    image_w = output_size[1]
    image_h = output_size[0]
    self._net_down_scale = 2**max_level

    # assert that the width and height is viable
    assert image_w % self._net_down_scale == 0
    assert image_h % self._net_down_scale == 0

    # set the width and height properly
    self._image_w = image_w
    self._image_h = image_h

    # set the anchor boxes and masks for each scale
    self._anchors = anchors
    self._masks = {
        key: tf.convert_to_tensor(value) for key, value in masks.items()
    }
    self._use_tie_breaker = use_tie_breaker
    self._max_num_instances = max_num_instances

    # image spatial distortion
    self._aug_rand_crop = 0.0 if aug_rand_crop is None else aug_rand_crop
    self._aug_scale_aspect = 0.0 if aug_scale_aspect is None else aug_scale_aspect
    self._random_flip = random_flip
    self._letter_box = letter_box
    self._aug_scale_min = aug_scale_min
    self._aug_scale_max = aug_scale_max
    self._random_pad = random_pad
    self._aug_rand_angle = aug_rand_angle
    self._aug_rand_translate = aug_rand_transalate

    # color space distortion of the image
    self._aug_rand_saturation = aug_rand_saturation
    self._aug_rand_brightness = aug_rand_brightness
    self._aug_rand_hue = aug_rand_hue

    # set the per level values needed for operation
    self._scale_xy = scale_xy
    self._anchor_t = anchor_t
    self._use_scale_xy = use_scale_xy
    keys = list(self._masks.keys())
    self._scale_up = {
        key: int(self._anchor_t + len(keys) - i) for i, key in enumerate(keys)
    } if self._use_scale_xy else {key: 1 for key in keys}

    # set the data type based on input string
    if dtype == 'float16':
      self._dtype = tf.float16
    elif dtype == 'bfloat16':
      self._dtype = tf.bfloat16
    elif dtype == 'float32':
      self._dtype = tf.float32
    else:
      raise Exception(
          'Unsupported datatype used in parser only {float16, bfloat16, or float32}.'
      )

  def _build_grid(self,
                  raw_true,
                  width,
                  batch=False,
                  use_tie_breaker=False,
                  is_training=True):
    '''Private function for building the full scale object and class grid'''
    mask = self._masks
    inds = {}
    upds = {}
    true_conf = {}

    # based on if training or not determine how to scale up the number of
    # boxes that may result for final loss computation
    if is_training:
      scale_up = self._scale_up
    else:
      scale_up = {key: 1 for key in self._masks.keys()}

    # for each prediction path generate a properly scaled output prediction map
    for key in self._masks.keys():
      if self._use_scale_xy:
        scale_xy = self._scale_xy[key]
      else:
        scale_xy = 1

      # build the actual grid as well and the list of boxes and classes AND
      # their index in the prediction grid
      indexes, updates, true_grid = preprocessing_ops.build_grided_gt_ind(
          raw_true, self._masks[key], width // 2**int(key), 0,
          raw_true['bbox'].dtype, scale_xy, scale_up[key], use_tie_breaker)

      # set/fix the shape of the indexes
      ishape = indexes.get_shape().as_list()
      ishape[-2] = self._max_num_instances * scale_up[key]
      indexes.set_shape(ishape)

      # set/fix the shape of the updates
      ishape = updates.get_shape().as_list()
      ishape[-2] = self._max_num_instances * scale_up[key]
      updates.set_shape(ishape)

      # add all the values to the final dictionary
      inds[key] = indexes
      upds[key] = tf.cast(updates, self._dtype)
      true_conf[key] = true_grid
    return mask, inds, upds, true_conf

  def _parse_train_data(self, data):
    """Parses data for training and evaluation."""

    # initialize the shape constants
    shape = tf.shape(data['image'])
    image = data['image'] / 255
    boxes = data['groundtruth_boxes']
    classes = data['groundtruth_classes']
    height, width = preprocessing_ops.get_image_shape(image)


    if self._random_flip:
      # randomly flip the image horizontally
      image, boxes, _ = preprocess_ops.random_horizontal_flip(image, boxes)

    # if not data['is_mosaic']:  
    # resize the image irrespective of the aspect ratio
    # if not self._letter_box:
    #   clipper = tf.reduce_max((height, width))
    #   image = tf.image.resize(
    #       image, (clipper, clipper), preserve_aspect_ratio=False)
    # if not self._letter_box:
    #   # clipper = tf.reduce_max((height, width))
    #   image = image = tf.image.resize(
    #     image, (self._image_h, self._image_w),preserve_aspect_ratio=False)
    # else:
    #   height, width = preprocessing_ops.get_image_shape(image)
    #   clipper = tf.reduce_max((height, width))
    #   w_scale = width / clipper
    #   h_scale = height / clipper

    #   height_, width_ = self._image_h, self._image_w
    #   height_ = tf.cast(h_scale * tf.cast(height_, h_scale.dtype), tf.int32)
    #   width_ = tf.cast(w_scale * tf.cast(width_, w_scale.dtype), tf.int32)

    #   image = image = tf.image.resize(
    #     image, (height_, width_),preserve_aspect_ratio=False)

      # tf.print(width/height, width_/height_, width, height, width_, height_)

    # apply the random aspect ratio crop to the image
    # if self._aug_rand_crop > 0 and not data['is_mosaic']:

    #   # # crop the image
    #   # image, info = preprocessing_ops.random_aspect_crop(image, 
    #   #                                                     daspect = self._aug_rand_crop)
                                                          
    #   # # compute the net jitter
    #   jmi = 1 - self._aug_rand_crop
    #   jma = 1 + self._aug_rand_crop
    #   image, info = preprocessing_ops.random_crop_image(
    #       image, aspect_ratio_range=[jmi, jma], area_range=[jmi, 1.0])

    #   # use the info to crop the boxes and classes as well
    #   boxes = box_ops.denormalize_boxes(boxes, info[0, :])
    #   boxes = preprocess_ops.resize_and_crop_boxes(boxes, info[2, :],
    #                                               info[1, :], info[3, :])

    #   inds = box_ops.get_non_empty_box_indices(boxes)
    #   boxes = tf.gather(boxes, inds)
    #   classes = tf.gather(classes, inds)
    #   boxes = box_ops.normalize_boxes(boxes, info[1, :])
    # # elif data['is_mosaic'] and self._aug_rand_crop > 0:
    # #   # same as above but the crops applied to the mosaiced images are treated
    # #   # differently
    # #   jmi = 1 - self._aug_rand_crop
    # #   jma = 1 + self._aug_rand_crop
    # #   image, info = preprocessing_ops.random_crop_mosaic(
    # #       image, aspect_ratio_range=[jmi, jma], area_range=[0.25, 1.0])

    # #   boxes = box_ops.denormalize_boxes(boxes, info[0, :])
    # #   boxes = preprocess_ops.resize_and_crop_boxes(boxes, info[2, :],
    # #                                               info[1, :], info[3, :])

    # #   inds = box_ops.get_non_empty_box_indices(boxes)
    # #   boxes = tf.gather(boxes, inds)
    # #   classes = tf.gather(classes, inds)
    # #   boxes = box_ops.normalize_boxes(boxes, info[1, :])

    # if self._letter_box:
    #   # use the saved distortion values to return the cropeed image to proper
    #   # aspect ratio, it is doen this way in order to allow the random crop to
    #   # be indpendent of the images natural input resolution
    #   height_, width_ = preprocessing_ops.get_image_shape(image)
    #   height_ = tf.cast(h_scale * tf.cast(height_, h_scale.dtype), tf.int32)
    #   width_ = tf.cast(w_scale * tf.cast(width_, w_scale.dtype), tf.int32)
    #   image = tf.image.resize(
    #       image, (height_, width_), preserve_aspect_ratio=False)

    # if self._aug_scale_aspect > 0.0 and not data['is_mosaic']:
    #   # apply aspect ratio distortion (stretching and compressing)
    #   height_, width_ = preprocessing_ops.get_image_shape(image)

    #   shiftx = 1.0 + preprocessing_ops.rand_uniform_strong(
    #       -self._aug_scale_aspect, self._aug_scale_aspect)
    #   shifty = 1.0 + preprocessing_ops.rand_uniform_strong(
    #       -self._aug_scale_aspect, self._aug_scale_aspect)
    #   width_ = tf.cast(tf.cast(width_, shifty.dtype) * shifty, tf.int32)
    #   height_ = tf.cast(tf.cast(height_, shiftx.dtype) * shiftx, tf.int32)

    #   image = tf.image.resize(image, (height_, width_))



    # apply the final scale jittering of the image, if the image is mosaiced
    # ensure the minimum is larger than 0.4, or 0.1 for each image in the
    # mosaic
    if not data['is_mosaic']:
      image, infos = preprocessing_ops.resize_and_jitter_image(
          image, [self._image_h, self._image_w], [self._image_h, self._image_w],
          letter_box = self._letter_box, 
          scale_aspect=self._aug_scale_aspect,
          aug_scale_min=self._aug_scale_min,
          aug_scale_max=self._aug_scale_max,
          jitter=self._aug_rand_crop,
          random_pad=self._random_pad)
    else:
      # works well
      image, infos = preprocessing_ops.resize_and_jitter_image(
          image, [self._image_h, self._image_w], [self._image_h, self._image_w],
          letter_box = self._letter_box, 
          scale_aspect=self._aug_scale_aspect,
          aug_scale_min=1.0, #self._aug_scale_min if self._aug_scale_min > 0.4 else 0.4,
          aug_scale_max=1.0, #self._aug_scale_max, #self._aug_scale_max / 2,
          jitter=0.0,
          random_pad=self._random_pad)
      # image, infos = preprocessing_ops.resize_and_jitter_image(
      #     image, [self._image_h, self._image_w], [self._image_h, self._image_w],
      #     aug_scale_min=self._aug_scale_min if self._aug_scale_min > 0.4 else 0.4,
      #     aug_scale_max=self._aug_scale_max, #self._aug_scale_max / 2,
      #     jitter=self._aug_rand_crop,
      #     random_pad=self._random_pad)

    # again crop the boxes and classes and only use those that are still
    # in the image.
    for info in infos:
      boxes = box_ops.denormalize_boxes(boxes, info[0, :])
      boxes = preprocessing_ops.resize_and_crop_boxes(boxes, info[2, :], info[1, :],
                                                  info[3, :], keep_thresh = 0.1)

      inds = box_ops.get_non_empty_box_indices(boxes)
      boxes = tf.gather(boxes, inds)
      classes = tf.gather(classes, inds)
      boxes = box_ops.normalize_boxes(boxes, info[1, :])

    if self._aug_rand_translate > 0.0:
      # apply random translation to the image
      image, tx, ty = preprocessing_ops.random_translate(
          image, self._aug_rand_translate)
      boxes, classes = preprocessing_ops.translate_boxes(boxes, classes, tx, ty)

    if self._aug_rand_angle > 0:
      # apply rotation to the images
      image, angle = preprocessing_ops.random_rotate_image(
          image, self._aug_rand_angle)
      boxes = preprocessing_ops.rotate_boxes(boxes, angle)

    image = tf.image.resize(
        image, (self._image_h, self._image_w),
        preserve_aspect_ratio=False,
        antialias=False,
        name=None)

    # propagate all the changes to the images to the boxes and classes
    # mainly image clipping and removing boxes no longer in the image
    h_, w_ = preprocessing_ops.get_image_shape(image)
    im_shape = tf.cast([h_, w_], tf.float32)
    boxes = box_ops.denormalize_boxes(boxes, im_shape)
    boxes = box_ops.clip_boxes(boxes, im_shape)

    # inds = preprocessing_ops.get_non_empty_box_indices(boxes, im_shape)
    inds = box_ops.get_non_empty_box_indices(boxes)
    boxes = tf.gather(boxes, inds)
    classes = tf.gather(classes, inds)
    boxes = box_ops.normalize_boxes(boxes, im_shape)

    # apply scaling to the hue saturation and brightness of an image
    num_dets = tf.shape(classes)[0]
    if self._aug_rand_hue > 0.0:
      delta = preprocessing_ops.rand_uniform_strong(-self._aug_rand_hue,
                                                    self._aug_rand_hue)
      image = tf.image.adjust_hue(image, delta)
    if self._aug_rand_saturation > 0.0:
      delta = preprocessing_ops.rand_scale(self._aug_rand_saturation)
      image = tf.image.adjust_saturation(image, delta)
    if self._aug_rand_brightness > 0.0:
      delta = preprocessing_ops.rand_scale(self._aug_rand_brightness)
      image *= delta
    # clip the values of the image between 0.0 and 1.0
    image = tf.clip_by_value(image, 0.0, 1.0)

    # cast the image to the selcted datatype
    image = tf.cast(image, self._dtype)
    image, labels = self._build_label(
        image, boxes, classes, width, height, info, data, is_training=True)
    return image, labels

  def _parse_eval_data(self, data):

    # get the image shape constants
    shape = tf.shape(data['image'])
    image = data['image'] / 255
    boxes = data['groundtruth_boxes']
    classes = data['groundtruth_classes']
    height, width = preprocessing_ops.get_image_shape(image)

    image, infos = preprocessing_ops.resize_and_jitter_image(
          image, [self._image_h, self._image_w], [self._image_h, self._image_w],
          letter_box = self._letter_box, 
          scale_aspect = 0.0,
          aug_scale_min = 1.0, #self._aug_scale_min if self._aug_scale_min > 0.4 else 0.4,
          aug_scale_max = 1.0, #self._aug_scale_max, #self._aug_scale_max / 2,
          jitter = 0.0,
          random_pad = False, 
          shiftx = 0.0, 
          shifty = 0.0)

    for info in infos:
      boxes = box_ops.denormalize_boxes(boxes, info[0, :])
      boxes = preprocessing_ops.resize_and_crop_boxes(boxes, info[2, :], info[1, :],
                                                  info[3, :], keep_thresh = 0.0)

      inds = box_ops.get_non_empty_box_indices(boxes)
      boxes = tf.gather(boxes, inds)
      classes = tf.gather(classes, inds)
      boxes = box_ops.normalize_boxes(boxes, info[1, :])

    # cast the image to the selcted datatype
    image = tf.cast(image, self._dtype)
    image, labels = self._build_label(
        image, boxes, classes, width, height, info, data, is_training=False)
    return image, labels

  def _build_label(self,
                   image,
                   boxes,
                   classes,
                   width,
                   height,
                   info,
                   data,
                   is_training=True):
    """Label construction for both the train and eval data. """

    # set the image shape
    imshape = image.get_shape().as_list()
    imshape[-1] = 3
    image.set_shape(imshape)

    # get the best anchors
    boxes = box_utils.yxyx_to_xcycwh(boxes)
    best_anchors, ious = preprocessing_ops.get_best_anchor(
        boxes,
        self._anchors,
        width=self._image_w,
        height=self._image_h,
        iou_thresh=self._anchor_t)

    # set/fix the boxes shape
    bshape = boxes.get_shape().as_list()
    boxes = pad_max_instances(boxes, self._max_num_instances, 0)
    bshape[0] = self._max_num_instances
    boxes.set_shape(bshape)

    # set/fix the classes shape
    cshape = classes.get_shape().as_list()
    classes = pad_max_instances(classes, self._max_num_instances, -1)
    cshape[0] = self._max_num_instances
    classes.set_shape(cshape)

    # set/fix the best anchor shape
    bashape = best_anchors.get_shape().as_list()
    best_anchors = pad_max_instances(best_anchors, self._max_num_instances, -1)
    bashape[0] = self._max_num_instances
    best_anchors.set_shape(bashape)

    # set/fix the ious shape
    ishape = ious.get_shape().as_list()
    ious = pad_max_instances(ious, self._max_num_instances, 0)
    ishape[0] = self._max_num_instances
    ious.set_shape(ishape)

    # set/fix the area shape
    area = data['groundtruth_area']
    ashape = area.get_shape().as_list()
    area = pad_max_instances(area, self._max_num_instances, 0)
    ashape[0] = self._max_num_instances
    area.set_shape(ashape)

    # set/fix the is_crowd shape
    is_crowd = data['groundtruth_is_crowd']
    ishape = is_crowd.get_shape().as_list()
    is_crowd = pad_max_instances(
        tf.cast(is_crowd, tf.int32), self._max_num_instances, 0)
    ishape[0] = self._max_num_instances
    is_crowd.set_shape(ishape)

    # build the dictionary set
    labels = {
        'source_id': utils.process_source_id(data['source_id']),
        'bbox': tf.cast(boxes, self._dtype),
        'classes': tf.cast(classes, self._dtype),
        'area': tf.cast(area, self._dtype),
        'is_crowd': is_crowd,
        'best_anchors': tf.cast(best_anchors, self._dtype),
        'best_iou_match': ious,
        'width': width,
        'height': height,
        'info': info,
        'num_detections': tf.shape(data['groundtruth_classes'])[0]
    }

    # build the grid formatted for loss computation in model output format
    grid, inds, upds, true_conf = self._build_grid(
        labels,
        self._image_w,
        use_tie_breaker=self._use_tie_breaker,
        is_training=is_training)

    # update the labels dictionary
    labels['bbox'] = box_utils.xcycwh_to_yxyx(labels['bbox'])
    labels['upds'] = upds
    labels['inds'] = inds
    labels['true_conf'] = true_conf
    return image, labels

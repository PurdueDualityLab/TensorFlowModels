""" Detection Data parser and processing for YOLO.
Parse image and ground truths in a dataset to training targets and package them
into (image, labels) tuple for RetinaNet.
"""
import tensorflow as tf
from yolo.ops import preprocessing_ops
from yolo.ops import box_ops as box_utils
from official.vision.beta.ops import preprocess_ops
from official.vision.beta.dataloaders import parser, utils


def _coco91_to_80(classif, box, areas, iscrowds):
  """Function used to reduce COCO 91 to COCO 80, or to convert from the 2017 
  foramt to the 2014 format"""
  # Vector where index i coralates to the class at index[i].
  x = [
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
      23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
      44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62,
      63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85,
      86, 87, 88, 89, 90
  ]
  no = tf.expand_dims(tf.convert_to_tensor(x), axis=0)
  
  # Resahpe the classes to in order to build a class mask.
  ce = tf.expand_dims(classif, axis=-1)

  # One hot the classificiations to match the 80 class format.
  ind = ce == tf.cast(no, ce.dtype)

  # Select the max values. 
  co = tf.reshape(tf.math.argmax(tf.cast(ind, tf.float32), axis=-1), [-1])
  ind = tf.where(tf.reduce_any(ind, axis=-1))

  # Gather the valuable instances.
  classif = tf.gather_nd(co, ind)
  box = tf.gather_nd(box, ind)
  areas = tf.gather_nd(areas, ind)
  iscrowds = tf.gather_nd(iscrowds, ind)

  # Restate the number of viable detections, ideally it should be the same.
  num_detections = tf.shape(classif)[0]
  return classif, box, areas, iscrowds, num_detections


class Parser(parser.Parser):
  """Parser to parse an image and its annotations into a dictionary of 
  tensors."""

  def __init__(self,
               output_size,
               min_level=3,
               max_level=5,
               jitter=0.0,
               jitter_mosaic=0.0,
               resize=1.0,
               resize_mosaic=1.0,
               area_thresh=0.1,
               max_num_instances=200,
               aug_rand_angle=0.0,
               aug_rand_saturation=1.0,
               aug_rand_brightness=1.0,
               aug_rand_hue=1.0,
               random_pad=True,
               aug_scale_min=1.0,
               aug_scale_max=1.0,
               mosaic_min=1.0,
               mosaic_max=1.0,
               aug_rand_transalate=0.0,
               mosaic_translate=0.0,
               anchor_t=4.0,
               dynamic_conv=False,
               stride=None,
               scale_xy=None,
               use_scale_xy=False,
               best_match_only=False,
               masks=None,
               anchors=None,
               letter_box=False,
               random_flip=True,
               use_tie_breaker=True,
               dtype='float32',
               coco91to80=False,
               anchor_free_limits=None, 
               seed=None):
    """Initializes parameters for parsing annotations in the dataset.
    Args:
      output_size: `Tensor` or `list` for [height, width] of output image. The
        output_size should be divided by the largest feature stride 2^max_level.
      min_level: `int` number of minimum level of the output feature pyramid.
      max_level: `int` number of maximum level of the output feature pyramid.
      jitter: `float` for the maximum change in aspect ratio expected in 
        each preprocessing step.
      jitter_mosaic: `float` for the maximum change in aspect ratio expected in 
        each preprocessing step to be applied to mosaiced images.
      resize: `float` for the maximum change in image size.
      resize_mosaic: `float` for the maximum change in image size to be applied 
        to mosaiced images.
      area_thresh: `float` for the minimum area of a box to allow to pass 
        through for optimization.
      max_num_instances: `int` for the number of boxes to compute loss on.
      aug_rand_angle: `float` indicating the maximum angle value for 
        angle. angle will be changes between 0 and value.
      aug_rand_saturation: `float` indicating the maximum scaling value for 
        saturation. saturation will be scaled between 1/value and value.
      aug_rand_brightness: `float` indicating the maximum scaling value for 
        brightness. brightness will be scaled between 1/value and value.
      aug_rand_hue: `float` indicating the maximum scaling value for 
        hue. saturation will be scaled between 1 - value and 1 + value.
      aug_scale_min: `float` indicating the minimum scaling value for image 
        scale jitter. 
      aug_scale_max: `float` indicating the maximum scaling value for image 
        scale jitter.
      mosaic_min: `float` indicating the minimum scaling value for image 
        scale jitter for mosaiced images. 
      mosaic_max: `float` indicating the maximum scaling value for image 
        scale jitter for mosaiced images.
      random_pad: `bool` indiccating wether to use padding to apply random 
        translation true for darknet yolo false for scaled yolo.
      aug_rand_transalate: `float` ranging from 0 to 1 indicating the maximum 
        amount to randomly translate an image.
      mosaic_translate: `float` ranging from 0 to 1 indicating the maximum 
        amount to randomly translate an image for mosaiced images.
      anchor_t: `float` indicating the threshold over which an anchor will be 
        considered for prediction, at zero, all the anchors will be used and at
        1.0 only the best will be used. for anchor thresholds larger than 1.0 
        we stop using the IOU for anchor comparison and resort directly to 
        comparing the width and height, this is used for the scaled models.  
      dynamic_conv: `bool` for whether to use a padding in evaluation on the 
        GPUs.
      stride: `int` for how much the model scales down the images at the largest
        level.
      scale_xy: dictionary `float` values inidcating how far each pixel can see 
        outside of its containment of 1.0. a value of 1.2 indicates there is a 
        20% extended radius around each pixel that this specific pixel can 
        predict values for a center at. the center can range from 0 - value/2 
        to 1 + value/2, this value is set in the yolo filter, and resused here. 
        there should be one value for scale_xy for each level from min_level to 
        max_level.
      use_scale_xy: `boolean` indicating weather the scale_xy values shoudl be 
        used or ignored. used if set to True. 
      best_match_only: `boolean` indicating how boxes are selected for 
        optimization.
      masks: dictionary of lists of `int` values indicating the indexes in the 
        list of anchor boxes to use an each prediction level between min_level 
        and max_level. each level must have a list of indexes.  
      anchors: list of lists of `float` values for each anchor box.
      letter_box: `boolean` indicating whether upon start of the datapipeline 
        regardless of the preprocessing ops that are used, the aspect ratio of 
        the images should be preserved.  
      random_flip: `boolean` indicating whether or not to randomly flip the 
        image horizontally. 
      use_tie_breaker: `boolean` indicating whether to use the anchor threshold 
        value.
      dtype: `str` indicating the output datatype of the datapipeline selecting 
        from {"float32", "float16", "bfloat16"}.
      coco91to80: `bool` for wether to convert coco91 to coco80 to minimize 
        model parameters.
      seed: `int` the seed for random number generation. 
    """
    self._coco91to80 = coco91to80

    # Base initialization
    image_w = output_size[1]
    image_h = output_size[0]
    if stride is None:
      self._net_down_scale = 2**max_level
    else:
      self._net_down_scale = stride

    # Assert that the width and height is viable
    assert image_w % self._net_down_scale == 0
    assert image_h % self._net_down_scale == 0

    # Set the width and height properly
    self._image_w = image_w
    self._image_h = image_h

    # Set the anchor boxes and masks for each scale
    self._anchors = anchors
    self._masks = {
        key: tf.convert_to_tensor(value) for key, value in masks.items()
    }
    self._use_tie_breaker = use_tie_breaker
    self._max_num_instances = max_num_instances

    # Image scaling params
    self._jitter = 0.0 if jitter is None else jitter
    self._resize = 1.0 if resize is None else resize
    self._aug_scale_min = aug_scale_min
    self._aug_scale_max = aug_scale_max
    self._aug_rand_translate = aug_rand_transalate

    # Mosaic scaling params
    self._jitter_mosaic = 0.0 if jitter_mosaic is None else jitter_mosaic
    self._resize_mosaic = 0.0 if resize_mosaic is None else resize_mosaic
    self._mosaic_min = mosaic_min
    self._mosaic_max = mosaic_max
    self._mosaic_translate = mosaic_translate

    # Image spatial distortion
    self._random_flip = random_flip
    self._letter_box = letter_box
    self._random_pad = random_pad
    self._aug_rand_angle = aug_rand_angle

    # Color space distortion of the image
    self._aug_rand_saturation = aug_rand_saturation
    self._aug_rand_brightness = aug_rand_brightness
    self._aug_rand_hue = aug_rand_hue
    self._best_match_only = best_match_only

    # Set the per level values needed for operation
    self._dynamic_conv = dynamic_conv
    self._scale_xy = scale_xy
    self._anchor_t = anchor_t
    self._use_scale_xy = use_scale_xy
    keys = list(self._masks.keys())

    scale = max(max(self._image_w, self._image_h)/640, 1)
    # self._scale_up = {key: max(round(2 * scale - 0.5 * i), 1) for i, key in enumerate(keys)
    #                  } if self._use_scale_xy else {key: 1 for key in keys}
    self._scale_up = {key: 4 for i, key in enumerate(keys)
                     } if self._use_scale_xy else {key: 1 for key in keys}
    self._area_thresh = area_thresh
    self._anchor_free_limits = anchor_free_limits

    self._seed = seed

    # Set the data type based on input string
    if dtype == 'float16':
      self._dtype = tf.float16
    elif dtype == 'bfloat16':
      self._dtype = tf.bfloat16
    elif dtype == 'float32':
      self._dtype = tf.float32
    else:
      raise Exception(
          'Unsupported datatype used in parser\
          only {float16, bfloat16, or float32}.'
      )

  def _build_grid(self,
                  raw_true,
                  width,
                  height,
                  batch=False,
                  use_tie_breaker=False,
                  is_training=True):
    '''Private function for building the full scale object and class grid.'''
    mask = self._masks
    inds = {}
    upds = {}
    true_conf = {}

    # based on if training or not determine how to scale up the number of
    # boxes that may result for final loss computation
    scale_up = self._scale_up

    # for each prediction path generate a properly scaled output prediction map
    for key in self._masks.keys():
      if self._use_scale_xy:
        scale_xy = self._scale_xy[key]
      else:
        scale_xy = 1

      # build the actual grid as well and the list of boxes and classes AND
      # their index in the prediction grid
      indexes, updates, true_grid = preprocessing_ops.build_grided_gt_ind(
          raw_true, self._masks[key], width // 2**int(key),
          height // 2**int(key), 0, raw_true['bbox'].dtype, scale_xy,
          scale_up[key], use_tie_breaker)

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

  def _get_identity_info(self, image):
    """Get an identity image op to pad all info vectors, this is used because 
    graph compilation if there are a variable number of info objects in a list.
    """
    shape_ = tf.shape(image)
    val = tf.stack([
        tf.cast(shape_[:2], tf.float32),
        tf.cast(shape_[:2], tf.float32),
        tf.ones_like(tf.cast(shape_[:2], tf.float32)),
        tf.zeros_like(tf.cast(shape_[:2], tf.float32)),
    ])
    return val

  def _jitter_scale(self, image, shape, letter_box, jitter, resize, random_pad,
                    aug_scale_min, aug_scale_max, translate, angle,
                    perspective):
    if (aug_scale_min != 1.0 or aug_scale_max != 1.0):
      crop_only = True
      # jitter gives you only one info object, resize and crop gives you one,
      # if crop only then there can be 1 form jitter and 1 from crop
      reps = 1
    else:
      crop_only = False
      reps = 0
    infos = []
    image, info_a, _ = preprocessing_ops.resize_and_jitter_image(
        image,
        shape,
        letter_box=letter_box,
        jitter=jitter,
        resize=resize,
        crop_only=crop_only,
        random_pad=random_pad,
        seed=self._seed,
    )
    infos.extend(info_a)
    stale_a = self._get_identity_info(image)
    for _ in range(reps):
      infos.append(stale_a)
    image, _, affine = preprocessing_ops.affine_warp_image(
        image,
        shape,
        scale_min=aug_scale_min,
        scale_max=aug_scale_max,
        translate=translate,
        degrees=angle,
        perspective=perspective,
        random_pad=random_pad,
        seed=self._seed,
    )
    return image, infos, affine

  def reorg91to80(self, data):
    """Function used to reduce COCO 91 to COCO 80, or to convert from the 2017 
    foramt to the 2014 format"""
    if self._coco91to80:
      (data['groundtruth_classes'], data['groundtruth_boxes'],
       data['groundtruth_area'], data['groundtruth_is_crowd'],
       _) = _coco91_to_80(data['groundtruth_classes'], data['groundtruth_boxes'],
                         data['groundtruth_area'], data['groundtruth_is_crowd'])
    return data

  def _parse_train_data(self, data):
    """Parses data for training and evaluation."""
    # Down size coco 91 to coco 80 if the option is selected.
    data = self.reorg91to80(data)

    # Initialize the shape constants.
    image = data['image']
    boxes = data['groundtruth_boxes']
    classes = data['groundtruth_classes']
    height, width = preprocessing_ops.get_image_shape(image)

    if self._random_flip:
      # Randomly flip the image horizontally.
      image, boxes, _ = preprocess_ops.random_horizontal_flip(
          image, boxes, seed=self._seed)

    if not data['is_mosaic']:
      image, infos, affine = self._jitter_scale(
          image, [self._image_h, self._image_w], self._letter_box, self._jitter,
          self._resize, self._random_pad, self._aug_scale_min,
          self._aug_scale_max, self._aug_rand_translate, self._aug_rand_angle,
          0.0)
    else:
      image, infos, affine = self._jitter_scale(
          image, [self._image_h, self._image_w], self._letter_box,
          self._jitter_mosaic, self._resize_mosaic, self._random_pad,
          self._mosaic_min, self._mosaic_max, self._mosaic_translate,
          self._aug_rand_angle, 0.0)

    # Clip and clean boxes.
    boxes, inds = preprocessing_ops.apply_infos(
        boxes,
        infos,
        affine=affine,
        shuffle_boxes=False,
        area_thresh=self._area_thresh,
        seed=self._seed)
    classes = tf.gather(classes, inds)
    info = infos[-1]

    # Apply scaling to the hue saturation and brightness of an image.
    image = tf.cast(image, self._dtype)
    image = image / 255
    if self._aug_rand_hue != 0 and self._aug_rand_saturation != 0 and self._aug_rand_brightness != 0: 
      image = preprocessing_ops.image_rand_hsv(
          image,
          self._aug_rand_hue,
          self._aug_rand_saturation,
          self._aug_rand_brightness,
          seed=self._seed, 
          darknet= not self._use_scale_xy)

      # Cast the image to the selcted datatype.
      image = tf.clip_by_value(image, 0.0, 1.0)
    height, width = self._image_h, self._image_w
    image, labels = self._build_label(
        image,
        boxes,
        classes,
        width,
        height,
        info,
        inds,
        data,
        is_training=True)
    return image, labels

  def _parse_eval_data(self, data):
    # Down size coco 91 to coco 80 if the option is selected.
    data = self.reorg91to80(data)

    # Get the image shape constants and cast the image to the selcted datatype.
    image = tf.cast(data['image'], self._dtype)
    boxes = data['groundtruth_boxes']
    classes = data['groundtruth_classes']

    if not self._dynamic_conv:
      height, width = self._image_h, self._image_w
    else:
      fit = lambda x: tf.cast((tf.math.ceil(
          (x / self._net_down_scale) + 0.5) * self._net_down_scale), x.dtype)
      height, width = preprocessing_ops.get_image_shape(image)
      height, width = fit(height), fit(width)

    image, infos, _ = preprocessing_ops.resize_and_jitter_image(
        image, [height, width],
        letter_box=self._letter_box,
        random_pad=False,
        shiftx=0.5,
        shifty=0.5,
        jitter=0.0,
        resize=1.0)

    # Clip and clean boxes.
    image = image / 255
    boxes, inds = preprocessing_ops.apply_infos(
        boxes, infos, shuffle_boxes=False, area_thresh=self._area_thresh)
    classes = tf.gather(classes, inds)
    info = infos[-1]

    image, labels = self._build_label(
        image,
        boxes,
        classes,
        width,
        height,
        info,
        inds,
        data,
        is_training=False)
    return image, labels

  def _build_label(self,
                   image,
                   boxes_,
                   classes,
                   width,
                   height,
                   info,
                   inds,
                   data,
                   is_training=True):
    """Label construction for both the train and eval data. """

    # Set the image shape.
    imshape = image.get_shape().as_list()
    imshape[-1] = 3
    image.set_shape(imshape)

    # Get the best anchors.
    boxes = box_utils.yxyx_to_xcycwh(boxes_)
    best_anchors, ious = preprocessing_ops.get_best_anchor(
        boxes,
        self._anchors,
        width=width,
        height=height,
        iou_thresh=self._anchor_t,
        anchor_free_limits = self._anchor_free_limits, 
        best_match_only=self._best_match_only)

    # Set/fix the boxes shape.
    bshape = boxes.get_shape().as_list()
    boxes = preprocessing_ops.pad_max_instances(boxes, self._max_num_instances, 0)
    bshape[0] = self._max_num_instances
    boxes.set_shape(bshape)

    # Set/fix the classes shape.
    cshape = classes.get_shape().as_list()
    classes = preprocessing_ops.pad_max_instances(classes, self._max_num_instances, -1)
    cshape[0] = self._max_num_instances
    classes.set_shape(cshape)

    # Set/fix the best anchor shape.
    bashape = best_anchors.get_shape().as_list()
    best_anchors = preprocessing_ops.pad_max_instances(best_anchors, self._max_num_instances, -1)
    bashape[0] = self._max_num_instances
    best_anchors.set_shape(bashape)

    # Set/fix the ious shape.
    ishape = ious.get_shape().as_list()
    ious = preprocessing_ops.pad_max_instances(ious, self._max_num_instances, 0)
    ishape[0] = self._max_num_instances
    ious.set_shape(ishape)

    # Set/fix the area shape.
    area = data['groundtruth_area']
    area = tf.gather(area, inds)
    ashape = area.get_shape().as_list()
    area = preprocessing_ops.pad_max_instances(area, self._max_num_instances, 0)
    ashape[0] = self._max_num_instances
    area.set_shape(ashape)

    # Set/fix the is_crowd shape.
    is_crowd = data['groundtruth_is_crowd']
    is_crowd = tf.gather(is_crowd, inds)
    ishape = is_crowd.get_shape().as_list()
    is_crowd = preprocessing_ops.pad_max_instances(
        tf.cast(is_crowd, tf.int32), self._max_num_instances, 0)
    ishape[0] = self._max_num_instances
    is_crowd.set_shape(ishape)

    # Build the dictionary set.
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
        'num_detections': tf.shape(inds)[0]
    }

    # Build the grid formatted for loss computation in model output format.
    grid, inds, upds, true_conf = self._build_grid(
        labels,
        width,
        height,
        use_tie_breaker=self._use_tie_breaker,
        is_training=is_training)

    # Update the labels dictionary.
    labels['bbox'] = box_utils.xcycwh_to_yxyx(labels['bbox'])
    labels['upds'] = upds
    labels['inds'] = inds
    labels['true_conf'] = true_conf

    # Sets up groundtruth data for evaluation.
    groundtruths = {
        'source_id': data['source_id'],
        'height': data['height'],
        'width': data['width'],
        'num_detections': labels['num_detections'],
        'image_info': info,
        'boxes': boxes_,
        'classes': labels['classes'],
        'areas': data['groundtruth_area'],
        'is_crowds': tf.cast(data['groundtruth_is_crowd'], tf.int32),
    }
    groundtruths['source_id'] = utils.process_source_id(
        groundtruths['source_id'])
    groundtruths = utils.pad_groundtruths_to_fixed_size(groundtruths,
                                                        self._max_num_instances)

    labels['groundtruths'] = groundtruths
    return image, labels

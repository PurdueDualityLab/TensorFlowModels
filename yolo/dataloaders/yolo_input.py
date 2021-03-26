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


class Parser(parser.Parser):
  """Parser to parse an image and its annotations into a dictionary of tensors."""

  def __init__(self,
               image_w=416,
               image_h=416,
               min_level=3,
               max_level=5,
               num_classes=80,
               masks=None,
               anchors=None,
               fixed_size=False,
               letter_box=False,
               use_tie_breaker=True,
               random_flip=True,
               mosaic_frequency=0.0,
               jitter_im=0.0,
               jitter_boxes=0.0025,
               aug_rand_transalate=0.0, 
               aug_rand_zoom=0.0,
               aug_rand_hue=0.1,
               aug_rand_saturation=1.5,
               aug_rand_brightness=1.5,
               max_process_size=608,
               min_process_size=320,
               max_num_instances=200,
               keep_thresh=0.25,
               pct_rand=0.5,
               seed=10,
               dtype='float32'):
    """Initializes parameters for parsing annotations in the dataset.
    Args:
      image_w: a `Tensor` or `int` for width of input image.
      image_h: a `Tensor` or `int` for height of input image.
      num_classes: a `Tensor` or `int` for the number of classes.
      fixed_size: a `bool` if True all output images have the same size.
      jitter_im: a `float` that is the maximum jitter applied to the image for
        data augmentation during training.
      jitter_boxes: a `float` that is the maximum jitter applied to the bounding
        box for data augmentation during training.
      net_down_scale: an `int` that down scales the image width and height to the
        closest multiple of net_down_scale.
      max_process_size: an `int` for maximum image width and height.
      min_process_size: an `int` for minimum image width and height ,
      max_num_instances: an `int` number of maximum number of instances in an image.
      random_flip: a `bool` if True, augment training with random horizontal flip.
      pct_rand: an `int` that prevents do_scale from becoming larger than 1-pct_rand.
      masks: a `Tensor`, `List` or `numpy.ndarrray` for anchor masks.
      aug_rand_saturation: `bool`, if True, augment training with random
        saturation.
      aug_rand_brightness: `bool`, if True, augment training with random
        brightness.
      aug_rand_zoom: `bool`, if True, augment training with random
        zoom.
      aug_rand_hue: `bool`, if True, augment training with random
        hue.
      anchors: a `Tensor`, `List` or `numpy.ndarrray` for bounding box priors.
      seed: an `int` for the seed used by tf.random
    """
    self._net_down_scale = 2**max_level

    self._num_classes = num_classes
    self._image_w = (image_w // self._net_down_scale) * self._net_down_scale
    self._image_h = self._image_w if image_h is None else (
        image_h // self._net_down_scale) * self._net_down_scale

    self._max_process_size = max_process_size
    self._min_process_size = min_process_size

    self._anchors = anchors
    self._masks = {
        key: tf.convert_to_tensor(value) for key, value in masks.items()
    }
    self._use_tie_breaker = use_tie_breaker

    self._jitter_im = 0.0 if jitter_im is None else jitter_im
    self._jitter_boxes = 0.0 if jitter_boxes is None else jitter_boxes
    self._pct_rand = pct_rand
    self._max_num_instances = max_num_instances
    self._random_flip = random_flip
    self._letter_box = letter_box

    self._aug_rand_translate = aug_rand_transalate
    self._aug_rand_saturation = aug_rand_saturation
    self._aug_rand_brightness = aug_rand_brightness
    self._aug_rand_zoom = aug_rand_zoom
    self._aug_rand_hue = aug_rand_hue
    self._keep_thresh = keep_thresh
    self._mosaic_frequency = mosaic_frequency

    self._seed = seed
    self._fixed_size = fixed_size

    if dtype == 'float16':
      self._dtype = tf.float16
    elif dtype == 'bfloat16':
      self._dtype = tf.bfloat16
    elif dtype == 'float32':
      self._dtype = tf.float32
    else:
      raise Exception(
          'Unsupported datatype used in parser only {float16, bfloat16, or float32}'
      )

  def _build_grid(self, raw_true, width, batch=False, use_tie_breaker=False):
    mask = self._masks
    inds = {}
    upds = {}
    true_conf = {}

    for key in self._masks.keys():
      indexes, updates, true_grid = preprocessing_ops.build_grided_gt_ind(raw_true, self._masks[key],
                                                    width // 2**int(key),
                                                    self._num_classes,
                                                    raw_true['bbox'].dtype,
                                                    use_tie_breaker)
      
      ishape = indexes.get_shape().as_list()
      ishape[-2] = self._max_num_instances
      indexes.set_shape(ishape)

      ishape = updates.get_shape().as_list()
      ishape[-2] = self._max_num_instances
      updates.set_shape(ishape)
      
      inds[key] = indexes
      upds[key] = tf.cast(updates, self._dtype)
      true_conf[key] = true_grid
    return mask, inds, upds, true_conf

  def _parse_train_data(self, data):
    """Generates images and labels that are usable for model training.
        Args:
          data: a dict of Tensors produced by the decoder.
        Returns:
          images: the image tensor.
          labels: a dict of Tensors that contains labels.
        """

    shape = tf.shape(data['image'])
    image = data['image'] / 255
    boxes = data['groundtruth_boxes']
    classes = data['groundtruth_classes']
    width = shape[1]
    height = shape[0]

    if self._aug_rand_hue > 0.0:
      delta = preprocessing_ops.rand_uniform_strong(-self._aug_rand_hue, self._aug_rand_hue)
      image = tf.image.adjust_hue(image, delta)
    if self._aug_rand_saturation > 0.0:
      delta = preprocessing_ops.rand_scale(self._aug_rand_saturation)
      image = tf.image.adjust_saturation(image, delta)
    if self._aug_rand_brightness > 0.0:
      delta = preprocessing_ops.rand_scale(self._aug_rand_brightness)
      image *= delta
    image = tf.clip_by_value(image, 0.0, 1.0)

    if self._random_flip:
      image, boxes, _ = preprocess_ops.random_horizontal_flip(
          image, boxes, seed=self._seed)

    if self._jitter_boxes > 0.0:
      height_, width_ = preprocessing_ops.get_image_shape(image)
      
      shiftx = 1.0 + preprocessing_ops.rand_uniform_strong(-self._jitter_boxes , self._jitter_boxes)
      shifty = 1.0 + preprocessing_ops.rand_uniform_strong(-self._jitter_boxes , self._jitter_boxes)
      width_ = tf.cast(tf.cast(width_, shifty.dtype) * shifty, tf.int32)
      height_ = tf.cast(tf.cast(height_, shiftx.dtype) * shiftx, tf.int32)

      image = tf.image.resize(image, (height_, width_))

    if data['is_mosaic']:
      zooms = [0.25, 0.5]
      image, info = preprocessing_ops.random_crop_image(image, aspect_ratio_range = [1.0, 1.0], area_range=zooms)
    else:
      shiftx = preprocessing_ops.rand_uniform_strong(0.0, 1.0)
      shifty = preprocessing_ops.rand_uniform_strong(0.0, 1.0)
      image, boxes, info = preprocessing_ops.letter_box(image, boxes, xs = shiftx, ys = shifty, target_dim=self._image_w)
      jmi = 1 - 2 * self._jitter_im
      jma = 1 + 2 * self._jitter_im
      image, info = preprocessing_ops.random_crop_image(image, aspect_ratio_range = [jmi, jma], area_range=[0.5, 1.0])

    boxes = box_ops.denormalize_boxes(boxes, info[0, :])
    boxes = preprocess_ops.resize_and_crop_boxes(boxes, info[2, :], info[1, :], info[3, :])
    
    inds = box_ops.get_non_empty_box_indices(boxes)
    boxes = tf.gather(boxes, inds)
    classes = tf.gather(classes, inds)
    boxes = box_ops.normalize_boxes(boxes, info[1, :])


    # if self._aug_rand_zoom > 0.0 and self._mosaic_frequency > 0.0:
    #   zfactor = preprocessing_ops.rand_uniform_strong(self._aug_rand_zoom, 1.0)
    # elif self._aug_rand_zoom > 0.0:
    #   zfactor = preprocessing_ops.rand_scale(self._aug_rand_zoom)
    # else:
    #   zfactor = tf.convert_to_tensor(1.0)

    # image, crop_info = preprocessing_ops.random_op_image(
    #     image, self._jitter_im, zfactor, zfactor, self._aug_rand_translate)
    # boxes, classes = preprocessing_ops.filter_boxes_and_classes(
    #     boxes, classes, crop_info, keep_thresh=self._keep_thresh)

    # if self._jitter_im > 0.0:
    #   image, _ = preprocessing_ops.random_jitter(image, self._jitter_im)

    # image_shape = tf.shape(image)[:2]
    # boxes = box_ops.denormalize_boxes(boxes, image_shape)

    # if self._jitter_boxes > 0.0:
    #   boxes = box_ops.jitter_boxes(boxes, self._jitter_boxes)
    
    
    # if data['is_mosaic']:
    #   image, image_info = preprocess_ops.resize_and_crop_image(image, 
    #                                                           [self._image_h, self._image_w], 
    #                                                           [self._image_h, self._image_w], 
    #                                                           aug_scale_min= 1.0, 
    #                                                           aug_scale_max = 1/self._aug_rand_zoom) 
    # else:
    #   image, image_info = preprocess_ops.resize_and_crop_image(image, 
    #                                                           [self._image_h, self._image_w], 
    #                                                           [self._image_h, self._image_w], 
    #                                                           aug_scale_min= self._aug_rand_zoom, 
    #                                                           aug_scale_max = 1/self._aug_rand_zoom)                 
    
    # offset = image_info[3, :]
    # image_scale = image_info[2, :]
    # info_scale = tf.math.minimum(tf.round(image_info[0, :]*image_scale) - offset, image_info[1, :])
    # info = tf.cast(tf.convert_to_tensor([0, 0, info_scale[0], info_scale[1]]), tf.int32)
    # image_height, image_width, _ = image.get_shape().as_list()

    # # tf.print(info, image_info)
    # # Resizes and crops boxes.
    # boxes = preprocess_ops.resize_and_crop_boxes(boxes, image_scale,
    #                                              image_info[1, :], offset)

    # # Filters out ground truth boxes that are all zeros.
    # indices = box_ops.get_non_empty_box_indices(boxes)
    # boxes = tf.gather(boxes, indices)
    # classes = tf.gather(classes, indices)

    # boxes = box_ops.normalize_boxes(boxes, [self._image_h, self._image_w])

    #if self._letter_box:
    shiftx = preprocessing_ops.rand_uniform_strong(0.0, 1.0)
    shifty = preprocessing_ops.rand_uniform_strong(0.0, 1.0)
    image, boxes, info = preprocessing_ops.letter_box(image, boxes, xs = shiftx, ys = shifty, target_dim=self._image_w)
    # else:
    #   height, width = preprocessing_ops.get_image_shape(image)
    #   minscale = tf.math.minimum(width, height)
    #   image, image_info = preprocessing_ops.random_crop_or_pad(
    #       image,
    #       target_width=minscale,
    #       target_height=minscale,
    #       random_patch=True)
    #   image = tf.image.resize(image, (self._image_w, self._image_h))
    #   boxes, classes = preprocessing_ops.filter_boxes_and_classes(
    #       boxes, classes, image_info, keep_thresh=self._keep_thresh)
    #   info = tf.convert_to_tensor([0, 0, self._image_w, self._image_h])

    num_dets = tf.shape(classes)[0]

    image = tf.cast(image, self._dtype)
    image, labels = self._build_label(image, boxes, classes, width, height, info, data)
    return image, labels

  def _parse_eval_data(self, data):
    """Generates images and labels that are usable for model training.
        Args:
          data: a dict of Tensors produced by the decoder.
        Returns:
          images: the image tensor.
          labels: a dict of Tensors that contains labels.
        """

    shape = tf.shape(data['image'])
    image = data['image'] / 255
    boxes = data['groundtruth_boxes']
    classes = data['groundtruth_classes']
    width = shape[1]
    height = shape[0]

    image_shape = tf.shape(image)[:2]

    # if self._letter_box:
    image, boxes, info = preprocessing_ops.letter_box(
        image, boxes, xs = 0.5, ys = 0.5, target_dim=self._image_w)
    # else:
    #   height, width = preprocessing_ops.get_image_shape(image)
    #   minscale = tf.math.minimum(width, height)
    #   image, image_info = preprocessing_ops.random_crop_or_pad(
    #       image,
    #       target_width=minscale,
    #       target_height=minscale,
    #       random_patch=True)
    #   image = tf.image.resize(image, (self._image_w, self._image_h))
    #   boxes, classes = preprocessing_ops.filter_boxes_and_classes(
    #       boxes, classes, image_info, keep_thresh=self._keep_thresh)
    #   info = tf.convert_to_tensor([0, 0, self._image_w, self._image_h])

    # boxes = box_ops.denormalize_boxes(boxes, image_shape)

    # image, image_info = preprocess_ops.resize_and_crop_image(image, 
    #                                                          [self._image_h, self._image_w], 
    #                                                          [self._image_h, self._image_w], 
    #                                                          aug_scale_min=1.0, 
    #                                                          aug_scale_max = 1.0)                 
    
    # offset = image_info[3, :]
    # image_scale = image_info[2, :]
    # info_scale = tf.math.minimum(tf.round(image_info[0, :]*image_scale) - offset, image_info[1, :])
    # info = tf.cast(tf.convert_to_tensor([0, 0, info_scale[0], info_scale[1]]), tf.int32)
    # image_height, image_width, _ = image.get_shape().as_list()

    # # tf.print(info, image_info)
    # # Resizes and crops boxes.
    # boxes = preprocess_ops.resize_and_crop_boxes(boxes, image_scale,
    #                                              image_info[1, :], offset)

    # # Filters out ground truth boxes that are all zeros.
    # indices = box_ops.get_non_empty_box_indices(boxes)
    # boxes = tf.gather(boxes, indices)
    # classes = tf.gather(classes, indices)

    # boxes = box_ops.normalize_boxes(boxes, [self._image_h, self._image_w])

    image = tf.cast(image, self._dtype)
    image, labels = self._build_label(image, boxes, classes, width, height, info, data)

    return image, labels

  def _build_label(self, image, boxes, classes, width, height, info, data):


    imshape = image.get_shape().as_list()
    imshape[-1] = 3
    image.set_shape(imshape)


    boxes = box_utils.yxyx_to_xcycwh(boxes)

    best_anchors, ious = preprocessing_ops.get_best_anchor(
        boxes, self._anchors, width=self._image_w, height=self._image_h)

    bshape = boxes.get_shape().as_list()
    boxes = pad_max_instances(boxes, self._max_num_instances, 0)
    bshape[0] = self._max_num_instances
    boxes.set_shape(bshape)
    
    
    cshape = classes.get_shape().as_list()
    classes = pad_max_instances(classes,
                                self._max_num_instances, -1)
    cshape[0] = self._max_num_instances
    classes.set_shape(cshape)

    bashape = best_anchors.get_shape().as_list()
    best_anchors = pad_max_instances(best_anchors, self._max_num_instances, -1)
    bashape[0] = self._max_num_instances
    best_anchors.set_shape(bashape)

    ishape = ious.get_shape().as_list()
    ious = pad_max_instances(ious, self._max_num_instances, 0)
    ishape[0] = self._max_num_instances
    ious.set_shape(ishape)

    area = data['groundtruth_area']
    ashape = area.get_shape().as_list()
    area = pad_max_instances(area, self._max_num_instances,0)
    ashape[0] = self._max_num_instances
    area.set_shape(ashape)

    is_crowd = data['groundtruth_is_crowd']
    ishape = is_crowd.get_shape().as_list()
    is_crowd = pad_max_instances(
        tf.cast(is_crowd, tf.int32), self._max_num_instances, 0)
    ishape[0] = self._max_num_instances
    is_crowd.set_shape(ishape)

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

    grid, inds, upds, true_conf = self._build_grid(
        labels, self._image_w, use_tie_breaker=self._use_tie_breaker)
    # labels.update({'grid_form': grid})
    labels['bbox'] = box_utils.xcycwh_to_yxyx(labels['bbox'])
    labels['upds'] = upds 
    labels['inds'] = inds
    labels['true_conf'] = true_conf
    return image, labels


  def postprocess_fn(self, is_training):
    if is_training:  #or self._cutmix
      return None # if not self._fixed_size or self._mosaic else None
    else:
      return None
  


  def sample_fn(self, dataset):
    dataset = dataset.padded_batch(4)
    dataset.unbatch()


# class Mosaic():
#   def __init__(self, frequency, )
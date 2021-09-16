import random
import tensorflow as tf
from tensorflow._api.v2 import data
import tensorflow_addons as tfa
import tensorflow.keras.backend as K
from yolo.ops import box_ops as bbox_ops
from yolo.ops import preprocessing_ops
from official.vision.beta.ops import box_ops, preprocess_ops

import tensorflow_datasets as tfds
from yolo.dataloaders.decoders import tfds_coco_decoder
from yolo.utils.demos import utils, coco
import matplotlib.pyplot as plt


# gen a random number for each sample a subset of the dataset to mosaic?
class Mosaic(object):

  def __init__(self,
               output_size,

               mosaic_frequency=1.0,
               mixup_frequency=0.0,
               
               letter_box=True,
               jitter=0.0,

               mosaic_crop_mode='scale',
               mosaic_center=[0.25],

               aug_scale_min = 1.0, 
               aug_scale_max = 1.0,
               aug_rand_angle=0.0, 
               aug_rand_perspective=0.0,
               aug_rand_transalate=0.0,
               random_pad=False,
               area_thresh=0.1,

               deterministic=True,
               seed=None):

    # Establish the expected output size and the maximum resolution to use for
    # padding and batching throughout the mosaic process.
    self._output_size = output_size
    self._area_thresh = area_thresh

    # Establish the mosaic frequency and mix up frequency.
    self._mosaic_frequency = mosaic_frequency
    self._mixup_frequency = mixup_frequency

    # Establish how the image to treat images prior to cropping
    # letter boxing preserves aspect ratio, cropping will take random crops
    # of the input samples prior to patching, distortion will allow for
    # arbitraty changes to aspect ratio and crops.
    self._letter_box = letter_box
    self._random_crop = jitter
    
    # How to treat final output images, None indicates that nothing will occur,
    # if the mode is crop then the mosaic will be generate by cropping and
    # slicing. If scale is selected the images are concatnated together and
    # and scaled.
    self._mosaic_crop_mode = mosaic_crop_mode
    self._mosaic_center = mosaic_center

    self._aug_scale_min = aug_scale_min
    self._aug_scale_max = aug_scale_max
    self._random_pad = random_pad
    self._aug_rand_transalate = aug_rand_transalate
    self._aug_rand_angle = aug_rand_angle
    self._aug_rand_perspective = aug_rand_perspective

    self._seed = seed if seed is not None else random.randint(0, 2**30)
    self._deterministic = deterministic
    return

  def _generate_cut(self):
    """Using the provided maximum delat for center location generate a 
    random delta from the center of the image to use for image patching 
    and slicing."""
    if self._mosaic_crop_mode == 'crop':
      min_offset = self._mosaic_center
      cut_x = preprocessing_ops.rand_uniform_strong(
          self._output_size[1] * min_offset,
          self._output_size[1] * (1 - min_offset),
          seed=self._seed)
      cut_y = preprocessing_ops.rand_uniform_strong(
          self._output_size[0] * min_offset,
          self._output_size[0] * (1 - min_offset),
          seed=self._seed)
      cut = [cut_x, cut_y]
      ishape = tf.convert_to_tensor(
          [self._output_size[1], self._output_size[0], 3])
    else:
      cut = None
      ishape = tf.convert_to_tensor(
          [self._output_size[1] * 2, self._output_size[0] * 2, 3])
    return cut, ishape

  def _process_image(self,
                     image,
                     boxes,
                     classes,
                     is_crowd,
                     area,
                     xs=0.0,
                     ys=0.0,
                     cut=None):
    """Process a single image prior to the application of patching."""
    # Randomly flip the image horizontally.
    letter_box = self._letter_box

    image, infos, crop_points = preprocessing_ops.resize_and_jitter_image(
        image, [self._output_size[0], self._output_size[1]],
        random_pad=False,
        letter_box=letter_box,
        jitter=self._random_crop,
        shiftx=xs,
        shifty=ys,
        cut=cut,
        seed=self._seed)

    # Clip and clean boxes.
    boxes, inds = preprocessing_ops.apply_infos(
        boxes,
        infos,
        area_thresh=self._area_thresh,
        shuffle_boxes=False,
        augment=True,
        seed=self._seed)
    classes = tf.gather(classes, inds)
    is_crowd = tf.gather(is_crowd, inds)
    area = tf.gather(area, inds)
    return image, boxes, classes, is_crowd, area, crop_points

  def _mosaic_crop_image(self, image, boxes, classes, is_crowd, area):
    """Process a patched image in preperation for final output."""
    if self._mosaic_crop_mode != "crop":
      shape = tf.cast(preprocessing_ops.get_image_shape(image), tf.float32)
      center = shape * self._mosaic_center

      ch = tf.math.round(
          preprocessing_ops.rand_uniform_strong(
              -center[0], center[0], seed=self._seed))
      cw = tf.math.round(
          preprocessing_ops.rand_uniform_strong(
              -center[1], center[1], seed=self._seed))

      image = tfa.image.translate(
          image, [cw, ch], fill_value=preprocessing_ops.get_pad_value())
      boxes = box_ops.denormalize_boxes(boxes, shape[:2])
      boxes = boxes + tf.cast([ch, cw, ch, cw], boxes.dtype)
      boxes = box_ops.clip_boxes(boxes, shape[:2])
      boxes = box_ops.normalize_boxes(boxes, shape[:2])

    image, _, affine = preprocessing_ops.affine_warp_image(
        image, [self._output_size[0], self._output_size[1]],
        scale_min=self._aug_scale_min,
        scale_max=self._aug_scale_max,
        translate=self._aug_rand_transalate,
        degrees=self._aug_rand_angle,
        perspective=self._aug_rand_perspective,
        random_pad=self._random_pad,
        seed=self._seed)
    height, width = self._output_size[0], self._output_size[1]
    image = tf.image.resize(image, (height, width))

    # Clip and clean boxes.
    boxes, inds = preprocessing_ops.apply_infos(
        boxes, None, 
        affine=affine, 
        area_thresh=self._area_thresh, 
        augment=True,
        seed=self._seed)
    classes = tf.gather(classes, inds)
    is_crowd = tf.gather(is_crowd, inds)
    area = tf.gather(area, inds)
    return image, boxes, classes, is_crowd, area, area

  def scale_boxes(self, patch, ishape, boxes, classes, xs, ys):
    """Scale and translate the boxes for each image after patching has been 
    completed"""
    xs = tf.cast(xs, boxes.dtype)
    ys = tf.cast(ys, boxes.dtype)
    pshape = tf.cast(tf.shape(patch), boxes.dtype)
    ishape = tf.cast(ishape, boxes.dtype)
    translate = tf.cast((ishape - pshape), boxes.dtype)

    boxes = box_ops.denormalize_boxes(boxes, pshape[:2])
    boxes = boxes + tf.cast([
        translate[0] * ys, translate[1] * xs, translate[0] * ys,
        translate[1] * xs
    ], boxes.dtype)
    boxes = box_ops.normalize_boxes(boxes, ishape[:2])
    return boxes, classes

  # mosaic full frequency doubles model speed
  def _im_process(self, sample, shiftx, shifty, cut, ishape):
    """Distributed processing of each image"""
    (image, boxes, classes, is_crowd, area, crop_points) = self._process_image(
        sample['image'], sample['groundtruth_boxes'],
        sample['groundtruth_classes'], sample['groundtruth_is_crowd'],
        sample['groundtruth_area'], shiftx, shifty, cut)
    if cut is None and ishape is None:
      cut, ishape = self._generate_cut()

    (boxes, classes) = self.scale_boxes(image, ishape, boxes, classes,
                                        1 - shiftx, 1 - shifty)

    sample['image'] = image
    sample['groundtruth_boxes'] = boxes
    sample['groundtruth_classes'] = classes
    sample['groundtruth_is_crowd'] = is_crowd
    sample['groundtruth_area'] = area
    sample['cut'] = cut
    sample['shiftx'] = shiftx
    sample['shifty'] = shifty
    sample['crop_points'] = crop_points
    return sample

  def _patch2(self, one, two):
    sample = one
    sample['image'] = tf.concat([one["image"], two["image"]], axis=-2)

    sample['groundtruth_boxes'] = tf.concat(
        [one['groundtruth_boxes'], two['groundtruth_boxes']], axis=0)
    sample['groundtruth_classes'] = tf.concat(
        [one['groundtruth_classes'], two['groundtruth_classes']], axis=0)
    sample['groundtruth_is_crowd'] = tf.concat(
        [one['groundtruth_is_crowd'], two['groundtruth_is_crowd']], axis=0)
    sample['groundtruth_area'] = tf.concat(
        [one['groundtruth_area'], two['groundtruth_area']], axis=0)
    return sample

  def _patch(self, one, two):
    image = tf.concat([one["image"], two["image"]], axis=-3)
    boxes = tf.concat([one['groundtruth_boxes'], two['groundtruth_boxes']],
                      axis=0)
    classes = tf.concat(
        [one['groundtruth_classes'], two['groundtruth_classes']], axis=0)
    is_crowd = tf.concat(
        [one['groundtruth_is_crowd'], two['groundtruth_is_crowd']], axis=0)
    area = tf.concat([one['groundtruth_area'], two['groundtruth_area']], axis=0)

    if self._mosaic_crop_mode is not None:
      image, boxes, classes, is_crowd, area, info = self._mosaic_crop_image(
          image, boxes, classes, is_crowd, area)

    sample = one
    height, width = preprocessing_ops.get_image_shape(image)
    sample['image'] = tf.cast(image, tf.uint8)
    sample['groundtruth_boxes'] = boxes
    sample['groundtruth_area'] = area
    sample['groundtruth_classes'] = tf.cast(classes,
                                            sample['groundtruth_classes'].dtype)
    sample['groundtruth_is_crowd'] = tf.cast(is_crowd, tf.bool)
    sample['width'] = tf.cast(width, sample['width'].dtype)
    sample['height'] = tf.cast(height, sample['height'].dtype)
    sample['num_detections'] = tf.shape(sample['groundtruth_boxes'])[1]
    sample['is_mosaic'] = tf.cast(1.0, tf.bool)

    del sample['shiftx'], sample['shifty'], sample['crop_points'], sample['cut']
    return sample

  # mosaic partial frequency
  def _mosaic(self, one, two, three, four):
    if self._mosaic_frequency >= 1.0:
      domo = 1.0
    else:
      domo = preprocessing_ops.rand_uniform_strong(
        0.0, 1.0, dtype=tf.float32, seed=self._seed)
      noop = one.copy()

    if domo >= (1 - self._mosaic_frequency):
      cut, ishape = self._generate_cut()
      one   = self._im_process(one, 1.0, 1.0, cut, ishape)
      two   = self._im_process(two, 0.0, 1.0, cut, ishape)
      three = self._im_process(three, 1.0, 0.0, cut, ishape)
      four  = self._im_process(four, 0.0, 0.0, cut, ishape)
      patch1 = self._patch2(one, two)
      patch2 = self._patch2(three, four)
      stitched = self._patch(patch1, patch2)
      return stitched
    else:
      return self._add_param(noop)

  def _mixup(self, one, two):
    if self._mixup_frequency >= 1.0:
      domo = 1.0
    else:
      domo = preprocessing_ops.rand_uniform_strong(
        0.0, 1.0, dtype=tf.float32, seed=self._seed)
      noop = one.copy()
    
    if domo >= (1 - self._mixup_frequency):
      sample = one
      otype = one["image"].dtype
      r = preprocessing_ops.rand_uniform_strong(
          0.4, 0.6, tf.float32, seed=self._seed)
      sample['image'] = (
          r * tf.cast(one["image"], tf.float32) +
          (1 - r) * tf.cast(two["image"], tf.float32))

      sample['image'] = tf.cast(sample['image'], otype)
      sample['groundtruth_boxes'] = tf.concat(
          [one['groundtruth_boxes'], two['groundtruth_boxes']], axis=0)
      sample['groundtruth_classes'] = tf.concat(
          [one['groundtruth_classes'], two['groundtruth_classes']], axis=0)
      sample['groundtruth_is_crowd'] = tf.concat(
          [one['groundtruth_is_crowd'], two['groundtruth_is_crowd']], axis=0)
      sample['groundtruth_area'] = tf.concat(
          [one['groundtruth_area'], two['groundtruth_area']], axis=0)
      return sample
    else:
      return self._add_param(noop)

  def _add_param(self, sample):
    sample['is_mosaic'] = tf.cast(0.0, tf.bool)
    sample['num_detections'] = tf.shape(sample['groundtruth_boxes'])[0]
    return sample

  def _apply(self, dataset):
    determ = self._deterministic
    one = dataset.shuffle(100, seed=self._seed, reshuffle_each_iteration=True)
    two = dataset.shuffle(100, seed=self._seed + 1, reshuffle_each_iteration=True)
    three = dataset.shuffle(100, seed=self._seed + 2, reshuffle_each_iteration=True)
    four = dataset.shuffle(100, seed=self._seed + 3, reshuffle_each_iteration=True)

    dataset = tf.data.Dataset.zip((one, two, three, four))
    dataset = dataset.map(
          self._mosaic,
          num_parallel_calls=tf.data.AUTOTUNE,
          deterministic=determ)

    if self._mixup_frequency > 0:
      one = dataset.shuffle(10, seed=self._seed + 4, reshuffle_each_iteration=True)
      two = dataset.shuffle(10, seed=self._seed + 5, reshuffle_each_iteration=True)
      dataset = tf.data.Dataset.zip((one, two))
      dataset = dataset.map(
        self._mixup, num_parallel_calls=tf.data.AUTOTUNE, deterministic=determ)
    return dataset

  def _skip(self, dataset):
    determ = self._deterministic
    return dataset.map(
        self._add_param,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=determ)

  def mosaic_fn(self, is_training=True):
    if is_training and self._mosaic_frequency > 0.0:
      return self._apply
    else:
      return self._skip

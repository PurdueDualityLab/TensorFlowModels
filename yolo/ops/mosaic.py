from random import seed
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
               max_resolution=640,
               mosaic_frequency=1.0,
               random_crop=0.0,
               resize=1.0,
               aspect_ratio_mode='letter',
               random_flip=True,
               random_pad=False,
               translate=0.5,
               crop_area=[0.2],
               crop_area_mosaic=[0.5, 1.0],
               mosaic_crop_mode=None,
               mixup_frequency=0.0,
               area_thresh=0.1,
               seed=None):

    # Establish the expected output size and the maximum resolution to use for 
    # padding and batching throughout the mosaic process.
    self._output_size = output_size
    self._max_resolution = max_resolution

    # Establish the mosaic frequency and mix up frequency.
    self._mosaic_frequency = mosaic_frequency
    self._mixup_frequency = mixup_frequency

    # Establish how the image to treat images prior to cropping
    # letter boxing preserves aspect ratio, cropping will take random crops 
    # of the input samples prior to patching, distortion will allow for 
    # arbitraty changes to aspect ratio and crops.  
    self._aspect_ratio_mode = aspect_ratio_mode

    self._random_flip = random_flip
    self._random_crop = random_crop
    self._resize = resize
    self._random_pad = random_pad
    self._translate = translate

    # Center location for constructed mosaic.
    self._crop_area = crop_area
    # Minimum area of boxes for optimization.
    self._area_thresh = area_thresh

    # How to treat final output images, None indicates that nothing will occur,
    # if the mode is crop then the mosaic will be generate by cropping and 
    # slicing. If scale is selected the images are concatnated together and
    # and scaled.  
    self._mosaic_crop_mode = mosaic_crop_mode
    self._crop_area_mosaic = crop_area_mosaic

    self._seed = seed
    return

  def _estimate_shape(self, image):
    """Estimates a padded shape to use when batching images."""
    height, width = preprocessing_ops.get_image_shape(image)
    oheight, owidth = self._max_resolution, self._max_resolution
    if height < oheight and width < owidth:
      oheight = height
      owidth = width
    else:
      if width > height:
        oheight = height * self._max_resolution // width
      else:
        owidth = width * self._max_resolution // height
    return oheight, owidth

  def _generate_cut(self):
    """Using the provided maximum delat for center location generate a 
    random delta from the center of the image to use for image patching 
    and slicing."""
    if self._mosaic_crop_mode == 'crop':
      min_offset = self._crop_area[0]
      cut_x = preprocessing_ops.rand_uniform_strong(
          self._output_size[1] * min_offset,
          self._output_size[1] * (1 - min_offset),
          seed=self._seed)
      cut_y = preprocessing_ops.rand_uniform_strong(
          self._output_size[1] * min_offset,
          self._output_size[1] * (1 - min_offset),
          seed=self._seed)
      cut = [cut_x, cut_y]
      ishape = tf.convert_to_tensor(
          [self._output_size[1], self._output_size[0], 3])
    else:
      cut = None
      ishape = tf.convert_to_tensor(
          [self._output_size[1] * 2, self._output_size[0] * 2, 3])
    return cut, ishape

  def _pad_images(self, sample):
    """Pad images in order to batch them and stitch together images."""
    image = sample['image']
    height, width = self._estimate_shape(image)
    hclipper, wclipper = self._max_resolution, self._max_resolution
    image = tf.image.resize(image, (height, width))
    image = tf.image.pad_to_bounding_box(image, 0, 0, hclipper, wclipper)
    info = tf.convert_to_tensor([0, 0, height, width])

    sample['image'] = image
    sample['info'] = info
    sample['num_detections'] = tf.shape(sample['groundtruth_boxes'])[0]
    sample['is_mosaic'] = tf.cast(0.0, tf.bool)
    return sample

  def _unpad_images(self, image, info, squeeze=True):
    """Unpad images after patching is applied."""
    if squeeze:
      image = tf.squeeze(image, axis=0)
      info = tf.squeeze(info, axis=0)
    image = tf.slice(image, [info[0], info[1], 0], [info[2], info[3], -1])
    return image

  def _unpad_gt_comps(self, boxes, classes, is_crowd, area, squeeze=True):
    """Unpad labels after patching is applied."""
    if squeeze:
      boxes = tf.squeeze(boxes, axis=0)
      classes = tf.squeeze(classes, axis=0)
      is_crowd = tf.squeeze(is_crowd, axis=0)
      area = tf.squeeze(area, axis=0)

    indices = box_ops.get_non_empty_box_indices(boxes)
    boxes = tf.gather(boxes, indices)
    classes = tf.gather(classes, indices)
    is_crowd = tf.gather(classes, indices)
    area = tf.gather(area, indices)
    return boxes, classes, is_crowd, area

  def resample_unpad(self, sample):
    """Unpad images that are not mosaiced."""
    if not sample["is_mosaic"]:
      sample['image'] = self._unpad_images(
          sample['image'], sample['info'], squeeze=False)
      (sample['groundtruth_boxes'], sample['groundtruth_classes'],
       sample['groundtruth_is_crowd'],
       sample['groundtruth_area']) = self._unpad_gt_comps(
           sample['groundtruth_boxes'],
           sample['groundtruth_classes'],
           sample['groundtruth_is_crowd'],
           sample['groundtruth_area'],
           squeeze=False)
      sample['groundtruth_is_crowd'] = tf.cast(sample['groundtruth_is_crowd'],
                                               tf.bool)
    return sample

  def _crop_image(self, image, crop_area):
    """A function to apply a random crop with in the crop area to a given 
    image."""
    scale = preprocessing_ops.rand_uniform_strong(
        tf.math.sqrt(crop_area[0]), tf.math.sqrt(crop_area[1]))
    height, width = preprocessing_ops.get_image_shape(image)
    width = tf.cast(tf.cast(width, scale.dtype) * scale, tf.int32)
    height = tf.cast(tf.cast(height, scale.dtype) * scale, tf.int32)
    image, info = preprocessing_ops.random_window_crop(
        image, height, width, translate=1.0)
    return image, info

  def _gen_blank_info(self, image):
    """Get an identity image op to pad all info vectors, this is used because 
    graph compilation if there are a variable number of info objects in a list.
    """
    shape = tf.shape(image)
    info = tf.stack([
        tf.cast(shape[:2], tf.float32),
        tf.cast(shape[:2], tf.float32),
        tf.ones_like(tf.cast(shape[:2], tf.float32)),
        tf.zeros_like(tf.cast(shape[:2], tf.float32)),
    ])
    return info

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
    if self._random_flip:
      image, boxes, _ = preprocess_ops.random_horizontal_flip(image, boxes)

    infos = []
    random_crop = self._random_crop
    letter_box = True
    if self._aspect_ratio_mode == 'fixed':
      letter_box = None
    elif self._aspect_ratio_mode == 'distort':
      letter_box = False
    elif self._aspect_ratio_mode == 'crop':
      letter_box = None
      info = self._gen_blank_info(image)
      if tf.random.uniform([], 0.0, 1.0, dtype=tf.float32) > 0.5:
        image, info = self._crop_image(image, [0.25, 1.0])
      infos.append(info)
      

    image, infos_, crop_points = preprocessing_ops.resize_and_jitter_image(
        image, [self._output_size[0], self._output_size[1]],
        random_pad=False,
        letter_box=letter_box,
        jitter=random_crop,
        resize=self._resize,
        shiftx=xs,
        shifty=ys,
        cut=cut,
        seed=self._seed)
    infos.extend(infos_)

    # Clip and clean boxes.
    boxes, inds = preprocessing_ops.apply_infos(
        boxes, infos, area_thresh=self._area_thresh, seed=self._seed)
    classes = tf.gather(classes, inds)
    is_crowd = tf.gather(is_crowd, inds)
    area = tf.gather(area, inds)
    return image, boxes, classes, is_crowd, area, crop_points

  def _mosaic_crop_image(self, image, boxes, classes, is_crowd, area):
    """Process a patched image in preperation for final output."""
    infos = None
    affine = None

    if self._mosaic_crop_mode == "scale":
      shape = tf.cast(preprocessing_ops.get_image_shape(image), tf.float32)
      center = shape * self._crop_area[0]
      ch = preprocessing_ops.rand_uniform_strong(
          -center[0], center[0], seed=self._seed)
      cw = preprocessing_ops.rand_uniform_strong(
          -center[1], center[1], seed=self._seed)

      image = tfa.image.translate(
          image, [cw, ch], fill_value=preprocessing_ops.PAD_VALUE)
      info = tf.convert_to_tensor(
          [shape, shape,
           tf.ones_like(shape), -tf.cast([ch, cw], tf.float32)])
      infos = [info]

      image, _, affine = preprocessing_ops.affine_warp_image(
          image, [self._output_size[0], self._output_size[1]],
          scale_min=self._crop_area_mosaic[0],
          scale_max=self._crop_area_mosaic[1],
          translate=self._translate,
          seed=self._seed)
      height, width = self._output_size[0], self._output_size[1]
      image = tf.image.resize(image, (height, width))
    else:
      height, width = self._output_size[0], self._output_size[1]
      image = tf.image.resize(image, (height, width))
      image, info = self._crop_image(image, self._crop_area)
      infos = [info]

    # Clip and clean boxes.
    boxes, inds = preprocessing_ops.apply_infos(
        boxes,
        infos,
        affine=affine,
        area_thresh=self._area_thresh,
        seed=self._seed)

    classes = tf.gather(classes, inds)
    is_crowd = tf.gather(is_crowd, inds)
    area = tf.gather(area, inds)
    return image, boxes, classes, is_crowd, area, area

  def translate_boxes(self, box, classes, translate_x, translate_y):
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
      box = bbox_ops.yxyx_to_xcycwh(box)
      x, y, w, h = tf.split(box, 4, axis=-1)
      x = x + translate_x
      y = y + translate_y
      box = tf.cast(tf.concat([x, y, w, h], axis=-1), box.dtype)
      box = bbox_ops.xcycwh_to_yxyx(box)
    return box, classes

  def scale_boxes(self, patch, ishape, boxes, classes, xs, ys):
    """Scale and translate the boxes for each image after patching has been 
    completed"""
    dtype = boxes.dtype
    under_flow = tf.cast(640, tf.float32)
    boxes = tf.cast(boxes, tf.float32) * under_flow
    xs = tf.cast(xs, boxes.dtype)
    ys = tf.cast(ys, boxes.dtype)
    scale = tf.cast(tf.shape(patch) / ishape, boxes.dtype)
    translate = tf.cast((ishape - tf.shape(patch)) / ishape,
                                                   boxes.dtype) * under_flow
    boxes = boxes * tf.cast([scale[0], scale[1], scale[0], scale[1]],
                            boxes.dtype)
    boxes, classes = self.translate_boxes(boxes, classes, translate[1] * xs,
                                             translate[0] * ys)
    boxes = tf.cast(boxes, tf.float32) / under_flow
    return tf.cast(boxes, dtype), classes 

  # mosaic partial frequency
  def _mapped(self, sample):
    """This image patches batches of 4 images together when the input mosaic
    Frequency is lower than 1.0. This allows for speed optimization when the 
    user chooses to use a mixed dataset of both Mosaiced and Regular input 
    samples. For typical frequencies less than about 85% this method can 
    output patched images at about 2.5-4 Samples per second. When set to 100% 
    this rate drops to about 0.9 to 1.0 Samples per second. A non mapped mosiac 
    is there for implemented to optimize for the 100% case."""

    domo = tf.random.uniform([], 0.0, 1.0, dtype=tf.float32, seed=self._seed)
    if self._mosaic_frequency > 0.0 and domo >= (1 - self._mosaic_frequency):
      images = tf.split(sample['image'], 4, axis=0)
      box_list = tf.split(sample['groundtruth_boxes'], 4, axis=0)
      class_list = tf.split(sample['groundtruth_classes'], 4, axis=0)
      is_crowds = tf.split(sample['groundtruth_is_crowd'], 4, axis=0)
      areas = tf.split(sample['groundtruth_area'], 4, axis=0)
      infos = tf.split(sample['info'], 4, axis=0)

      cut, ishape = self._generate_cut()
      shifts = [[1.0, 1.0], [0.0, 1.0], [1.0, 0.0], [0.0, 0.0]]
      for i in range(0, 4):
        images[i] = self._unpad_images(images[i], infos[i])
        (box_list[i], class_list[i], is_crowds[i],
         areas[i]) = self._unpad_gt_comps(box_list[i], class_list[i],
                                          is_crowds[i], areas[i])
        (images[i], box_list[i], class_list[i], is_crowds[i], areas[i],
         _) = self._process_image(images[i], box_list[i], class_list[i],
                                  is_crowds[i], areas[i], shifts[i][0],
                                  shifts[i][1], cut)
        box_list[i], class_list[i] = self.scale_boxes(images[i], ishape,
                                                      box_list[i],
                                                      class_list[i],
                                                      1 - shifts[i][0],
                                                      1 - shifts[i][1])

      patch1 = tf.concat([images[0], images[1]], axis=-2)
      patch2 = tf.concat([images[2], images[3]], axis=-2)
      image = tf.concat([patch1, patch2], axis=-3)

      boxes = tf.concat(box_list, axis=0)
      classes = tf.concat(class_list, axis=0)
      is_crowd = tf.concat(is_crowds, axis=0)
      area = tf.concat(areas, axis=0)

      if self._mosaic_crop_mode is None or self._mosaic_crop_mode == "crop":
        height, width = self._output_size[0], self._output_size[1]
        image = tf.image.resize(image, (height, width))
      else:
        image, boxes, classes, is_crowd, area, info = self._mosaic_crop_image(
            image, boxes, classes, is_crowd, area)

      height, width = preprocessing_ops.get_image_shape(image)
      sample['image'] = tf.expand_dims(image, axis=0)
      sample['source_id'] = tf.expand_dims(sample['source_id'][0], axis=0)
      sample['width'] = tf.expand_dims(
          tf.cast(width, sample['width'].dtype), axis=0)
      sample['height'] = tf.expand_dims(
          tf.cast(height, sample['height'].dtype), axis=0)
      sample['groundtruth_boxes'] = tf.expand_dims(boxes, axis=0)
      sample['groundtruth_classes'] = tf.expand_dims(
          tf.cast(classes, sample['groundtruth_classes'].dtype), axis=0)
      sample['groundtruth_is_crowd'] = tf.expand_dims(
          tf.cast(is_crowd, tf.bool), axis=0)
      sample['groundtruth_area'] = tf.expand_dims(area, axis=0)
      sample['info'] = tf.expand_dims(
          tf.convert_to_tensor([0, 0, height, width]), axis=0)
      sample['num_detections'] = tf.expand_dims(
          tf.shape(sample['groundtruth_boxes'])[1], axis=0)
      sample['is_mosaic'] = tf.expand_dims(tf.cast(1.0, tf.bool), axis=0)

    return sample

  # mosaic full frequency doubles model speed
  def _im_process(self, sample, shiftx, shifty):
    """Distributed processing of each image"""
    (image, boxes, classes, is_crowd, area, crop_points) = self._process_image(
        sample['image'], sample['groundtruth_boxes'],
        sample['groundtruth_classes'], sample['groundtruth_is_crowd'],
        sample['groundtruth_area'], shiftx, shifty, None)

    cut, ishape = self._generate_cut()

    (boxes, classes) = self.scale_boxes(image, ishape, boxes, classes,
                                        1 - shiftx, 1 - shifty)

    sample['image'] = image
    sample['groundtruth_boxes'] = boxes
    sample['groundtruth_classes'] = classes
    sample['groundtruth_is_crowd'] = is_crowd
    sample['groundtruth_area'] = area
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

    if self._mosaic_crop_mode is None or self._mosaic_crop_mode == "crop":
      height, width = self._output_size[0], self._output_size[1]
      image = tf.image.resize(image, (height, width))
    else:
      image, boxes, classes, is_crowd, area, info = self._mosaic_crop_image(
          image, boxes, classes, is_crowd, area)

    sample = one
    height, width = preprocessing_ops.get_image_shape(image)
    sample['image'] = image
    sample['groundtruth_boxes'] = boxes
    sample['groundtruth_area'] = area
    sample['groundtruth_classes'] = tf.cast(classes,
                                            sample['groundtruth_classes'].dtype)
    sample['groundtruth_is_crowd'] = tf.cast(is_crowd, tf.bool)

    sample['width'] = tf.cast(width, sample['width'].dtype)
    sample['height'] = tf.cast(height, sample['height'].dtype)

    sample['num_detections'] = tf.shape(sample['groundtruth_boxes'])[1]
    sample['is_mosaic'] = tf.cast(1.0, tf.bool)
    return sample

  def _full_frequency_apply(self, dataset):
    """This image patches batches of 4 images together when the input mosaic
    Frequency is 1.0 or 100%. This allows for speed optimization when the 
    user chooses to use only Mosaiced input samples. When set to 100% 
    this rate is about 2.0 to 2.5 Samples per second. Mapped and full frequency 
    apply are implemented seperately in order to preserve the integrety of the 
    dataset and to ensure that samples are not skipped or repeated more than 
    expected during training."""

    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    one = dataset.shuffle(10, seed=self._seed, reshuffle_each_iteration=True)
    two = dataset.shuffle(10, seed=self._seed, reshuffle_each_iteration=True)
    three = dataset.shuffle(10, seed=self._seed, reshuffle_each_iteration=True)
    four = dataset.shuffle(10, seed=self._seed, reshuffle_each_iteration=True)

    num = tf.data.AUTOTUNE
    one = one.map(
        lambda x: self._im_process(x, 1.0, 1.0), num_parallel_calls=num)
    two = two.map(
        lambda x: self._im_process(x, 0.0, 1.0), num_parallel_calls=num)
    three = three.map(
        lambda x: self._im_process(x, 1.0, 0.0), num_parallel_calls=num)
    four = four.map(
        lambda x: self._im_process(x, 0.0, 0.0), num_parallel_calls=num)

    patch1 = tf.data.Dataset.zip((one, two)) 
    patch1 = patch1.map(self._patch2, num_parallel_calls=tf.data.AUTOTUNE)
    patch2 = tf.data.Dataset.zip((three, four))
    patch2 = patch2.map(self._patch2, num_parallel_calls=tf.data.AUTOTUNE)

    stitched = tf.data.Dataset.zip((patch1, patch2))
    stitched = stitched.map(self._patch, num_parallel_calls=tf.data.AUTOTUNE)

    if self._mixup_frequency > 0:
      stitched = self._apply_mixup(stitched)
    return stitched

  def _apply(self, dataset):
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    dataset = dataset.map(self._pad_images, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.padded_batch(4)
    dataset = dataset.map(self._mapped, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.unbatch()
    dataset = dataset.map(
        self.resample_unpad, num_parallel_calls=tf.data.AUTOTUNE)
    if self._mixup_frequency > 0:
      dataset = self._apply_mixup(dataset)
    return dataset

  def _mixup(self, one, two):
    domo = tf.random.uniform([], 0.0, 1.0, dtype=tf.float32, seed=self._seed)
    if domo > 0.5:
      sample = one
    else:
      sample = two

    domo = tf.random.uniform([], 0.0, 1.0, dtype=tf.float32, seed=self._seed)
    if domo >= (1 - self._mixup_frequency):
      otype = one["image"].dtype
      r = tf.random.uniform([], 0.4, 0.6, tf.float32, seed=self._seed)
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

  def _apply_mixup(self, dataset):
    one = dataset.shuffle(10, seed=self._seed, reshuffle_each_iteration=True)  #.shard(num_shards=4, index=0)
    two = dataset.shuffle(10, seed=self._seed, reshuffle_each_iteration=True)  #.shard(num_shards=4, index=1)
    mixed = tf.data.Dataset.zip((one, two))  #.prefetch(tf.data.AUTOTUNE)
    mixed = mixed.map(self._mixup, num_parallel_calls=tf.data.AUTOTUNE)
    return mixed

  def _add_param(self, sample):
    sample['is_mosaic'] = tf.cast(0.0, tf.bool)
    sample['num_detections'] = tf.shape(sample['groundtruth_boxes'])[0]
    return sample

  # mosaic skip
  def _no_apply(self, dataset):
    return dataset.map(self._add_param, num_parallel_calls=tf.data.AUTOTUNE)

  def mosaic_fn(self, is_training=True):
    if (is_training and self._mosaic_frequency >= 1.0 and
        self._mosaic_crop_mode != "crop"):
      return self._full_frequency_apply
    elif is_training and self._mosaic_frequency > 0.0:
      return self._apply
    else:
      return self._no_apply

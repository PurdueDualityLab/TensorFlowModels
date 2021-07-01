import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras.backend as K
from yolo.ops import box_ops as bbox_ops
from yolo.ops import preprocessing_ops
from official.vision.beta.ops import box_ops, preprocess_ops

import tensorflow_datasets as tfds
from yolo.dataloaders.decoders import tfds_coco_decoder
from yolo.utils.demos import utils, coco
import matplotlib.pyplot as plt


# TODO: revert back to old mosaic?
class Mosaic(object):

  def __init__(self,
               output_size,
               max_resolution=640,
               mosaic_frequency=1.0,
               random_crop=0.0,
               resize=1.0,
               aspect_ratio_mode='distort',
               aug_scale_min=1.0,
               aug_scale_max=1.0,
               random_flip=True,
               random_pad = False, 
               crop_area=[0.5, 1.0],
               crop_area_mosaic=[0.5, 1.0],
               mosaic_crop_mode=None,
               aug_probability=1.0,
               area_thresh = 0.1, 
               seed=None):

    self._output_size = output_size
    self._max_resolution = max_resolution

    self._mosaic_frequency = mosaic_frequency
    self._aspect_ratio_mode = aspect_ratio_mode
    self._random_crop = random_crop
    self._resize = resize
    self._aug_scale_max = aug_scale_max
    self._aug_scale_min = aug_scale_min
    self._random_flip = random_flip
    self._aug_probability = aug_probability
    self._random_pad = random_pad

    self._crop_area = crop_area
    self._area_thresh = area_thresh

    self._mosaic_crop_mode = mosaic_crop_mode
    self._crop_area_mosaic = crop_area_mosaic

    self._seed = seed
    return

  def _estimate_shape(self, image):
    height, width = preprocessing_ops.get_image_shape(image)
    # oheight, owidth = self._output_size[0], self._output_size[1]
    oheight, owidth = self._max_resolution, self._max_resolution

    if height > width:
      scale = oheight / height
      nwidth = tf.cast(scale * tf.cast(width, scale.dtype), width.dtype)
      if nwidth > owidth:
        scale = owidth / width
        oheight = tf.cast(scale * tf.cast(height, scale.dtype), height.dtype)
      else:
        owidth = nwidth
    else:
      scale = owidth / width
      nheight = tf.cast(scale * tf.cast(height, scale.dtype), height.dtype)
      if nheight > oheight:
        scale = oheight / height
        owidth = tf.cast(scale * tf.cast(width, scale.dtype), width.dtype)
      else:
        oheight = nheight

    if height < oheight and width < owidth:
      oheight = height
      owidth = width
    return oheight, owidth

  def _letter_box(self, image, boxes=None, xs=0.0, ys=0.0):
    height, width = self._estimate_shape(image)
    hclipper, wclipper = self._max_resolution, self._max_resolution

    xs = tf.convert_to_tensor(xs)
    ys = tf.convert_to_tensor(ys)
    pad_width_p = wclipper - width
    pad_height_p = hclipper - height
    pad_height = tf.cast(tf.cast(pad_height_p, ys.dtype) * ys, tf.int32)
    pad_width = tf.cast(tf.cast(pad_width_p, xs.dtype) * xs, tf.int32)

    image = tf.image.resize(image, (height, width))
    image = tf.image.pad_to_bounding_box(image, pad_height, pad_width, hclipper,
                                         wclipper)
    info = tf.convert_to_tensor([pad_height, pad_width, height, width])
    return image, boxes, info

  def _pad_images(self, sample):
    image = sample['image']
    image, _, info = self._letter_box(image, xs=0.0, ys=0.0)
    sample['image'] = image
    sample['info'] = info
    sample['num_detections'] = tf.shape(sample['groundtruth_boxes'])[0]
    sample['is_mosaic'] = tf.cast(0.0, tf.bool)
    return sample

  def _unpad_images(self, image, info, squeeze=True):
    if squeeze:
      image = tf.squeeze(image, axis=0)
      info = tf.squeeze(info, axis=0)
    image = tf.slice(image, [info[0], info[1], 0], [info[2], info[3], -1])
    return image

  def _unpad_gt_comps(self, boxes, classes, is_crowd, area, squeeze=True):
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

  def _mosaic_crop(self, image, crop_area, skip_zero = True):
    height, width = preprocessing_ops.get_image_shape(image)
    scale = preprocessing_ops.rand_uniform_strong(tf.math.sqrt(crop_area[0]), 
                                                  tf.math.sqrt(crop_area[1]))
    width = tf.cast(tf.cast(width, scale.dtype) * scale, tf.int32)
    height = tf.cast(tf.cast(height, scale.dtype) * scale, tf.int32)

    # tf.print(width, height, scale)
    image, info = preprocessing_ops.random_window_crop(
      image, 
      height, 
      width, 
      skip_zero=skip_zero
    )

    infos = [info]
    return image, infos

  def _crop_image(self, image, crop_area):
    image, info = preprocessing_ops.random_crop_image(
          image, 
          aspect_ratio_range=[self._output_size[1]/self._output_size[0], 
                              self._output_size[1]/self._output_size[0]], 
          area_range=crop_area)

    return image, info

  def _process_image(self,
                     image,
                     boxes,
                     classes,
                     is_crowd,
                     area,
                     xs=0.0,
                     ys=0.0, 
                     cut = None, 
                     scale = 1):
    # initialize the shape constants
    height, width = preprocessing_ops.get_image_shape(image)
    infos = []

    if self._random_flip:
      # randomly flip the image horizontally
      image, boxes, _ = preprocess_ops.random_horizontal_flip(image, boxes)

    # resize the image irrespective of the aspect ratio
    random_crop = self._random_crop
    if self._aspect_ratio_mode == 'distort':
      letter_box = False
    elif self._aspect_ratio_mode == 'crop':
      shape = tf.shape(image)
      info = tf.stack([
          tf.cast(tf.shape(image)[:2], tf.float32),
          tf.cast(tf.shape(image)[:2], tf.float32),
          tf.ones_like(tf.cast(shape[:2], tf.float32)),
          tf.zeros_like(tf.cast(shape[:2], tf.float32)),
          ])
      
      docrop = tf.random.uniform([],
                                 0.0,
                                 1.0,
                                 dtype=tf.float32,
                                 seed=self._seed)
      if docrop > 1 - self._random_crop:
        image, info = self._crop_image(image, self._crop_area)
      
      infos.append(info)
      random_crop = 0.0
      letter_box = True
    else:
      letter_box = True

    height, width = self._output_size[0] // scale, self._output_size[1] // scale
    if random_crop != 0:
      image, infos_ = preprocessing_ops.resize_and_jitter_image(
          image,
          [height, width],
          letter_box=letter_box,
          jitter=random_crop,
          random_pad=False,
          resize = self._resize,
          shiftx=xs,
          shifty=ys,
          cut = cut)
    else:
      image, infos_ = preprocessing_ops.resize_and_crop_image(
          image,
          [height, width],
          [height, width],
          letter_box=letter_box,
          aug_scale_min=1,
          aug_scale_max=1,
          shiftx=xs,
          shifty=ys,
          random_pad=False)
    infos.extend(infos_)

    # clip and clean boxes
    boxes, inds = preprocessing_ops.apply_infos(boxes, infos, area_thresh = self._area_thresh)
    classes = tf.gather(classes, inds)
    is_crowd = tf.gather(is_crowd, inds)
    area = tf.gather(area, inds)
    return image, boxes, classes, is_crowd, area

  def _mosaic_crop_image(self, image, boxes, classes, is_crowd, area):
 
    if self._mosaic_crop_mode == "scale":
      height, width = self._output_size[0], self._output_size[1]
      image = tf.image.resize(image, (height, width))
      image, infos = preprocessing_ops.resize_and_crop_image(
          image, [self._output_size[0], self._output_size[1]],
          [self._output_size[0], self._output_size[1]],
          letter_box=None,
          aug_scale_min=self._crop_area_mosaic[0],
          aug_scale_max=self._crop_area_mosaic[1], 
          random_pad=self._random_pad,
          seed=self._seed)
      # height, width = self._output_size[0], self._output_size[1]
      # image = tf.image.resize(image, (height, width))
    elif self._mosaic_crop_mode == 'crop_scale':
      height, width = preprocessing_ops.get_image_shape(image)
      image, infos = self._mosaic_crop(image, self._crop_area)
      image, infos_ = preprocessing_ops.resize_and_crop_image(
          image, 
          [height, width],
          [height, width],
          letter_box=None,
          aug_scale_min=self._crop_area_mosaic[0],
          aug_scale_max=self._crop_area_mosaic[1],
          random_pad=self._random_pad,
          seed=self._seed)
      infos.extend(infos_)
      height, width = self._output_size[0], self._output_size[1]
      image = tf.image.resize(image, (height, width))
    else:
      height, width = self._output_size[0], self._output_size[1]
      image = tf.image.resize(image, (height, width))
      image, infos = self._mosaic_crop(image, self._crop_area, skip_zero = False)

    # clip and clean boxes
    boxes, inds = preprocessing_ops.apply_infos(boxes, infos, area_thresh = self._area_thresh)
    classes = tf.gather(classes, inds)
    is_crowd = tf.gather(is_crowd, inds)
    area = tf.gather(area, inds)
    return image, boxes, classes, is_crowd, area, infos[-1]

  def scale_boxes(self, patch, ishape, boxes, classes, xs, ys):
    xs = tf.cast(xs, boxes.dtype)
    ys = tf.cast(ys, boxes.dtype)
    scale = tf.cast(tf.shape(patch)/ishape, boxes.dtype)
    translate = tf.cast((ishape - tf.shape(patch))/ishape, boxes.dtype)
    boxes = boxes * tf.cast([scale[0], 
                             scale[1], 
                             scale[0], 
                             scale[1]], boxes.dtype)
    return preprocessing_ops.translate_boxes(boxes, 
                                             classes, 
                                             translate[1] * xs, 
                                             translate[0] * ys)

  def _mapped(self, sample):
    if self._mosaic_frequency > 0.0:
      # mosaic is enabled 0.666667 = 0.5 of images are mosaiced
      domo = tf.random.uniform([], 0.0, 1.0, dtype=tf.float32, seed=self._seed)
      if domo >= (1 - self._mosaic_frequency):
        image = sample['image']
        boxes = sample['groundtruth_boxes']
        classes = sample['groundtruth_classes']
        is_crowd = sample['groundtruth_is_crowd']
        area = sample['groundtruth_area']
        info = sample['info']

        if self._mosaic_crop_mode == 'crop':
          min_offset = 0.2
          cut_x = preprocessing_ops.rand_uniform_strong(
            self._output_size[1] * min_offset, 
            self._output_size[1] * (1 - min_offset)
          )
          cut_y = preprocessing_ops.rand_uniform_strong(
            self._output_size[1] * min_offset, 
            self._output_size[1] * (1 - min_offset)
          )
          cut = [cut_x, cut_y]
          ishape = tf.convert_to_tensor([self._output_size[1], 
                                         self._output_size[0], 3])
          scale = 1
        elif self._mosaic_crop_mode == 'scale':
          cut = None
          ishape = tf.convert_to_tensor([self._output_size[1], 
                                         self._output_size[0], 3])
          scale = 2
        else:
          cut = None
          ishape = tf.convert_to_tensor([self._output_size[1] * 2, 
                                         self._output_size[0] * 2, 3])
          scale = 1

        images = tf.split(image, 4, axis=0)
        box_list = tf.split(boxes, 4, axis=0)
        class_list = tf.split(classes, 4, axis=0)
        is_crowds = tf.split(is_crowd, 4, axis=0)
        areas = tf.split(area, 4, axis=0)
        infos = tf.split(info, 4, axis=0)

        shifts = [[1.0, 1.0], [0.0, 1.0], [1.0, 0.0], [0.0, 0.0]]
        for i in range(0, 4):
          images[i] = self._unpad_images(images[i], infos[i])
          (box_list[i], class_list[i], is_crowds[i],
          areas[i]) = self._unpad_gt_comps(box_list[i], class_list[i],
                                            is_crowds[i], areas[i])
          (images[i], box_list[i], class_list[i], is_crowds[i],
          areas[i]) = self._process_image(images[i], box_list[i], class_list[i],
                                          is_crowds[i], areas[i], shifts[i][0], 
                                          shifts[i][1], cut)
          box_list[i], class_list[i] = self.scale_boxes(
            images[i], ishape, box_list[i], class_list[i], 
            1 - shifts[i][0], 1 - shifts[i][1]
          )
        

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
        sample['width'] = tf.cast(
            tf.expand_dims(width, axis=0), sample['width'].dtype)
        sample['height'] = tf.cast(
            tf.expand_dims(height, axis=0), sample['height'].dtype)
        sample['groundtruth_boxes'] = tf.expand_dims(boxes, axis=0)
        sample['groundtruth_classes'] = tf.cast(
            tf.expand_dims(classes, axis=0),
            sample['groundtruth_classes'].dtype)
        sample['groundtruth_is_crowd'] = tf.cast(
            tf.expand_dims(is_crowd, axis=0), tf.bool)
        sample['groundtruth_area'] = tf.expand_dims(area, axis=0)
        sample['info'] = tf.expand_dims(
            tf.convert_to_tensor([0, 0, height, width]), axis=0)
        sample['num_detections'] = tf.expand_dims(
            tf.shape(sample['groundtruth_boxes'])[1], axis=0)
        
        if self._mosaic_crop_mode == 'pre_crop':
          sample['is_mosaic'] = tf.expand_dims(tf.cast(0.0, tf.bool), axis=0)
        else:
          sample['is_mosaic'] = tf.expand_dims(tf.cast(1.0, tf.bool), axis=0)
    return sample

  def _full_frequency_mosaic(self, sample):
    image = sample['image']
    boxes = sample['groundtruth_boxes']
    classes = sample['groundtruth_classes']
    is_crowd = sample['groundtruth_is_crowd']
    area = sample['groundtruth_area']
    info = sample['info']

    if self._mosaic_crop_mode == 'crop':
      min_offset = 0.2
      cut_x = preprocessing_ops.rand_uniform_strong(
        self._output_size[1] * min_offset, 
        self._output_size[1] * (1 - min_offset)
      )
      cut_y = preprocessing_ops.rand_uniform_strong(
        self._output_size[1] * min_offset, 
        self._output_size[1] * (1 - min_offset)
      )
      cut = [cut_x, cut_y]
      ishape = tf.convert_to_tensor([self._output_size[1], 
                                      self._output_size[0], 3])
      scale = 1
    elif self._mosaic_crop_mode == 'scale':
      cut = None
      ishape = tf.convert_to_tensor([self._output_size[1], 
                                      self._output_size[0], 3])
      scale = 2
    else:
      cut = None
      ishape = tf.convert_to_tensor([self._output_size[1] * 2, 
                                      self._output_size[0] * 2, 3])
      scale = 1

    image__ = tf.TensorArray(image.dtype, size=0, dynamic_size=True)
    image__ = image__.unstack(image)
    tf.print

    images = tf.split(image, 4, axis=0)
    box_list = tf.split(boxes, 4, axis=0)
    class_list = tf.split(classes, 4, axis=0)
    is_crowds = tf.split(is_crowd, 4, axis=0)
    areas = tf.split(area, 4, axis=0)
    infos = tf.split(info, 4, axis=0)

    shifts = [[1.0, 1.0], [0.0, 1.0], [1.0, 0.0], [0.0, 0.0]]
    for i in range(0, 4):
      images[i] = self._unpad_images(images[i], infos[i])
      (box_list[i], class_list[i], is_crowds[i],
      areas[i]) = self._unpad_gt_comps(box_list[i], class_list[i],
                                        is_crowds[i], areas[i])
      (images[i], box_list[i], class_list[i], is_crowds[i],
      areas[i]) = self._process_image(images[i], box_list[i], class_list[i],
                                      is_crowds[i], areas[i], shifts[i][0], 
                                      shifts[i][1], cut)
      box_list[i], class_list[i] = self.scale_boxes(
        images[i], ishape, box_list[i], class_list[i], 
        1 - shifts[i][0], 1 - shifts[i][1]
      )
    

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
    sample['width'] = tf.cast(
        tf.expand_dims(width, axis=0), sample['width'].dtype)
    sample['height'] = tf.cast(
        tf.expand_dims(height, axis=0), sample['height'].dtype)
    sample['groundtruth_boxes'] = tf.expand_dims(boxes, axis=0)
    sample['groundtruth_classes'] = tf.cast(
        tf.expand_dims(classes, axis=0),
        sample['groundtruth_classes'].dtype)
    sample['groundtruth_is_crowd'] = tf.cast(
        tf.expand_dims(is_crowd, axis=0), tf.bool)
    sample['groundtruth_area'] = tf.expand_dims(area, axis=0)
    sample['info'] = tf.expand_dims(
        tf.convert_to_tensor([0, 0, height, width]), axis=0)
    sample['num_detections'] = tf.expand_dims(
        tf.shape(sample['groundtruth_boxes'])[1], axis=0)
    
    #if self._mosaic_crop_mode == 'pre_crop':
    #sample['is_mosaic'] = tf.expand_dims(tf.cast(0.0, tf.bool), axis=0)
    
    sample['is_mosaic'] = tf.expand_dims(tf.cast(1.0, tf.bool), axis=0)
    return sample

  def resample_unpad(self, sample):
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

  def _add_param(self, sample):
    sample['is_mosaic'] = tf.cast(0.0, tf.bool)
    sample['num_detections'] = tf.shape(sample['groundtruth_boxes'])[0]
    return sample

  def _im_process(self, sample, shiftx, shifty):
    image = sample['image']
    boxes = sample['groundtruth_boxes']
    classes = sample['groundtruth_classes']
    is_crowd = sample['groundtruth_is_crowd']
    area = sample['groundtruth_area']

    image, boxes, classes, is_crowd, area = self._process_image(image, boxes, classes,
                                          is_crowd, area, shiftx, shifty, None)

    sample['image'] = image
    sample['groundtruth_boxes'] = boxes
    sample['groundtruth_classes'] = classes
    sample['groundtruth_is_crowd'] = is_crowd
    sample['groundtruth_area'] = area
    sample['shiftx'] = shiftx
    sample['shifty'] = shifty
    return sample

  def _map_concat(self, one, two, three, four):
    patch1 = tf.concat([one["image"], two["image"]], axis=-2)
    patch2 = tf.concat([three["image"], four["image"]], axis=-2)
    image = tf.concat([patch1, patch2], axis=-3)

    one_boxes, one_classes = self.scale_boxes(
      one["image"], tf.shape(image), one['groundtruth_boxes'], 
      one['groundtruth_classes'], 1 - one["shiftx"], 1 - one["shifty"]
    )

    two_boxes, two_classes = self.scale_boxes(
      two["image"], tf.shape(image), two['groundtruth_boxes'], 
      two['groundtruth_classes'], 1 - two["shiftx"], 1 - two["shifty"]
    )

    three_boxes, three_classes = self.scale_boxes(
      three["image"], tf.shape(image), three['groundtruth_boxes'], 
      three['groundtruth_classes'], 1 - three["shiftx"], 1 - three["shifty"]
    )

    four_boxes, four_classes = self.scale_boxes(
      four["image"], tf.shape(image), four['groundtruth_boxes'], 
      four['groundtruth_classes'], 1 - four["shiftx"], 1 - four["shifty"]
    )

    boxes = tf.concat([one_boxes, two_boxes, 
                       three_boxes, four_boxes] , axis=0)
    classes = tf.concat([one_classes, two_classes, 
                       three_classes, four_classes] , axis=0)

    is_crowd = tf.concat([one['groundtruth_is_crowd'], two['groundtruth_is_crowd'], 
                         three['groundtruth_is_crowd'], four['groundtruth_is_crowd']], axis=0)

    area = tf.concat([one['groundtruth_area'], two['groundtruth_area'], 
                         three['groundtruth_area'], four['groundtruth_area']], axis=0)

    if self._mosaic_crop_mode is None:
      height, width = self._output_size[0], self._output_size[1]
      image = tf.image.resize(image, (height, width))
    else:
      image, boxes, classes, is_crowd, area, info = self._mosaic_crop_image(
          image, boxes, classes, is_crowd, area)

    height, width = preprocessing_ops.get_image_shape(image)
    sample = one
    sample['image'] = image
    sample['source_id'] = sample['source_id']
    sample['width'] = tf.cast(width, sample['width'].dtype)
    sample['height'] = tf.cast(height, sample['height'].dtype)
    sample['groundtruth_boxes'] = boxes
    sample['groundtruth_classes'] = tf.cast(classes,sample['groundtruth_classes'].dtype)
    sample['groundtruth_is_crowd'] = tf.cast(is_crowd, tf.bool)
    sample['groundtruth_area'] = area
    sample['info'] = tf.convert_to_tensor([0, 0, height, width])
    sample['num_detections'] = tf.shape(sample['groundtruth_boxes'])[1]
    
    # if self._mosaic_crop_mode == 'pre_crop':
    #   sample['is_mosaic'] = tf.cast(0.0, tf.bool)
    # else:
    sample['is_mosaic'] = tf.expand_dims(tf.cast(1.0, tf.bool), axis=0)
    return sample
   
  def _full_frequency_apply(self, dataset):
    one = dataset.shard(num_shards=4, index=0)
    two = dataset.shard(num_shards=4, index=1)
    three = dataset.shard(num_shards=4, index=2)
    four = dataset.shard(num_shards=4, index=3)

    one = one.map(lambda x: self._im_process(x, 1.0, 1.0), num_parallel_calls=tf.data.AUTOTUNE, deterministic = False).prefetch(tf.data.AUTOTUNE)
    two = two.map(lambda x: self._im_process(x, 0.0, 1.0), num_parallel_calls=tf.data.AUTOTUNE, deterministic = False).prefetch(tf.data.AUTOTUNE)
    three = three.map(lambda x: self._im_process(x, 1.0, 0.0), num_parallel_calls=tf.data.AUTOTUNE, deterministic = False).prefetch(tf.data.AUTOTUNE)
    four = four.map(lambda x: self._im_process(x, 0.0, 0.0), num_parallel_calls=tf.data.AUTOTUNE, deterministic = False).prefetch(tf.data.AUTOTUNE)

    stitched = tf.data.Dataset.zip((one, two, three, four))
    stitched = stitched.map(self._map_concat, num_parallel_calls=tf.data.AUTOTUNE)
    return stitched

  def _apply(self, dataset):
    dataset = dataset.map(self._pad_images, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    dataset = dataset.padded_batch(4)
    dataset = dataset.map(self._mapped, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    dataset = dataset.unbatch()
    dataset = dataset.map(
        self.resample_unpad, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    return dataset

  def _no_apply(self, dataset):
    return dataset.map(self._add_param, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

  def mosaic_fn(self, is_training=True):
    if is_training and self._mosaic_frequency == 1.0:
      return self._full_frequency_apply
    elif is_training and self._mosaic_frequency > 0.0:
      return self._apply
    else:
      return self._no_apply

  @staticmethod
  def steps_per_epoch(steps_per_epoch, mosaic_frequency):
    # 0.667 = 50 % of the images
    steps = (steps_per_epoch /
             4) * (mosaic_frequency) + steps_per_epoch * (1 - mosaic_frequency)
    steps = tf.math.ceil(steps)
    return steps


if __name__ == "__main__":
  drawer = utils.DrawBoxes(labels=coco.get_coco_names(), thickness=2)
  decoder = tfds_coco_decoder.MSCOCODecoder()
  mosaic = Mosaic([640, 640],
                  random_crop=True,
                  random_crop_mosaic=True,
                  crop_area_mosaic=[0.25, 0.75])

  dataset = tfds.load('coco', split='train')
  dataset = dataset.map(decoder.decode)
  dataset = dataset.apply(mosaic.mosaic_fn(is_training=True))
  dataset = dataset.take(30).cache()

  import time

  a = time.time()
  for image in dataset:
    im = image['image'] / 255
    boxes = image['groundtruth_boxes']
    classes = image['groundtruth_classes']
    confidence = image['groundtruth_classes']

    draw_dict = {
        'bbox': boxes,
        'classes': classes,
        'confidence': confidence,
    }

    im = tf.image.draw_bounding_boxes(
        tf.expand_dims(im, axis=0), tf.expand_dims(boxes, axis=0),
        [[1.0, 0.0, 1.0]])
    plt.imshow(im[0])
    plt.show()
  b = time.time()

  print(b - a)

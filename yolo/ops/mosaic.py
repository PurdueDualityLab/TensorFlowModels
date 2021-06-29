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

  def _crop_image(self, image, boxes, classes, is_crowd, area, crop_area):

    # height, width = self._output_size[1], self._output_size[0]
    # scale = preprocessing_ops.rand_uniform_strong(tf.math.sqrt(crop_area[0]), 
    #                                               tf.math.sqrt(crop_area[1]))
    # width = tf.cast(tf.cast(width, scale.dtype) * scale, tf.int32)
    # height = tf.cast(tf.cast(height, scale.dtype) * scale, tf.int32)

    # # tf.print(width, height, scale)
    # image, info = preprocessing_ops.random_window_crop(
    #   image, 
    #   height, 
    #   width, 
    #   skip_zero=False
    # )

    # infos = [info]

    # image = tf.image.resize(image, (self._output_size[1], 
    #                                 self._output_size[0]))

    image, info = preprocessing_ops.random_crop_image(
          image, 
          aspect_ratio_range=[self._output_size[1]/self._output_size[0], 
                              self._output_size[1]/self._output_size[0]], 
          area_range=crop_area)

    infos = [info]
    boxes, inds = preprocessing_ops.apply_infos(boxes, infos, area_thresh = self._area_thresh)
    classes = tf.gather(classes, inds)
    is_crowd = tf.gather(is_crowd, inds)
    area = tf.gather(area, inds)
    return image, boxes, classes, is_crowd, area

  def _process_image(self,
                     image,
                     boxes,
                     classes,
                     is_crowd,
                     area,
                     xs=0.0,
                     ys=0.0, 
                     cut = None):
    # initialize the shape constants
    height, width = preprocessing_ops.get_image_shape(image)

    # resize the image irrespective of the aspect ratio
    random_crop = self._random_crop
    if self._aspect_ratio_mode == 'distort':
      letter_box = False
    elif self._aspect_ratio_mode == 'crop':
      docrop = tf.random.uniform([],
                                 0.0,
                                 1.0,
                                 dtype=tf.float32,
                                 seed=self._seed)
      if docrop > 1 - self._random_crop:
        image, boxes, classes, is_crowd, area = self._crop_image(
            image, boxes, classes, is_crowd, area, self._crop_area)
      random_crop = 0.1
      letter_box = True
    else:
      letter_box = True

    if self._random_flip:
      # randomly flip the image horizontally
      image, boxes, _ = preprocess_ops.random_horizontal_flip(image, boxes)


    if self._random_crop != 0:
      image, infos = preprocessing_ops.resize_and_jitter_image(
          image,
          [self._output_size[0], self._output_size[1]],
          letter_box=letter_box,
          jitter=random_crop,
          random_pad=False,
          resize = self._resize,
          shiftx=xs,
          shifty=ys,
          cut = cut)
    else:
      image, infos = preprocessing_ops.resize_and_crop_image(
          image,
          [self._output_size[0], self._output_size[1]],
          [self._output_size[0], self._output_size[1]],
          letter_box=self._letter_box,
          aug_scale_min=self._aug_scale_min,
          aug_scale_max=self._aug_scale_max,
          shiftx=xs,
          shifty=ys,
          random_pad=False)

    # clip and clean boxes
    boxes, inds = preprocessing_ops.apply_infos(boxes, infos, area_thresh = self._area_thresh)
    classes = tf.gather(classes, inds)
    is_crowd = tf.gather(is_crowd, inds)
    area = tf.gather(area, inds)
    return image, boxes, classes, is_crowd, area

  def _mosaic_crop_image(self, image, boxes, classes, is_crowd, area):
 
    if self._mosaic_crop_mode == "scale":
      image, infos = preprocessing_ops.resize_and_crop_image(
          image, [self._output_size[0] * 2, self._output_size[1] * 2],
          [self._output_size[0] * 2, self._output_size[1] * 2],
          letter_box=None,
          aug_scale_min=self._crop_area_mosaic[0],
          aug_scale_max=self._crop_area_mosaic[1], 
          random_pad=self._random_pad,
          seed=self._seed)
      height, width = self._output_size[0], self._output_size[1]
      image = tf.image.resize(image, (height, width))
    elif self._mosaic_crop_mode == 'crop_scale':
      image, infos = self._mosaic_crop(image, self._crop_area)
      image, infos_ = preprocessing_ops.resize_and_crop_image(
          image, [self._output_size[0] * 2, self._output_size[1] * 2],
          [self._output_size[0] * 2, self._output_size[1] * 2],
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

  def scale_boxes(self, patch, image, boxes, classes, xs, ys):
    xs = tf.cast(xs, boxes.dtype)
    ys = tf.cast(ys, boxes.dtype)
    scale = tf.cast(tf.shape(patch)/tf.shape(image), boxes.dtype)
    translate = tf.cast((tf.shape(image) - tf.shape(patch))/tf.shape(image)
                                    , boxes.dtype)
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

        images = tf.split(image, 4, axis=0)
        box_list = tf.split(boxes, 4, axis=0)
        class_list = tf.split(classes, 4, axis=0)
        is_crowds = tf.split(is_crowd, 4, axis=0)
        areas = tf.split(area, 4, axis=0)
        infos = tf.split(info, 4, axis=0)

        images[0] = self._unpad_images(images[0], infos[0])
        images[1] = self._unpad_images(images[1], infos[1])
        images[2] = self._unpad_images(images[2], infos[2])
        images[3] = self._unpad_images(images[3], infos[3])

        (box_list[0], class_list[0], is_crowds[0],
         areas[0]) = self._unpad_gt_comps(box_list[0], class_list[0],
                                          is_crowds[0], areas[0])
        (box_list[1], class_list[1], is_crowds[1],
         areas[1]) = self._unpad_gt_comps(box_list[1], class_list[1],
                                          is_crowds[1], areas[1])

        (box_list[2], class_list[2], is_crowds[2],
         areas[2]) = self._unpad_gt_comps(box_list[2], class_list[2],
                                          is_crowds[2], areas[2])

        (box_list[3], class_list[3], is_crowds[3],
         areas[3]) = self._unpad_gt_comps(box_list[3], class_list[3],
                                          is_crowds[3], areas[3])

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
        else:
          cut = None

        (images[0], box_list[0], class_list[0], is_crowds[0],
         areas[0]) = self._process_image(images[0], box_list[0], class_list[0],
                                         is_crowds[0], areas[0], 1.0, 1.0, cut)

        (images[1], box_list[1], class_list[1], is_crowds[1],
         areas[1]) = self._process_image(images[1], box_list[1], class_list[1],
                                         is_crowds[1], areas[1], 0.0, 1.0, cut)

        (images[2], box_list[2], class_list[2], is_crowds[2],
         areas[2]) = self._process_image(images[2], box_list[2], class_list[2],
                                         is_crowds[2], areas[2], 1.0, 0.0, cut)

        (images[3], box_list[3], class_list[3], is_crowds[3],
         areas[3]) = self._process_image(images[3], box_list[3], class_list[3],
                                         is_crowds[3], areas[3], 0.0, 0.0, cut)

        patch1 = tf.concat([images[0], images[1]], axis=-2)
        patch2 = tf.concat([images[2], images[3]], axis=-2)
        image = tf.concat([patch1, patch2], axis=-3)


        box_list[0], class_list[0] = self.scale_boxes(
          images[0], image, box_list[0], class_list[0], 0.0, 0.0
        )
        box_list[1], class_list[1] = self.scale_boxes(
          images[1], image, box_list[1], class_list[1], 1.0, 0.0
        )
        box_list[2], class_list[2] = self.scale_boxes(
          images[2], image, box_list[2], class_list[2], 0.0, 1.0
        )
        box_list[3], class_list[3] = self.scale_boxes(
          images[3], image, box_list[3], class_list[3], 1.0, 1.0
        )

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

  def _apply(self, dataset):
    dataset = dataset.map(self._pad_images, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.padded_batch(4)
    dataset = dataset.map(self._mapped, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.unbatch()
    dataset = dataset.map(
        self.resample_unpad, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset

  def _no_apply(self, dataset):
    return dataset.map(self._add_param, num_parallel_calls=tf.data.AUTOTUNE)

  def mosaic_fn(self, is_training=True):
    if is_training and self._mosaic_frequency > 0.0:
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

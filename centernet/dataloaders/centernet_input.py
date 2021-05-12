import time

import tensorflow as tf

from centernet.ops import preprocessing_ops
from official.vision.beta.dataloaders import parser, utils
from official.vision.beta.ops.preprocess_ops import normalize_image
from yolo.ops import preprocessing_ops as yolo_preprocessing_ops

from typing import List


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

class CenterNetParser(parser.Parser):
  """ Parser to parse an image and its annotations into a dictionary of tensors """

  def __init__(self,
               image_w: int = 512,
               image_h: int = 512,
               max_num_instances: int = 128,
               bgr_ordering: bool = True,
               channel_means: List[int] = [104.01362025, 114.03422265, 119.9165958],
               channel_stds: List[int] = [73.6027665, 69.89082075, 70.9150767],
               dtype: str = 'float32'):
    """Initializes parameters for parsing annotations in the dataset.
    Args:
      image_w: A `Tensor` or `int` for width of input image.
      image_h: A `Tensor` or `int` for height of input image.
      num_classes: A `Tensor` or `int` for the number of classes.
      max_num_instances: An `int` number of maximum number of instances in an image.
      use_gaussian_bump: A `boolean` indicating whether or not to splat a
        gaussian onto the heatmaps. If set to False, a value of 1 is placed at 
        the would-be center of the gaussian.
      gaussian_rad: A `int` for the desired radius of the gaussian. If this
        value is set to -1, then the radius is computed using gaussian_iou. 
      gaussian_iou: A `float` number for the minimum desired IOU used when
        determining the gaussian radius of center locations in the heatmap.
      output_dims: A `Tensor` or `int` for output dimensions of the heatmap.
    """
    self._image_w = image_w
    self._image_h = image_h
    self._max_num_instances = max_num_instances
    self._bgr_ordering = bgr_ordering
    self._channel_means = channel_means
    self._channel_stds = channel_stds

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

  def _parse_train_data(self, data):
    """Generates images and labels that are usable for model training.

    Args:
        data: a dict of Tensors produced by the decoder.

    Returns:
        images: the image tensor.
        labels: a dict of Tensors that contains labels.
    """
    # FIXME: This is a copy of parse eval data
    image = data['image'] / 255
    boxes = data['groundtruth_boxes']
    classes = data['groundtruth_classes']

    image, boxes, info = yolo_preprocessing_ops.letter_box(
      image, boxes, xs = 0.5, ys = 0.5, target_dim=self._image_w)

    image = tf.cast(image, self._dtype)
    shape = tf.shape(image)
    height = shape[0]
    width = shape[1]

    image, labels = self._build_label(
      image, boxes, classes, width, height, info, data, is_training=False
    )
    
    return image, labels

  def _parse_eval_data(self, data):
    """Generates images and labels that are usable for model evaluation.

    Args:
      decoded_tensors: a dict of Tensors produced by the decoder.

    Returns:
      images: the image tensor.
      labels: a dict of Tensors that contains labels.
    """
    image = tf.cast(data['image'], dtype=tf.float32)
    
    if self._bgr_ordering:
      red, green, blue = tf.unstack(image, num=3, axis=2)
      image = tf.stack([blue, green, red], axis=2)

    image = normalize_image(image, offset=self._channel_means, scale=self._channel_stds)

    boxes = data['groundtruth_boxes']
    classes = data['groundtruth_classes']

    image, boxes, info = yolo_preprocessing_ops.letter_box(
      image, boxes, xs = 0.5, ys = 0.5, target_dim=self._image_w)

    image = tf.cast(image, self._dtype)
    shape = tf.shape(image)
    height = shape[0]
    width = shape[1]

    image, labels = self._build_label(
      image, boxes, classes, width, height, info, data, is_training=False
    )
    
    return image, labels

  def _build_label(self, image, boxes, classes, width, height, info, data, 
                   is_training):
    imshape = image.get_shape().as_list()
    imshape[-1] = 3
    image.set_shape(imshape)

    bshape = boxes.get_shape().as_list()
    boxes = pad_max_instances(boxes, self._max_num_instances, 0)
    bshape[0] = self._max_num_instances
    boxes.set_shape(bshape)

    cshape = classes.get_shape().as_list()
    classes = pad_max_instances(classes,
                                self._max_num_instances, -1)
    cshape[0] = self._max_num_instances
    classes.set_shape(cshape)

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

    num_detections = tf.shape(data['groundtruth_classes'])[0]
    labels = {
      'source_id': utils.process_source_id(data['source_id']),
      'bbox': tf.cast(boxes, self._dtype),
      'classes': tf.cast(classes, self._dtype),
      'area': tf.cast(area, self._dtype),
      'is_crowd': is_crowd,
      'width': width,
      'height': height,
      'info': info, 
      'num_detections': num_detections
    }

    return image, labels

  def postprocess_fn(self, is_training):
    if is_training:  #or self._cutmix
      return None # if not self._fixed_size or self._mosaic else None
    else:
      return None


if __name__ == '__main__':
  # This code is for visualization (outdated since we moved ground truth label
  # builder to the loss function)
  import matplotlib.pyplot as plt


  resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='node-8')
  tf.config.experimental_connect_to_cluster(resolver)
  tf.tpu.experimental.initialize_tpu_system(resolver)
  print("All devices: ", tf.config.list_logical_devices('TPU'))

  boxes = tf.constant([
    (10, 300, 15, 370), # center (y, x) = (12, 335)
    (100, 300, 150, 370), # center (y, x) = (125, 335)
    (15, 100, 200, 170), # center (y, x) = (107, 135)
  ], dtype=tf.float32)

  classes = tf.constant((1, 1, 1), dtype=tf.float32)

  boxes = pad_max_instances(boxes, 128, 0)
  classes = pad_max_instances(classes, 128, -1)

  parser = CenterNetParser()

  # print("testing original build heatmaps function: ")
  # a = time.time()
  # with tf.device('/TPU:0'):
  #   labels1 = parser._build_heatmap_and_regressed_features1(
  #     labels = {
  #       'bbox': boxes,
  #       'num_detections': 3,
  #       'classes': classes
  #     },
  #     output_size=[512, 512], input_size=[512, 512]
  #   )
  # b = time.time()
  # print("Time taken: {} ms", (b-a) * 1000)

  print("testing new build heatmaps function: ")
  a = time.time()
  with tf.device('/TPU:0'):
    labels = parser._build_heatmap_and_regressed_features(
       labels = {
       'bbox': boxes,
       'num_detections': 3,
       'classes': classes
      },
     output_size=[512, 512], input_size=[512, 512]
     )
  b = time.time()
  print("Time taken: {} ms", (b-a) * 1000)

  # tl_heatmaps = labels['tl_heatmaps']
  # br_heatmaps = labels['br_heatmaps']
  # ct_heatmaps = labels['ct_heatmaps']

  # plt.imshow(ct_heatmaps[..., 0])
  # plt.show()

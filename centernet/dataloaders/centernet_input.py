import tensorflow as tf

from centernet.ops import preprocessing_ops
from official.vision.beta.dataloaders import parser
from yolo.ops import preprocessing_ops as yolo_preprocessing_ops


class CenterNetParser(parser.Parser):
  def __init__(self,
                image_w: int = 512,
                image_h: int = 512,
                num_classes: int = 90,
                max_num_instances: int = 200,
                gaussian_iou: float = 0.7,
                dtype: str = 'float32'):
    
    self._image_w = image_w
    self._image_h = image_h
    self._num_classes = num_classes
    self._max_num_instances = max_num_instances
    self._gaussian_iou = gaussian_iou
    self._gaussian_bump = True
    self._gaussian_rad = -1

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
  
  def _generate_heatmap(self, boxes, output_size, input_size):
    boxes = tf.cast(boxes, dtype=tf.float32)

    tl_heatmaps = tf.zeros((self._num_classes, output_size[0], output_size[1]), dtype=tf.float32)
    br_heatmaps = tf.zeros((self._num_classes, output_size[0], output_size[1]), dtype=tf.float32)
    ct_heatmaps = tf.zeros((self._num_classes, output_size[0], output_size[1]), dtype=tf.float32)
    tl_offset = tf.zeros((self._max_num_instances, 2), dtype=tf.float32)
    br_offset = tf.zeros((self._max_num_instances, 2), dtype=tf.float32)
    ct_offset = tf.zeros((self._max_num_instances, 2), dtype=tf.float32)
    tl_size = tf.zeros((self._max_num_instances), dtype=tf.int64)
    br_size = tf.zeros((self._max_num_instances), dtype=tf.int64)
    ct_size = tf.zeros((self._max_num_instances), dtype=tf.int64)
    tag_masks = tf.zeros((self._max_num_instances), dtype=tf.uint8)

    width_ratio = output_size[1] / input_size[1]
    height_ratio = output_size[0] / input_size[0]

    width_ratio = tf.cast(width_ratio, tf.float32)
    height_ratio = tf.cast(height_ratio, tf.float32)

    num_boxes = tf.shape(boxes)[0]

    for tag_ind in tf.range(num_boxes):
      detection = boxes[tag_ind]
    
      category = detection[-1] # TODO: See if subtracting 1 from the class like the paper is unnecessary
      category = 0 # FIXME: For testing only

      xtl, ytl = detection[0], detection[1]
      xbr, ybr = detection[2], detection[3]

      xct, yct = (
          (detection[2] + detection[0]) / 2,
          (detection[3] + detection[1]) / 2
      )

      fxtl = (xtl * width_ratio)
      fytl = (ytl * height_ratio)
      fxbr = (xbr * width_ratio)
      fybr = (ybr * height_ratio)
      fxct = (xct * width_ratio)
      fyct = (yct * height_ratio)

      xtl = tf.math.floor(fxtl)
      ytl = tf.math.floor(fytl)
      xbr = tf.math.floor(fxbr)
      ybr = tf.math.floor(fybr)
      xct = tf.math.floor(fxct)
      yct = tf.math.floor(fyct)

      if self._gaussian_bump:
        width = detection[2] - detection[0]
        height = detection[3] - detection[1]

        width = tf.math.ceil(width * width_ratio)
        height = tf.math.ceil(height * height_ratio)

        if self._gaussian_rad == -1:
          radius = preprocessing_ops.gaussian_radius((height, width), self._gaussian_iou)
          radius = tf.math.maximum(0.0, tf.math.floor(radius))
        else:
          radius = self._gaussian_rad

      # test
      #   tl_heatmaps = preprocessing_ops.draw_gaussian(tl_heatmaps[category], category, [xtl, ytl], radius)
      # inputs heatmap, center, radius, k=1
        tl_heatmaps = preprocessing_ops.draw_gaussian(tl_heatmaps, [[category, xtl, ytl, radius]])
        br_heatmaps = preprocessing_ops.draw_gaussian(br_heatmaps, [[category, xbr, ybr, radius]])
        ct_heatmaps = preprocessing_ops.draw_gaussian(ct_heatmaps, [[category, xct, yct, radius]], scaling_factor=5)

      else:
        # TODO: See if this is a typo
        # tl_heatmaps[category, ytl, xtl] = 1
        # br_heatmaps[category, ybr, xbr] = 1
        # ct_heatmaps[category, yct, xct] = 1
        tl_heatmaps = tf.tensor_scatter_nd_update(tl_heatmaps, [[category, ytl, xtl]], [1])
        br_heatmaps = tf.tensor_scatter_nd_update(br_heatmaps, [[category, ybr, xbr]], [1])
        ct_heatmaps = tf.tensor_scatter_nd_update(ct_heatmaps, [[category, yct, xct]], [1])

      # tl_offset[tag_ind, :] = [fxtl - xtl, fytl - ytl]
      # br_offset[tag_ind, :] = [fxbr - xbr, fybr - ybr]
      # ct_offset[tag_ind, :] = [fxct - xct, fyct - yct]
      # tl_size[tag_ind] = ytl * output_size[1] + xtl
      # br_size[tag_ind] = ybr * output_size[1] + xbr
      # ct_size[tag_ind] = yct * output_size[1] + xct
      tl_offset = tf.tensor_scatter_nd_update(tl_offset, [[tag_ind, 0], [tag_ind, 1]], [fxtl - xtl, fytl - ytl])
      br_offset = tf.tensor_scatter_nd_update(br_offset, [[tag_ind, 0], [tag_ind, 1]], [fxbr - xbr, fybr - ybr])
      ct_offset = tf.tensor_scatter_nd_update(ct_offset, [[tag_ind, 0], [tag_ind, 1]], [fxct - xct, fyct - yct])
      tl_size = tf.tensor_scatter_nd_update(tl_size, [[tag_ind]], [ytl * output_size[1] + xtl])
      br_size = tf.tensor_scatter_nd_update(br_size, [[tag_ind]], [ybr * output_size[1] + xbr])
      ct_size = tf.tensor_scatter_nd_update(ct_size, [[tag_ind]], [yct * output_size[1] + xct])

    labels = {
      'tl_size': tl_size,
      'br_size': br_size,
      'ct_size': ct_size,
      'tl_heatmaps': tl_heatmaps,
      'br_heatmaps': br_heatmaps,
      'ct_heatmaps': ct_heatmaps,
      'tag_masks': tag_masks,
      'tl_offset': tl_offset,
      'br_offset': br_offset,
      'ct_offset': ct_offset,
    }
    return labels

  def _parse_train_data(self, data):
    """Generates images and labels that are usable for model training.

    Args:
        decoded_tensors: a dict of Tensors produced by the decoder.

    Returns:
        images: the image tensor.
        labels: a dict of Tensors that contains labels.
    """
    # FIXME: This is a copy of parse eval data
    image = data["image"] / 255
    boxes = data['groundtruth_boxes']
    classes = data['groundtruth_classes']

    image, boxes, info = yolo_preprocessing_ops.letter_box(
      image, boxes, xs = 0.5, ys = 0.5, target_dim=self._image_w)

    image = tf.cast(image, self._dtype)
    shape = tf.shape(image)
    height = shape[0]
    width = shape[1]

    labels = self._generate_heatmap(
        boxes=boxes, output_size=[self._image_h, self._image_w], input_size=[height, width]
    )

    labels.update({'bbox': boxes})
    
    return image, labels

  def _parse_eval_data(self, data):
    image = data["image"] / 255
    boxes = data['groundtruth_boxes']
    classes = data['groundtruth_classes']

    image, boxes, info = yolo_preprocessing_ops.letter_box(
      image, boxes, xs = 0.5, ys = 0.5, target_dim=self._image_w)

    image = tf.cast(image, self._dtype)
    shape = tf.shape(image)
    height = shape[0]
    width = shape[1]

    labels = self._generate_heatmap(
        boxes=boxes, output_size=[self._image_h, self._image_w], input_size=[height, width]
    )

    labels.update({'bbox': boxes})
    
    return image, labels

  def postprocess_fn(self, is_training):
    if is_training:  #or self._cutmix
      return None # if not self._fixed_size or self._mosaic else None
    else:
      return None

class ObjectDetectionTest(tf.test.TestCase):
    def generate_heatmaps(self, dectections):
      detections = [[
        (10, 30, 15, 17, 0)
      ]]

      labels = generate_heatmaps(1, 2, (416, 416), detections)
      pass

if __name__ == '__main__':
  # This code is for visualization
  import matplotlib.pyplot as plt
  detections = [
    (10, 300, 15, 370, 0),
    (100, 300, 150, 370, 0),
    (200, 100, 15, 170, 0),
  ]

  #labels = tf.function(CenterNetParser(2, 200, 0.7)._generate_heatmap)(
  labels = CenterNetParser()._generate_heatmap(
    tf.constant(detections, dtype=tf.float32), [416, 416], [416, 416]
  )
  tl_heatmaps = labels['tl_heatmaps']
  br_heatmaps = labels['br_heatmaps']
  ct_heatmaps = labels['ct_heatmaps']

#   tl_heatmaps, br_heatmaps, ct_heatmaps = generate_heatmaps(1, 2, (416, 416), detections)
  # ct_heatmaps[batch_id, class_id, ...]
  plt.imshow(ct_heatmaps[0, ...])
  plt.show()
  # This is to run the test
  # tf.test.main()

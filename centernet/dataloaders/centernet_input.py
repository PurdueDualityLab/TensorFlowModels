import tensorflow as tf

from centernet.ops import preprocessing_ops
from official.vision.beta.dataloaders import parser
from yolo.ops import preprocessing_ops as yolo_preprocessing_ops


class CenterNetParser(parser.Parser):
  def __init__(self,
               image_w: int = 512,
               image_h: int = 512,
               num_classes: int = 90,
               max_num_instances: int = 128, # 200 or 128?
               gaussian_iou: float = 0.7,
               output_dims: int = 128,
               dtype: str = 'float32'):
    
    self._image_w = image_w
    self._image_h = image_h
    self._num_classes = num_classes
    self._max_num_instances = max_num_instances
    self._gaussian_iou = gaussian_iou
    self._output_dims = output_dims
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

  def _build_labels(self, boxes, classes, output_size, input_size):
    """ Generates the ground truth labels for centernet.
    
    Ground truth labels are generated by splatting gaussians on heatmaps for
    corners and centers. Regressed features (offsets and sizes) are also
    generated.

    Args:
      boxes: Tensor of shape [num_boxes, 4], where the last dimension
        corresponds to the top left x, top left y, bottom right x, and
        bottom left y coordinates of the bounding box
      classes: Tensor of shape [num_boxes], that contains the class of each
        box, given in the same order as the boxes
      output_size: A list containing the desired output height and width of the 
        heatmaps
      input_size: A list the expected input height and width of the image
    Returns:
      Dictionary of labels with the following fields:
        'tl_heatmaps': Tensor of shape [output_h, output_w, num_classes],
          heatmap with splatted gaussians centered at the positions and channels
          corresponding to the top left location and class of the object
        'br_heatmaps': Tensor of shape [output_h, output_w, num_classes],
          heatmap with splatted gaussians centered at the positions and channels
          corresponding to the bottom right location and class of the object
        'ct_heatmaps': Tensor of shape [output_h, output_w, num_classes],
          heatmap with splatted gaussians centered at the positions and channels
          corresponding to the center location and class of the object
        'tl_offset': Tensor of shape [max_num_instances, 2], where the first
          num_boxes entries contain the x-offset and y-offset of the top-left
          corner of an object. All other entires are 0
        'br_offset': Tensor of shape [max_num_instances, 2], where the first
          num_boxes entries contain the x-offset and y-offset of the 
          bottom-right corner of an object. All other entires are 0
        'ct_offset': Tensor of shape [max_num_instances, 2], where the first
          num_boxes entries contain the x-offset and y-offset of the center of 
          an object. All other entires are 0
        'size': Tensor of shape [max_num_instances, 2], where the first
          num_boxes entries contain the width and height of an object. All 
          other entires are 0
        'box_mask': Tensor of shape [max_num_instances], where the first
          num_boxes entries are 1. All other entires are 0
        'box_indices': Tensor of shape [max_num_instances, 3], where the first
          num_boxes entries contain the class, y-center, and-x center of a
          valid box. These are referenced to extract the regressed box features
          from the prediction when computing the loss. 
    """
    boxes = tf.cast(boxes, dtype=tf.float32)
    classes = tf.cast(classes, dtype=tf.float32)
    input_h, input_w = input_size
    output_h, output_w = output_size
    
    # We will transpose these at the end
    tl_heatmaps = tf.zeros((self._num_classes, output_h, output_w), dtype=tf.float32)
    br_heatmaps = tf.zeros((self._num_classes, output_h, output_w), dtype=tf.float32)
    ct_heatmaps = tf.zeros((self._num_classes, output_h, output_w), dtype=tf.float32)

    tl_offset = tf.zeros((self._max_num_instances, 2), dtype=tf.float32)
    br_offset = tf.zeros((self._max_num_instances, 2), dtype=tf.float32)
    ct_offset = tf.zeros((self._max_num_instances, 2), dtype=tf.float32)
    size      = tf.zeros((self._max_num_instances, 2), dtype=tf.float32)

    box_mask = tf.zeros((self._max_num_instances), dtype=tf.int32)
    box_indices  = tf.zeros((self._max_num_instances, 2), dtype=tf.int32)

    # Scaling factor for determining center/corners
    width_ratio = tf.cast(output_w / input_w, tf.float32)
    height_ratio = tf.cast(output_h / input_h, tf.float32)

    num_boxes = tf.shape(boxes)[0]

    for tag_ind in tf.range(num_boxes):
      box = boxes[tag_ind]
      obj_class = classes[tag_ind] # TODO: See if subtracting 1 from the class like the paper is unnecessary

      ytl, xtl, ybr, xbr = box[0], box[1], box[2], box[3]

      xct, yct = (
        (xtl + xbr) / 2,
        (ytl + ybr) / 2
      )

      # Scale center and corner locations
      fxtl = (xtl * width_ratio)
      fytl = (ytl * height_ratio)
      fxbr = (xbr * width_ratio)
      fybr = (ybr * height_ratio)
      fxct = (xct * width_ratio)
      fyct = (yct * height_ratio)

      # Fit center and corners onto the output image
      xtl = tf.math.floor(fxtl)
      ytl = tf.math.floor(fytl)
      xbr = tf.math.floor(fxbr)
      ybr = tf.math.floor(fybr)
      xct = tf.math.floor(fxct)
      yct = tf.math.floor(fyct)
      
      # Splat gaussian at for the center/corner heatmaps
      if self._gaussian_bump:
        width = box[3] - box[1]
        height = box[2] - box[0]

        width = tf.math.ceil(width * width_ratio)
        height = tf.math.ceil(height * height_ratio)

        if self._gaussian_rad == -1:
          radius = preprocessing_ops.gaussian_radius((height, width), self._gaussian_iou)
          radius = tf.math.maximum(0.0, tf.math.floor(radius))
        else:
          radius = self._gaussian_rad

        tl_heatmaps = preprocessing_ops.draw_gaussian(tl_heatmaps, [[obj_class, xtl, ytl, radius]])
        br_heatmaps = preprocessing_ops.draw_gaussian(br_heatmaps, [[obj_class, xbr, ybr, radius]])
        ct_heatmaps = preprocessing_ops.draw_gaussian(ct_heatmaps, [[obj_class, xct, yct, radius]], scaling_factor=5)

      else:
        tl_heatmaps = tf.tensor_scatter_nd_update(tl_heatmaps, [[obj_class, ytl, xtl]], [1])
        br_heatmaps = tf.tensor_scatter_nd_update(br_heatmaps, [[obj_class, ybr, xbr]], [1])
        ct_heatmaps = tf.tensor_scatter_nd_update(ct_heatmaps, [[obj_class, yct, xct]], [1])
      
      # Add box offset and size to the ground truth
      tl_offset = tf.tensor_scatter_nd_update(tl_offset, [[tag_ind, 0], [tag_ind, 1]], [fxtl - xtl, fytl - ytl])
      br_offset = tf.tensor_scatter_nd_update(br_offset, [[tag_ind, 0], [tag_ind, 1]], [fxbr - xbr, fybr - ybr])
      ct_offset = tf.tensor_scatter_nd_update(ct_offset, [[tag_ind, 0], [tag_ind, 1]], [fxct - xct, fyct - yct])
      size      = tf.tensor_scatter_nd_update(size, [[tag_ind, 0], [tag_ind, 1]], [width, height])

      # Initialy the mask is zeros, but each valid box needs to be unmasked
      box_mask = tf.tensor_scatter_nd_update(box_mask, [[tag_ind]], [1])

      # Contains the y and x coordinate of the box center in the heatmap
      box_indices = tf.tensor_scatter_nd_update(box_indices, [[tag_ind, 0], [tag_ind, 1]], [yct, xct])

    # Make heatmaps of shape [height, width, num_classes]
    tl_heatmaps = tf.transpose(tl_heatmaps, perm=[1, 2, 0])
    br_heatmaps = tf.transpose(br_heatmaps, perm=[1, 2, 0])
    ct_heatmaps = tf.transpose(ct_heatmaps, perm=[1, 2, 0])

    labels = {
      'tl_heatmaps': tl_heatmaps,
      'br_heatmaps': br_heatmaps,
      'ct_heatmaps': ct_heatmaps,
      'tl_offset': tl_offset,
      'br_offset': br_offset,
      'ct_offset': ct_offset,
      'size': size,
      'box_mask': box_mask,
      'box_indices': box_indices,
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
    image = data['image'] / 255
    boxes = data['groundtruth_boxes']
    classes = data['groundtruth_classes']

    image, boxes, info = yolo_preprocessing_ops.letter_box(
      image, boxes, xs = 0.5, ys = 0.5, target_dim=self._image_w)

    image = tf.cast(image, self._dtype)
    shape = tf.shape(image)
    height = shape[0]
    width = shape[1]

    labels = self._build_labels(
        boxes=boxes, classes=classes, 
        output_size=[self._output_dims, self._output_dims], 
        input_size=[self._image_h, self._image_w]
    )

    labels.update({'bbox': boxes})
    
    return image, labels

  def _parse_eval_data(self, data):
    image = data['image'] / 255
    boxes = data['groundtruth_boxes']
    classes = data['groundtruth_classes']

    image, boxes, info = yolo_preprocessing_ops.letter_box(
      image, boxes, xs = 0.5, ys = 0.5, target_dim=self._image_w)

    image = tf.cast(image, self._dtype)
    shape = tf.shape(image)
    height = shape[0]
    width = shape[1]

    labels = self._build_labels(
        boxes=boxes, classes=classes, 
        output_size=[self._output_dims, self._output_dims], 
        input_size=[self._image_h, self._image_w]
    )

    labels.update({'bbox': boxes})
    labels.update({'boxes': boxes})
    labels.update({'source_id': data['source_id']})
    labels.update({'height': data['height']})
    labels.update({'width': data['width']})
    labels.update({'classes': classes})
    
    return image, labels

  def postprocess_fn(self, is_training):
    if is_training:  #or self._cutmix
      return None # if not self._fixed_size or self._mosaic else None
    else:
      return None


if __name__ == '__main__':
  # This code is for visualization
  import matplotlib.pyplot as plt
  boxes = [
    (10, 300, 15, 370),
    (100, 300, 150, 370),
    (200, 100, 15, 170),
  ]

  classes = (0, 1, 2)

  labels = CenterNetParser()._build_labels(
    tf.constant(boxes, dtype=tf.float32), 
    tf.constant(classes, dtype=tf.float32), 
    [512, 512], [512, 512]
  )
  tl_heatmaps = labels['tl_heatmaps']
  br_heatmaps = labels['br_heatmaps']
  ct_heatmaps = labels['ct_heatmaps']

  # plt.imshow(ct_heatmaps[0, ...])
  plt.imshow(ct_heatmaps[..., 1])
  # plt.imshow(ct_heatmaps[2, ...])
  plt.show()

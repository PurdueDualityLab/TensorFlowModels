import tensorflow as tf
import tensorflow.keras as ks

from yolo.ops.nms_ops import nms


@tf.keras.utils.register_keras_serializable(package='centernet')
class CenterNetLayer(ks.Model):
  """ CenterNet Detection Generator

  Parses predictions from the CenterNet decoder into the final bounding boxes, 
  confidences, and classes. This class contains repurposed methods from the 
  TensorFlow Object Detection API
  in: https://github.com/tensorflow/models/blob/master/research/object_detection/meta_architectures/center_net_meta_arch.py
  """
  def __init__(self,
               max_detections=100,
               peak_error=1e-6,
               peak_extract_kernel_size=3,
               class_offset=1,
               net_down_scale=4,
               input_image_dims=512,
               use_nms=False,
               nms_pre_thresh=0.1,
               nms_thresh=0.4,
               **kwargs):
    """
    Args:
      max_detections: An integer specifying the maximum number of bounding
        boxes generated. This is an upper bound, so the number of generated
        boxes may be less than this due to thresholding/non-maximum suppression.
      peak_error: A float for determining non-valid heatmap locations to mask.
      peak_extract_kernel_size: An integer indicating the kernel size used when
        performing max-pool over the heatmaps to detect valid center locations
        from its neighbors. From the paper, set this to 3 to detect valid.
        locations that have responses greater than its 8-connected neighbors
      use_nms: Boolean for whether or not to use non-maximum suppression to
        filter the bounding boxes.
      class_offset: An integer indicating to add an offset to the class 
        prediction if the dataset labels have been shifted.
      net_down_scale: An integer that specifies 

    call Returns:
      Dictionary with keys 'bbox', 'classes', 'confidence', and 'num_dets'
      storing each bounding box in [y_min, x_min, y_max, x_max] format, 
      its respective class and confidence score, and the number of detections
      made.
    """
    super().__init__(**kwargs)
    
    # Object center selection parameters
    self._max_detections = max_detections
    self._peak_error = peak_error
    self._peak_extract_kernel_size = peak_extract_kernel_size

    # Used for adjusting class prediction
    self._class_offset = class_offset

    # Box normalization parameters
    self._net_down_scale = net_down_scale
    self._input_image_dims = input_image_dims

    self._use_nms = use_nms
    self._nms_pre_thresh = nms_pre_thresh
    self._nms_thresh = nms_thresh
  
  def process_heatmap(self, 
                      feature_map,
                      kernel_size):
    """ Processes the heatmap into peaks for box selection.

    Given a heatmap, this function first masks out nearby heatmap locations of
    the same class using max-pooling such that, ideally, only one center for the
    object remains. Then, center locations are masked according to their scores
    in comparison to a threshold. NOTE: Repurposed from Google OD API.

    Args:
      feature_map: A Tensor with shape [batch_size, height, width, num_classes] 
        which is the center heatmap predictions.
      kernel_size: An integer value for max-pool kernel size.

    Returns:
      A Tensor with the same shape as the input but with non-valid center
        prediction locations masked out.
    """

    feature_map = tf.math.sigmoid(feature_map)
    if not kernel_size or kernel_size == 1:
      feature_map_peaks = feature_map
    else:
      feature_map_max_pool = tf.nn.max_pool(
          feature_map, ksize=kernel_size, 
          strides=1, padding='SAME')

      feature_map_peak_mask = tf.math.abs(
          feature_map - feature_map_max_pool) < self._peak_error
      
      # Zero out everything that is not a peak.
      feature_map_peaks = (
          feature_map * tf.cast(feature_map_peak_mask, feature_map.dtype))

    return feature_map_peaks
  
  def get_row_col_channel_indices_from_flattened_indices(self,
                                                         indices, 
                                                         num_cols,
                                                         num_channels):
    """ Computes row, column and channel indices from flattened indices.

    NOTE: Repurposed from Google OD API.

    Args:
      indices: An integer tensor of any shape holding the indices in the flattened
        space.
      num_cols: Number of columns in the image (width).
      num_channels: Number of channels in the image.

    Returns:
      row_indices: The row indices corresponding to each of the input indices.
        Same shape as indices.
      col_indices: The column indices corresponding to each of the input indices.
        Same shape as indices.
      channel_indices. The channel indices corresponding to each of the input
        indices.
    """
    # Avoid using mod operator to make the ops more easy to be compatible with
    # different environments, e.g. WASM.

    # all inputs and outputs are dtype int32
    row_indices = (indices // num_channels) // num_cols
    col_indices = (indices // num_channels) - row_indices * num_cols
    channel_indices_temp = indices // num_channels
    channel_indices = indices - channel_indices_temp * num_channels

    return row_indices, col_indices, channel_indices

  def multi_range(self,
                  limit,
                  value_repetitions=1,
                  range_repetitions=1,
                  dtype=tf.int32):
    """ Creates a sequence with optional value duplication and range repetition.

    As an example (see the Args section for more details),
    _multi_range(limit=2, value_repetitions=3, range_repetitions=4) returns:
    [0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1]
    NOTE: Repurposed from Google OD API.

    Args:
      limit: A 0-D Tensor (scalar). Upper limit of sequence, exclusive.
      value_repetitions: Integer. The number of times a value in the sequence is
        repeated. With value_repetitions=3, the result is [0, 0, 0, 1, 1, 1, ..].
      range_repetitions: Integer. The number of times the range is repeated. With
        range_repetitions=3, the result is [0, 1, 2, .., 0, 1, 2, ..].
      dtype: The type of the elements of the resulting tensor.
    
    Returns:
      A 1-D tensor of type `dtype` and size
        [`limit` * `value_repetitions` * `range_repetitions`] that contains the
        specified range with given repetitions.
    """
    return tf.reshape(
        tf.tile(
          tf.expand_dims(tf.range(limit, dtype=dtype), axis=-1),
          multiples=[range_repetitions, value_repetitions]), [-1])
  
  def get_top_k_peaks(self,
                      feature_map_peaks,
                      batch_size,
                      width,
                      num_classes,
                      k=100):
    """ Gets the scores and indices of the top-k peaks from the feature map

    This function flattens the feature map in order to retrieve the top-k
    peaks, then computes the x, y, and class indices for those scores.
    NOTE: Repurposed from Google OD API.

    Args:
      feature_map_peaks: A Tensor with shape [batch_size, height, width, num_classes] 
        which is the processed center heatmap peaks.
      batch_size: An integer that indicates the batch size of the input.
      width: An integer that indicates the width (and also height) of the input.
      num_classes: An integer for the number of possible classes. This is also
        the channel depth of the input.
      k: Integer that controls how many peaks to select.
    
    Returns:
      top_scores: A Tensor with shape [batch_size, k] containing the top-k scores.
      y_indices: A Tensor with shape [batch_size, k] containing the top-k 
        y-indices corresponding to top_scores.
      x_indices: A Tensor with shape [batch_size, k] containing the top-k 
        x-indices corresponding to top_scores.
      channel_indices: A Tensor with shape [batch_size, k] containing the top-k 
        channel indices corresponding to top_scores.
    """
    # Flatten the entire prediction per batch
    feature_map_peaks_flat = tf.reshape(feature_map_peaks, [batch_size, -1])

    # top_scores and top_indices have shape [batch_size, k]
    top_scores, top_indices = tf.math.top_k(feature_map_peaks_flat, k=k)

    # Get x, y and channel indices corresponding to the top indices in the flat
    # array.
    y_indices, x_indices, channel_indices = (
      self.get_row_col_channel_indices_from_flattened_indices(
        top_indices, width, num_classes))

    return top_scores, y_indices, x_indices, channel_indices

  def get_boxes(self, 
                y_indices, 
                x_indices,
                channel_indices, 
                height_width_predictions,
                offset_predictions,
                num_boxes):
    """ Organizes prediction information into the final bounding boxes.

    NOTE: Repurposed from Google OD API.

    Args:
      y_indices: A Tensor with shape [batch_size, k] containing the top-k 
        y-indices corresponding to top_scores.
      x_indices: A Tensor with shape [batch_size, k] containing the top-k 
        x-indices corresponding to top_scores.
      channel_indices: A Tensor with shape [batch_size, k] containing the top-k 
        channel indices corresponding to top_scores.
      height_width_predictions: A Tensor with shape [batch_size, height, width, 2] 
        containing the object size predictions.
      offset_predictions: A Tensor with shape [batch_size, height, width, 2] 
        containing the object local offset predictions.
      num_boxes: The number of boxes.

    Returns:
      boxes: A Tensor with shape [batch_size, num_boxes, 4] that contains the
        bounding box coordinates in [y_min, x_min, y_max, x_max] format.
      detection_classes: A Tensor with shape [batch_size, 100] that gives the 
        class prediction for each box.
      num_detections: Number of non-zero confidence detections made.
    """
    # TF Lite does not support tf.gather with batch_dims > 0, so we need to use
    # tf_gather_nd instead and here we prepare the indices for that.

    # shapes of heatmap output
    shape = tf.shape(height_width_predictions)
    batch_size, height, width = shape[0], shape[1], shape[2]

    # combined indices dtype=int32
    combined_indices = tf.stack([
        self.multi_range(batch_size, value_repetitions=num_boxes),
        tf.reshape(y_indices, [-1]),
        tf.reshape(x_indices, [-1])
      ], axis=1)

    new_height_width = tf.gather_nd(height_width_predictions, combined_indices)
    new_height_width = tf.reshape(new_height_width, [batch_size, num_boxes, 2])
    height_width = tf.maximum(new_height_width, 0.0)

    # height and widths dtype=float32
    heights = height_width[..., 0]
    widths = height_width[..., 1]

    # Get the offsets of center points
    new_offsets = tf.gather_nd(offset_predictions, combined_indices)
    offsets = tf.reshape(new_offsets, [batch_size, num_boxes, 2])
    
    # offsets are dtype=float32
    y_offsets = offsets[..., 0]
    x_offsets = offsets[..., 1]
    
    y_indices = tf.cast(y_indices, dtype=heights.dtype)
    x_indices = tf.cast(x_indices, dtype=widths.dtype)

    detection_classes = channel_indices + self._class_offset
    ymin = y_indices + y_offsets - heights / 2.0
    xmin = x_indices + x_offsets - widths / 2.0
    ymax = y_indices + y_offsets + heights / 2.0
    xmax = x_indices + x_offsets + widths / 2.0

    ymin = tf.clip_by_value(ymin, 0., tf.cast(height, ymin.dtype))
    xmin = tf.clip_by_value(xmin, 0., tf.cast(width, xmin.dtype))
    ymax = tf.clip_by_value(ymax, 0., tf.cast(height, ymax.dtype))
    xmax = tf.clip_by_value(xmax, 0., tf.cast(width, xmax.dtype))
    boxes = tf.stack([ymin, xmin, ymax, xmax], axis=2)

    return boxes, detection_classes
  
  def convert_strided_predictions_to_normalized_boxes(self,
                                                      boxes):
    boxes = boxes * tf.cast(self._net_down_scale, boxes.dtype)
    boxes = boxes / tf.cast(self._input_image_dims, boxes.dtype)
    boxes = tf.clip_by_value(boxes, 0.0, 1.0)
    return boxes

  def call(self, inputs):
    # Get heatmaps from decoded outputs via final hourglass stack output
    all_ct_heatmaps = inputs['ct_heatmaps']
    all_ct_sizes = inputs['ct_size']
    all_ct_offsets = inputs['ct_offset']
    
    ct_heatmaps = all_ct_heatmaps[-1]
    ct_sizes = all_ct_sizes[-1]
    ct_offsets = all_ct_offsets[-1]
    
    shape = tf.shape(ct_heatmaps)

    batch_size, height, width, num_channels = shape[0], shape[1], shape[2], shape[3]
    
    # Process heatmaps using 3x3 max pool and applying sigmoid
    peaks = self.process_heatmap(ct_heatmaps, 
      kernel_size=self._peak_extract_kernel_size)
    
    # Get top scores along with their x, y, and class
    scores, y_indices, x_indices, channel_indices = self.get_top_k_peaks(peaks, 
      batch_size, width, num_channels, k=self._max_detections)
    
    # Parse the score and indices into bounding boxes
    boxes, classes = self.get_boxes(y_indices, x_indices, channel_indices, ct_sizes, ct_offsets, self._max_detections)
    
    # Normalize bounding boxes
    boxes = self.convert_strided_predictions_to_normalized_boxes(boxes)

    # Apply nms 
    if self._use_nms:
      boxes = tf.expand_dims(boxes, axis=-2)
      multi_class_scores = tf.gather_nd(peaks, 
        tf.stack([y_indices, x_indices], -1), batch_dims=1)

      boxes, _, scores = nms(boxes=boxes, classes=multi_class_scores, 
        confidence=scores, k=self._max_detections, limit_pre_thresh=True,
        pre_nms_thresh=0.1, nms_thresh=0.4)

    num_det = tf.reduce_sum(tf.cast(scores > 0, dtype=tf.int32), axis=1)

    return {
      'bbox': boxes,
      'classes': classes,
      'confidence': scores,
      'num_dets': num_det
    }

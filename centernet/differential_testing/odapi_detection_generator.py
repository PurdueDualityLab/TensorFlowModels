import tensorflow.compat.v1 as tf

PEAK_EPSILON = 1e-6

class ODAPIDetectionGenerator(tf.keras.Model):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)

  def call(self, inputs):
    object_center_prob = tf.nn.sigmoid(inputs['ct_heatmaps'][-1])

    detection_scores, y_indices, x_indices, channel_indices = (
        top_k_feature_map_locations(
            object_center_prob, max_pool_kernel_size=3,
            k=100))
    multiclass_scores = tf.gather_nd(
        object_center_prob, tf.stack([y_indices, x_indices], -1), batch_dims=1)

    num_detections = tf.reduce_sum(tf.to_int32(detection_scores > 0), axis=1)
    boxes_strided = None

    boxes_strided = (
          prediction_tensors_to_boxes(y_indices, x_indices,
                                      inputs['ct_size'][-1],
                                      inputs['ct_offset'][-1]))
    boxes = convert_strided_predictions_to_normalized_boxes(
          boxes_strided, 4)
    return {
      'bbox': boxes,
      'classes': channel_indices,
      'confidence': detection_scores,
      'num_dets': num_detections
    }


def row_col_channel_indices_from_flattened_indices(indices, num_cols,
                                                   num_channels):
  """Computes row, column and channel indices from flattened indices.
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
  # Be careful with this function when running a model in float16 precision
  # (e.g. TF.js with WebGL) because the array indices may not be represented
  # accurately if they are too large, resulting in incorrect channel indices.
  # See:
  # https://en.wikipedia.org/wiki/Half-precision_floating-point_format#Precision_limitations_on_integer_values
  #
  # Avoid using mod operator to make the ops more easy to be compatible with
  # different environments, e.g. WASM.
  row_indices = (indices // num_channels) // num_cols
  col_indices = (indices // num_channels) - row_indices * num_cols
  channel_indices_temp = indices // num_channels
  channel_indices = indices - channel_indices_temp * num_channels

  return row_indices, col_indices, channel_indices

def combined_static_and_dynamic_shape(tensor):
  """Returns a list containing static and dynamic values for the dimensions.
  Returns a list of static and dynamic values for shape dimensions. This is
  useful to preserve static shapes when available in reshape operation.
  Args:
    tensor: A tensor of any type.
  Returns:
    A list of size tensor.shape.ndims containing integers or a scalar tensor.
  """
  static_tensor_shape = tensor.shape.as_list()
  dynamic_tensor_shape = tf.shape(tensor)
  combined_shape = []
  for index, dim in enumerate(static_tensor_shape):
    if dim is not None:
      combined_shape.append(dim)
    else:
      combined_shape.append(dynamic_tensor_shape[index])
  return combined_shape

def _get_shape(tensor, num_dims):
  assert len(tensor.shape.as_list()) == num_dims
  return combined_static_and_dynamic_shape(tensor)

def _to_float32(x):
  return tf.cast(x, tf.float32)

def convert_strided_predictions_to_normalized_boxes(boxes, stride):
  """Converts predictions in the output space to normalized boxes.
  Boxes falling outside the valid image boundary are clipped to be on the
  boundary.
  Args:
    boxes: A tensor of shape [batch_size, num_boxes, 4] holding the raw
     coordinates of boxes in the model's output space.
    stride: The stride in the output space.
    true_image_shapes: A tensor of shape [batch_size, 3] representing the true
      shape of the input not considering padding.
  Returns:
    boxes: A tensor of shape [batch_size, num_boxes, 4] representing the
      coordinates of the normalized boxes.
  """
  # Note: We use tf ops instead of functions in box_list_ops to make this
  # function compatible with dynamic batch size.
  boxes = boxes * stride
  boxes = boxes / 512
  boxes = tf.clip_by_value(boxes, 0.0, 1.0)
  return boxes

def _multi_range(limit,
                 value_repetitions=1,
                 range_repetitions=1,
                 dtype=tf.int32):
  return tf.reshape(
      tf.tile(
          tf.expand_dims(tf.range(limit, dtype=dtype), axis=-1),
          multiples=[range_repetitions, value_repetitions]), [-1])

def top_k_feature_map_locations(feature_map, max_pool_kernel_size=3, k=100,
                              per_channel=False):

  if not max_pool_kernel_size or max_pool_kernel_size == 1:
    feature_map_peaks = feature_map
  else:
    feature_map_max_pool = tf.nn.max_pool(
        feature_map, ksize=max_pool_kernel_size, strides=1, padding='SAME')

    feature_map_peak_mask = tf.math.abs(
        feature_map - feature_map_max_pool) < PEAK_EPSILON

    # Zero out everything that is not a peak.
    feature_map_peaks = (
        feature_map * _to_float32(feature_map_peak_mask))

  batch_size, _, width, num_channels = _get_shape(feature_map, 4)

  if per_channel:
    if k == 1:
      feature_map_flattened = tf.reshape(
          feature_map_peaks, [batch_size, -1, num_channels])
      scores = tf.math.reduce_max(feature_map_flattened, axis=1)
      peak_flat_indices = tf.math.argmax(
          feature_map_flattened, axis=1, output_type=tf.dtypes.int32)
      peak_flat_indices = tf.expand_dims(peak_flat_indices, axis=-1)
    else:
      # Perform top k over batch and channels.
      feature_map_peaks_transposed = tf.transpose(feature_map_peaks,
                                                  perm=[0, 3, 1, 2])
      feature_map_peaks_transposed = tf.reshape(
          feature_map_peaks_transposed, [batch_size, num_channels, -1])
      scores, peak_flat_indices = tf.math.top_k(
          feature_map_peaks_transposed, k=k)
    # Convert the indices such that they represent the location in the full
    # (flattened) feature map of size [batch, height * width * channels].
    channel_idx = tf.range(num_channels)[tf.newaxis, :, tf.newaxis]
    peak_flat_indices = num_channels * peak_flat_indices + channel_idx
    scores = tf.reshape(scores, [batch_size, -1])
    peak_flat_indices = tf.reshape(peak_flat_indices, [batch_size, -1])
  else:
    if k == 1:
      feature_map_peaks_flat = tf.reshape(feature_map_peaks, [batch_size, -1])
      scores = tf.math.reduce_max(feature_map_peaks_flat, axis=1, keepdims=True)
      peak_flat_indices = tf.expand_dims(tf.math.argmax(
          feature_map_peaks_flat, axis=1, output_type=tf.dtypes.int32), axis=-1)
    else:
      feature_map_peaks_flat = tf.reshape(feature_map_peaks, [batch_size, -1])
      scores, peak_flat_indices = tf.math.top_k(feature_map_peaks_flat, k=k)

  # Get x, y and channel indices corresponding to the top indices in the flat
  # array.
  y_indices, x_indices, channel_indices = (
      row_col_channel_indices_from_flattened_indices(
          peak_flat_indices, width, num_channels))
  return scores, y_indices, x_indices, channel_indices

def prediction_tensors_to_boxes(y_indices, x_indices, height_width_predictions,
                              offset_predictions):
  batch_size, num_boxes = _get_shape(y_indices, 2)
  _, height, width, _ = _get_shape(height_width_predictions, 4)
  height, width = tf.cast(height, tf.float32), tf.cast(width, tf.float32)

  # TF Lite does not support tf.gather with batch_dims > 0, so we need to use
  # tf_gather_nd instead and here we prepare the indices for that.
  combined_indices = tf.stack([
      _multi_range(batch_size, value_repetitions=num_boxes),
      tf.reshape(y_indices, [-1]),
      tf.reshape(x_indices, [-1])
  ], axis=1)

  new_height_width = tf.gather_nd(height_width_predictions, combined_indices)
  new_height_width = tf.reshape(new_height_width, [batch_size, num_boxes, 2])

  new_offsets = tf.gather_nd(offset_predictions, combined_indices)
  offsets = tf.reshape(new_offsets, [batch_size, num_boxes, 2])

  y_indices = _to_float32(y_indices)
  x_indices = _to_float32(x_indices)

  height_width = tf.maximum(new_height_width, 0)
  heights, widths = tf.unstack(height_width, axis=2)
  y_offsets, x_offsets = tf.unstack(offsets, axis=2)

  ymin = y_indices + y_offsets - heights / 2.0
  xmin = x_indices + x_offsets - widths / 2.0
  ymax = y_indices + y_offsets + heights / 2.0
  xmax = x_indices + x_offsets + widths / 2.0

  ymin = tf.clip_by_value(ymin, 0., height)
  xmin = tf.clip_by_value(xmin, 0., width)
  ymax = tf.clip_by_value(ymax, 0., height)
  xmax = tf.clip_by_value(xmax, 0., width)
  boxes = tf.stack([ymin, xmin, ymax, xmax], axis=2)

  return boxes

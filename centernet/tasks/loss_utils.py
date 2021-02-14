class BoxListFields(object):
  """Naming conventions for BoxLists.
  Attributes:
    boxes: bounding box coordinates.
    classes: classes per bounding box.
    scores: scores per bounding box.
    weights: sample weights per bounding box.
    objectness: objectness score per bounding box.
    masks: masks per bounding box.
    boundaries: boundaries per bounding box.
    keypoints: keypoints per bounding box.
    keypoint_heatmaps: keypoint heatmaps per bounding box.
    is_crowd: is_crowd annotation per bounding box.
  """
  boxes = 'boxes'
  classes = 'classes'
  scores = 'scores'
  weights = 'weights'
  objectness = 'objectness'
  masks = 'masks'
  boundaries = 'boundaries'
  keypoints = 'keypoints'
  keypoint_heatmaps = 'keypoint_heatmaps'
  is_crowd = 'is_crowd'


class DetectionModel(six.with_metaclass(abc.ABCMeta, _BaseClass)):
  """Abstract base class for detection models.
  Extends tf.Module to guarantee variable tracking.
  """

  def __init__(self, num_classes):
    """Constructor.
    Args:
      num_classes: number of classes.  Note that num_classes *does not* include
      background categories that might be implicitly predicted in various
      implementations.
    """
    self._num_classes = num_classes
    self._groundtruth_lists = {}

    super(DetectionModel, self).__init__()

  @property
  def num_classes(self):
    return self._num_classes

  def groundtruth_lists(self, field):
    """Access list of groundtruth tensors.
    Args:
      field: a string key, options are
        fields.BoxListFields.{boxes,classes,masks,keypoints,
        keypoint_visibilities, densepose_*, track_ids,
        temporal_offsets, track_match_flags}
        fields.InputDataFields.is_annotated.
    Returns:
      a list of tensors holding groundtruth information (see also
      provide_groundtruth function below), with one entry for each image in the
      batch.
    Raises:
      RuntimeError: if the field has not been provided via provide_groundtruth.
    """
    if field not in self._groundtruth_lists:
      raise RuntimeError('Groundtruth tensor {} has not been provided'.format(
          field))
    return self._groundtruth_lists[field]

class InputDataFields(object):
  """Names for the input tensors.
  Holds the standard data field names to use for identifying input tensors. This
  should be used by the decoder to identify keys for the returned tensor_dict
  containing input tensors. And it should be used by the model to identify the
  tensors it needs.
  Attributes:
    image: image.
    image_additional_channels: additional channels.
    original_image: image in the original input size.
    key: unique key corresponding to image.
    source_id: source of the original image.
    filename: original filename of the dataset (without common path).
    groundtruth_image_classes: image-level class labels.
    groundtruth_boxes: coordinates of the ground truth boxes in the image.
    groundtruth_classes: box-level class labels.
    groundtruth_label_types: box-level label types (e.g. explicit negative).
    groundtruth_is_crowd: [DEPRECATED, use groundtruth_group_of instead]
      is the groundtruth a single object or a crowd.
    groundtruth_area: area of a groundtruth segment.
    groundtruth_difficult: is a `difficult` object
    groundtruth_group_of: is a `group_of` objects, e.g. multiple objects of the
      same class, forming a connected group, where instances are heavily
      occluding each other.
    proposal_boxes: coordinates of object proposal boxes.
    proposal_objectness: objectness score of each proposal.
    groundtruth_instance_masks: ground truth instance masks.
    groundtruth_instance_boundaries: ground truth instance boundaries.
    groundtruth_instance_classes: instance mask-level class labels.
    groundtruth_keypoints: ground truth keypoints.
    groundtruth_keypoint_visibilities: ground truth keypoint visibilities.
    groundtruth_label_scores: groundtruth label scores.
    groundtruth_weights: groundtruth weight factor for bounding boxes.
    num_groundtruth_boxes: number of groundtruth boxes.
    true_image_shapes: true shapes of images in the resized images, as resized
      images can be padded with zeros.
    multiclass_scores: the label score per class for each box.
  """
  image = 'image'
  image_additional_channels = 'image_additional_channels'
  original_image = 'original_image'
  key = 'key'
  source_id = 'source_id'
  filename = 'filename'
  groundtruth_image_classes = 'groundtruth_image_classes'
  groundtruth_boxes = 'groundtruth_boxes'
  groundtruth_classes = 'groundtruth_classes'
  groundtruth_label_types = 'groundtruth_label_types'
  groundtruth_is_crowd = 'groundtruth_is_crowd'
  groundtruth_area = 'groundtruth_area'
  groundtruth_difficult = 'groundtruth_difficult'
  groundtruth_group_of = 'groundtruth_group_of'
  proposal_boxes = 'proposal_boxes'
  proposal_objectness = 'proposal_objectness'
  groundtruth_instance_masks = 'groundtruth_instance_masks'
  groundtruth_instance_boundaries = 'groundtruth_instance_boundaries'
  groundtruth_instance_classes = 'groundtruth_instance_classes'
  groundtruth_keypoints = 'groundtruth_keypoints'
  groundtruth_keypoint_visibilities = 'groundtruth_keypoint_visibilities'
  groundtruth_label_scores = 'groundtruth_label_scores'
  groundtruth_weights = 'groundtruth_weights'
  num_groundtruth_boxes = 'num_groundtruth_boxes'
  true_image_shape = 'true_image_shape'
  multiclass_scores = 'multiclass_scores'

  class CenterNetCenterHeatmapTargetAssigner(object):
  """Wrapper to compute the object center heatmap."""

  def __init__(self, stride, min_overlap=0.7, compute_heatmap_sparse=False):
    """Initializes the target assigner.
    Args:
      stride: int, the stride of the network in output pixels.
      min_overlap: The minimum IOU overlap that boxes need to have to not be
        penalized.
      compute_heatmap_sparse: bool, indicating whether or not to use the sparse
        version of the Op that computes the heatmap. The sparse version scales
        better with number of classes, but in some cases is known to cause
        OOM error. See (b/170989061).
    """

    self._stride = stride
    self._min_overlap = min_overlap
    self._compute_heatmap_sparse = compute_heatmap_sparse

  def assign_center_targets_from_boxes(self,
                                       height,
                                       width,
                                       gt_boxes_list,
                                       gt_classes_list,
                                       gt_weights_list=None):
    """Computes the object center heatmap target.
    Args:
      height: int, height of input to the model. This is used to
        determine the height of the output.
      width: int, width of the input to the model. This is used to
        determine the width of the output.
      gt_boxes_list: A list of float tensors with shape [num_boxes, 4]
        representing the groundtruth detection bounding boxes for each sample in
        the batch. The box coordinates are expected in normalized coordinates.
      gt_classes_list: A list of float tensors with shape [num_boxes,
        num_classes] representing the one-hot encoded class labels for each box
        in the gt_boxes_list.
      gt_weights_list: A list of float tensors with shape [num_boxes]
        representing the weight of each groundtruth detection box.
    Returns:
      heatmap: A Tensor of size [batch_size, output_height, output_width,
        num_classes] representing the per class center heatmap. output_height
        and output_width are computed by dividing the input height and width by
        the stride specified during initialization.
    """

    out_height = tf.cast(height // self._stride, tf.float32)
    out_width = tf.cast(width // self._stride, tf.float32)
    # Compute the yx-grid to be used to generate the heatmap. Each returned
    # tensor has shape of [out_height, out_width]
    (y_grid, x_grid) = ta_utils.image_shape_to_grids(out_height, out_width)

    heatmaps = []
    if gt_weights_list is None:
      gt_weights_list = [None] * len(gt_boxes_list)
    # TODO(vighneshb) Replace the for loop with a batch version.
    for boxes, class_targets, weights in zip(gt_boxes_list, gt_classes_list,
                                             gt_weights_list):
      boxes = box_list.BoxList(boxes)
      # Convert the box coordinates to absolute output image dimension space.
      boxes = box_list_ops.to_absolute_coordinates(boxes,
                                                   height // self._stride,
                                                   width // self._stride)
      # Get the box center coordinates. Each returned tensors have the shape of
      # [num_instances]
      (y_center, x_center, boxes_height,
       boxes_width) = boxes.get_center_coordinates_and_sizes()

      # Compute the sigma from box size. The tensor shape: [num_instances].
      sigma = _compute_std_dev_from_box_size(boxes_height, boxes_width,
                                             self._min_overlap)
      # Apply the Gaussian kernel to the center coordinates. Returned heatmap
      # has shape of [out_height, out_width, num_classes]
      heatmap = ta_utils.coordinates_to_heatmap(
          y_grid=y_grid,
          x_grid=x_grid,
          y_coordinates=y_center,
          x_coordinates=x_center,
          sigma=sigma,
          channel_onehot=class_targets,
          channel_weights=weights,
          sparse=self._compute_heatmap_sparse)
      heatmaps.append(heatmap)

    # Return the stacked heatmaps over the batch.
    return tf.stack(heatmaps, axis=0)

def _to_float32(x):
  return tf.cast(x, tf.float32)

def _flatten_spatial_dimensions(batch_images):
  batch_size, height, width, channels = _get_shape(batch_images, 4)
  return tf.reshape(batch_images, [batch_size, height * width,
                                   channels])

def get_num_instances_from_weights(groundtruth_weights_list):
  """Computes the number of instances/boxes from the weights in a batch.
  Args:
    groundtruth_weights_list: A list of float tensors with shape
      [max_num_instances] representing whether there is an actual instance in
      the image (with non-zero value) or is padded to match the
      max_num_instances (with value 0.0). The list represents the batch
      dimension.
  Returns:
    A scalar integer tensor incidating how many instances/boxes are in the
    images in the batch. Note that this function is usually used to normalize
    the loss so the minimum return value is 1 to avoid weird behavior.
  """
  num_instances = tf.reduce_sum(
      [tf.math.count_nonzero(w) for w in groundtruth_weights_list])
  num_instances = tf.maximum(num_instances, 1)
  return num_instances

  

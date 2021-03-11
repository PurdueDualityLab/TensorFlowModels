import tensorflow as tf

try:
  # Try to get TF 1.x version if possible
  import tensorflow.compat.v1 as tf_v1
  absolute_difference = tf_v1.losses.absolute_difference
except (ImportError, AttributeError):
  # The following code was adapted from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/losses/losses_impl.py
  from tensorflow.python.keras.utils import losses_utils

  def absolute_difference(
      labels, predictions, weights=1.0,
      reduction=tf.keras.losses.Reduction.SUM_BY_NONZERO_WEIGHTS):
    """Adds an Absolute Difference loss to the training procedure.
    `weights` acts as a coefficient for the loss. If a scalar is provided, then
    the loss is simply scaled by the given value. If `weights` is a `Tensor` of
    shape `[batch_size]`, then the total loss for each sample of the batch is
    rescaled by the corresponding element in the `weights` vector. If the shape
    of `weights` matches the shape of `predictions`, then the loss of each
    measurable element of `predictions` is scaled by the corresponding value of
    `weights`.
    Args:
      labels: The ground truth output tensor, same dimensions as 'predictions'.
      predictions: The predicted outputs.
      weights: Optional `Tensor` whose rank is either 0, or the same rank as
        `labels`, and must be broadcastable to `labels` (i.e., all dimensions
        must be either `1`, or the same as the corresponding `losses`
        dimension).
      reduction: Type of reduction to apply to loss.
    Returns:
      Weighted loss float `Tensor`. If `reduction` is `NONE`, this has the same
      shape as `labels`; otherwise, it is scalar.
    Raises:
      ValueError: If the shape of `predictions` doesn't match that of
        `labels` or if the shape of `weights` is invalid or if `labels`
        or `predictions` is None.
    """
    if labels is None:
      raise ValueError("labels must not be None.")
    if predictions is None:
      raise ValueError("predictions must not be None.")
    with tf.name_scope(scope, "absolute_difference",
                        (predictions, labels, weights)) as scope:
      predictions = tf.cast(predictions, dtype=tf.float32)
      labels = tf.cast(labels, dtype=tf.float32)
      predictions.get_shape().assert_is_compatible_with(labels.get_shape())
      losses = tf.abs(tf.subtract(predictions, labels))
      return losses_utils.compute_weighted_loss(
          losses, weights, reduction=reduction)

class L1LocalizationLoss(tf.keras.losses.Loss):
  """L1 loss or absolute difference.
  When used in a per-pixel manner, each pixel should be given as an anchor.
  """

  def __call__(self, y_true, y_pred, sample_weight=None):
    """Compute loss function.
    Args:
      y_true: A float tensor of shape [batch_size, num_anchors]
        representing the regression targets
      y_pred: A float tensor of shape [batch_size, num_anchors]
        representing the (encoded) predicted locations of objects.
      sample_weight: a float tensor of shape [batch_size, num_anchors]
    Returns:
      loss: a float tensor of shape [batch_size, num_anchors] tensor
        representing the value of the loss function.
    """
    return absolute_difference(
        y_true,
        y_pred,
      #  weights=sample_weight,
        reduction=self._get_reduction()
    )

  call = __call__

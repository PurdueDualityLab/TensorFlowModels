# may be a dupe of retinanet_losses. need to look later
import tensorflow as tf

class PenaltyReducedLogisticFocalLoss(tf.keras.losses.Loss):
  """Penalty-reduced pixelwise logistic regression with focal loss.
  The loss is defined in Equation (1) of the Objects as Points[1] paper.
  Although the loss is defined per-pixel in the output space, this class
  assumes that each pixel is an anchor to be compatible with the base class.
  [1]: https://arxiv.org/abs/1904.07850
  """

  def __init__(self, *args, alpha=2.0, beta=4.0, sigmoid_clip_value=1e-4, **kwargs):
    """Constructor.
    Args:
      alpha: Focussing parameter of the focal loss. Increasing this will
        decrease the loss contribution of the well classified examples.
      beta: The local penalty reduction factor. Increasing this will decrease
        the contribution of loss due to negative pixels near the keypoint.
      sigmoid_clip_value: The sigmoid operation used internally will be clipped
        between [sigmoid_clip_value, 1 - sigmoid_clip_value)
    """
    self._alpha = alpha
    self._beta = beta
    self._sigmoid_clip_value = sigmoid_clip_value
    super(PenaltyReducedLogisticFocalLoss, self).__init__(*args, **kwargs)

  def call(self, y_true, y_pred):
    """Compute loss function.
    In all input tensors, `num_anchors` is the total number of pixels in the
    the output space.
    Args:
      prediction_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing the predicted unscaled logits for each class.
        The function will compute sigmoid on this tensor internally.
      target_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing a tensor with the 'splatted' keypoints,
        possibly using a gaussian kernel. This function assumes that
        the target is bounded between [0, 1].
      weights: a float tensor of shape, either [batch_size, num_anchors,
        num_classes] or [batch_size, num_anchors, 1]. If the shape is
        [batch_size, num_anchors, 1], all the classses are equally weighted.
    Returns:
      loss: a float tensor of shape [batch_size, num_anchors, num_classes]
        representing the value of the loss function.
    """

    target_tensor = y_true

    is_present_tensor = tf.math.equal(target_tensor, 1.0)
    prediction_tensor = tf.clip_by_value(tf.sigmoid(y_pred),
                                         self._sigmoid_clip_value,
                                         1 - self._sigmoid_clip_value)

    positive_loss = (tf.math.pow((1 - prediction_tensor), self._alpha)*
                     tf.math.log(prediction_tensor))
    negative_loss = (tf.math.pow((1 - target_tensor), self._beta)*
                     tf.math.pow(prediction_tensor, self._alpha)*
                     tf.math.log(1 - prediction_tensor))

    loss = -tf.where(is_present_tensor, positive_loss, negative_loss)
    return loss

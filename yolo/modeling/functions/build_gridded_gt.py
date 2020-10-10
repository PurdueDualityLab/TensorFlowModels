import tensorflow as tf
from tensorflow.keras import backend as K


@tf.function
def build_gridded_gt_v1(y_true: tf.Tensor, size: int,
                        true_shape: tf.Tensor) -> tf.Tensor:
    """
    Convert ground truth for use in loss functions.
    Args:
        y_true: tf.Tensor ground truth
        size: dimensions of grid
    """
    batches = None  # figure out batches dimension
    num_boxes = None  # figure out number of boxes
    #gridded = tf.zeros()
    for batch in range(batches):
        for box in range(num_boxes):
            pass

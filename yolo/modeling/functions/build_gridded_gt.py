import tensorflow as tf
from tensorflow.keras import backend as K


@tf.function
def build_gridded_gt_v1(y_true: tf.Tensor, mask: list[int], size: int,
                        true_shape: tf.Tensor, use_tie_breaker: bool
                        ) -> tf.Tensor:
    pass
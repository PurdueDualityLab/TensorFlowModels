import tensorflow as tf


def gaussian_penalty(radius: int, type=tf.float32) -> tf.Tensor:
    """
    This represents the penalty reduction around a point.
    Params:
        radius (int): integer for radius of penalty reduction
        type (tf.dtypes.DType): datatype of returned tensor
    Returns:
        tf.Tensor of shape (2 * radius + 1, 2 * radius + 1).
    """
    width = 2 * radius + 1
    sigma = radius / 3
    x = tf.reshape(tf.range(width, dtype=type) - radius, (width, 1))
    y = tf.reshape(tf.range(width, dtype=type) - radius, (1, width))
    exponent = (-1 * (x ** 2) - (y ** 2)) / (2 * sigma ** 2)
    return tf.math.exp(exponent)

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow_addons.image import utils as img_utils
import tensorflow.keras.backend as K
import random

# Global Variable to introduce randomness among each element of a batch
RANDOM_SEED = tf.random.Generator.from_seed(int(random.randint(300, 9000)))


def _jitter_rand(box_jitter=0.005):
    """Image Normalization.
    Returns:
        jitter(tensorflow.python.framework.ops.Tensor): A random number generated
            from a uniform distrubution between -0.3 and 0.3.
        randscale(tensorflow.python.framework.ops.Tensor): A random integer between
            -10 and 19.
    """
    global RANDOM_SEED
    jitter_cx = RANDOM_SEED.uniform(minval=-box_jitter,
                                    maxval=box_jitter,
                                    shape=(),
                                    dtype=tf.float32)
    jitter_cy = RANDOM_SEED.uniform(minval=-box_jitter,
                                    maxval=box_jitter,
                                    shape=(),
                                    dtype=tf.float32)
    jitter_bw = RANDOM_SEED.uniform(
        minval=-box_jitter, maxval=box_jitter, shape=(),
        dtype=tf.float32) + 1.0
    jitter_bh = RANDOM_SEED.uniform(
        minval=-box_jitter, maxval=box_jitter, shape=(),
        dtype=tf.float32) + 1.0
    return jitter_cx, jitter_cy, jitter_bw, jitter_bh



def _translate_rand(image_jitter=0.1):
    global RANDOM_SEED
    translate_x = RANDOM_SEED.uniform(minval=-image_jitter,
                                      maxval=image_jitter,
                                      shape=(),
                                      dtype=tf.float32)
    translate_y = RANDOM_SEED.uniform(minval=-image_jitter,
                                      maxval=image_jitter,
                                      shape=(),
                                      dtype=tf.float32)
    return translate_x, translate_y


def _box_scale_rand(min_val=10, max_val=19, randscale=13, frac_dat_scale=0.5):
    global RANDOM_SEED
    scale_q = RANDOM_SEED.uniform(minval=0,
                                  maxval=tf.cast(1 / frac_dat_scale,
                                                 dtype=tf.int32),
                                  shape=(),
                                  dtype=tf.int32)
    if scale_q == 0:
        randscale = RANDOM_SEED.uniform(minval=10,
                                        maxval=19,
                                        shape=(),
                                        dtype=tf.int32)
    return randscale


def _rand_number(low, high):
    """Generates a random number along a uniform distrubution.
    Args:
        low(tensorflow.python.framework.ops.Tensor): Minimum Value of the Distrubution.
        high(tensorflow.python.framework.ops.EagerTensor): Maximum Value of the Distrubution.
    Returns:
        A tensor of the specified shape filled with random uniform values.
    """
    global RANDOM_SEED  # Global Variable defined at the beginning of the file.
    return RANDOM_SEED.uniform(minval=low,
                               maxval=high,
                               shape=(),
                               dtype=tf.float32)

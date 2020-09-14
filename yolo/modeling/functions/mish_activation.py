import tensorflow as tf
import tensorflow.keras as ks

@tf.function
def mish(x):
    return x * tf.math.tanh(ks.activations.softplus(x))
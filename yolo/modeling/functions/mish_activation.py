import tensorflow as tf
import tensorflow.keras as ks

# with out relax shapes it seems to trigger retracing
# @tf.function(experimental_relax_shapes=True)
# def mish(x):
#     return x * tf.math.tanh(ks.activations.softplus(x))

class mish(ks.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return 
    
    def call(self, x):
        return x * tf.math.tanh(ks.activations.softplus(x))

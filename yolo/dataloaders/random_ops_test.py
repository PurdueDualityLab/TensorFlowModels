from .random_ops import *

def test_rand_jitter():
    jitter = tf.py_function(jitter_rand, [0.1, 0.005], [tf.float32,  tf.float32,tf.float32,  tf.float32,tf.float32,  tf.float32])
    print(jitter[0], jitter[1], jitter[2], jitter[3], jitter[4], jitter[5])

def test_rand_scale():
    randscale = tf.py_function(box_scale_rand, [13, 0.5], [tf.int32])
    print(randscale)

for i in range(20):
    test_rand_scale()
test_rand_jitter()
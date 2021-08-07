"""A set of private math operations used to safely implement the yolo loss"""
import tensorflow as tf


def rm_nan_inf(x, val=0.0):
  """remove nan and infinity   

  Args:
    x: any `Tensor` of any type. 
    val: value to replace nan and infinity with. 

  Return:
    a `Tensor` with nan and infinity removed.
  """
  cond = tf.math.logical_or(tf.math.is_nan(x), tf.math.is_inf(x))
  val = tf.cast(val, dtype=x.dtype)
  x = tf.where(cond, val, x)
  return x


def rm_nan(x, val=0.0):
  """remove nan and infinity.   

  Args:
    x: any `Tensor` of any type. 
    val: value to replace nan. 

  Return:
    a `Tensor` with nan removed.
  """
  cond = tf.math.is_nan(x)
  val = tf.cast(val, dtype=x.dtype)
  x = tf.where(cond, val, x)
  return x


def divide_no_nan(a, b):
  """Nan safe divide operation built to allow model compilation in tflite. 

  Args:
    a: any `Tensor` of any type.
    b: any `Tensor` of any type with the same shape as tensor a. 

  Return:
    a `Tensor` representing a divided by b, with all nan values removed. 
  """
  # zero = tf.cast(0.0, b.dtype)
  # return tf.where(b == zero, zero, a / b)
  return a / (b + 1e-9)
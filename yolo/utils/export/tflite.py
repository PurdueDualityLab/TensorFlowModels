import tensorflow as tf


class tflite(object):

  def __init__(self,
               model_path,
               delegates=None,
               num_threads=2,
               input_dict_map=None,
               output_dict_map=None):
    self._interpreter = tf.lite.Interpreter(
        model_path=model_path,
        experimental_delegates=delegates,
        num_threads=num_threads)
    self._interpreter.allocate_tensors()
    self._id = self._interpreter.get_input_details()
    self._od = self._interpreter.get_output_details()
    self._in_struct = input_dict_map
    self._out_struct = output_dict_map
    return

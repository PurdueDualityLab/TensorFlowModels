import tensorflow as tf
from centernet.modeling.layers.nn_blocks import CenterNetHeadConv

"""
  # OLD: use to reference task_outputs dictionary, maybe can go in config
  # class later
  task_output_specs = {
    "2D" : {'heatmap': 91,
      'local_offset': 2, 
      'object_size': 2
    },
    "3D": {'heatmap': 91,
      'local_offset': 2, 
      'object_size': 3, 
      'depth': 1, 
      'orientation': 8
    },
    "pose": {'heatmap': 91,
      'joint_locs': 17 * 2, 
      'joint_heatmap': 17, 
      'joint_offset': 2
    }
  }
"""

class CenterNetHead(tf.keras.layers.Layer):
  """
  CenterNet Head
  """
  def __init__(self, 
               task_outputs: dict,
               heatmap_bias: float = -2.19,
               **kwargs):
    """
    Args:
      task_outputs: dict, with key-value pairs denoting the names of the outputs
      and the desired channel depth of each output
      heatmap_bias: float, constant value to initialize the convolution layer 
        bias vector if it is responsible for generating a heatmap (not for 
        regressed predictions)

    call Returns:
      dictionary where the keys-value pairs denote the names of the output 
      and the respective output tensor
    """
  
    self._task_outputs = task_outputs
    self._heatmap_bias = heatmap_bias

    super().__init__(**kwargs)
  
  def build(self, input_shape):
    self.layers = {}
    for key in self._task_outputs:
      num_filters = self._task_outputs[key]
      bias = 0
      if key is "heatmap":
        bias = self._heatmap_bias

      self.layers[key] = CenterNetHeadConv(output_filters=num_filters, 
        name=key, bias_init=bias)
    
    super().build(input_shape)

  def call(self, x):
    outputs = {}
    for key in self.layers:
      outputs[key] = self.layers[key](x)

    return outputs
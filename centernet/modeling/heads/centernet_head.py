import tensorflow as tf
from centernet.modeling.layers.nn_blocks import CenterNetHeadConv

class CenterNetHead(tf.keras.layers.Layer):
  """
  CenterNet Head
  """
  def __init__(self, 
               classes: int = 91,
               joints: int = 17,
               task: str = "2D",
               **kwargs):
    """
    Args:
      classes: int, number of possible class predictions for the network
      joints: int, number of possible joint location predictions for pose 
        estimation
      task: string, indicating the prediction task. Valid values are
        "2D", "3D", and "pose".

    call Returns:
      dictionary where the keys-value pairs are the output names and output 
        tensors
    """
    self._classes = classes
    self._joints = joints
    self._task = task

    # These specify the layers required for each type of task
    # Each spec is a dictionary of the output name and its 
    # respective channel depth
    task_output_specs = {
      "2D" : {'heatmaps': self._classes,
        'local_offset': 2, 
        'object_size': 2
      },
      "3D": {'heatmaps': self._classes,
        'local_offset': 2, 
        'object_size': 3, 
        'depth': 1, 
        'orientation': 8
      },
      "pose": {'heatmaps': self._classes,
        'joint_locs': self._joints * 2, 
        'joint_heatmap': self._joints, 
        'joint_offset': 2
      }
    }

    self.task_outputs = task_output_specs[self._task]

    super().__init__(**kwargs)
  
  def build(self, input_shape):
    self.layers = {}
    for key in self.task_outputs:
      num_filters = self.task_outputs[key]
      self.layers[key] = CenterNetHeadConv(output_filters=num_filters, name=key)
    
    super().build(input_shape)

  def call(self, x):
    outputs = {}
    for key in self.layers:
      outputs[key] = self.layers[key](x)

    return outputs
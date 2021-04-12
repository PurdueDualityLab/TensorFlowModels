import tensorflow as tf

from centernet.configs import centernet as cfg
from centernet.modeling.layers.nn_blocks import CenterNetDecoderConv


@tf.keras.utils.register_keras_serializable(package='centernet')
class CenterNetDecoder(tf.keras.Model):
  """
  CenterNet Decoder
  """
  def __init__(self,
               input_specs, 
               task_outputs: dict,
               heatmap_bias: float = -2.19,
               num_inputs: int = 2,
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
    self._input_specs = input_specs
    self._task_outputs = task_outputs
    self._heatmap_bias = heatmap_bias
    self._num_inputs = num_inputs


    inputs = [tf.keras.layers.Input(shape=value[1:]) for value in self._input_specs]
    outputs = dict()

    for key in self._task_outputs:
      num_filters = self._task_outputs[key]
      bias = 0
      if 'heatmaps' in key:
        bias = self._heatmap_bias
        
      outputs[key] = [CenterNetDecoderConv(output_filters=num_filters,
        name=key + str(i), bias_init=bias)(inputs[i]) for i in range(self._num_inputs)]

    self._output_specs = {
      key: [value[i].get_shape() for i in range(num_inputs)] 
        for key, value in outputs.items()
    }

    super().__init__(inputs=inputs, outputs=outputs, name='CenterNetDecoder')

  def get_config(self):
    layer_config = {
      'task_outputs': self._task_outputs,
      'heatmap_bias': self._heatmap_bias
    }

    return layer_config

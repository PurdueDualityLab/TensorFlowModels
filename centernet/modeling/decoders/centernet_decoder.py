import tensorflow as tf
from centernet.modeling.layers.nn_blocks import CenterNetDecoderConv
from centernet.configs import centernet as cfg


class CenterNetDecoder(tf.keras.Model):
  """
  CenterNet Decoder
  """
  def __init__(self,
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
    super().__init__(**kwargs)

    self._task_outputs = task_outputs
    self._heatmap_bias = heatmap_bias
    self._num_inputs = num_inputs

    self.outputs_layers = {}

  def build(self, inputs):
    for key in self._task_outputs:
      num_filters = self._task_outputs[key]
      bias = 0
      if 'heatmaps' in key:
        bias = self._heatmap_bias

      self.outputs_layers[key] = [CenterNetDecoderConv(output_filters=num_filters,
        name=key, bias_init=bias) for _ in range(self._num_inputs)] 

  def call(self, inputs):
    outputs = {}

    for key in self._task_outputs:
      outputs[key] = [self.outputs_layers[key][i](inputs[i]) for i in range(self._num_inputs)]
    
    return outputs

  def get_config(self):
    layer_config = {
      'task_outputs': self._task_outputs,
      'heatmap_bias': self._heatmap_bias
    }

    #layer_config.update(super().get_config())
    return layer_config

def build_centernet_decoder(input_specs, task_config, num_inputs):
  # NOTE: For now just support the default config
  
  # model specific 
  heatmap_bias = task_config.model.base.decoder.heatmap_bias
  
  # task specific
  task_outputs = task_config._get_output_length_dict()
  model = CenterNetDecoder(
      task_outputs=task_outputs,
      heatmap_bias=heatmap_bias,
      num_inputs=num_inputs)
  
  model.build(input_specs)
  return model
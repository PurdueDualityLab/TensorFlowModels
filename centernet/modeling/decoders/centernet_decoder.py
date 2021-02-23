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
    self.out_layers = {}
    for key in self._task_outputs:
      num_filters = self._task_outputs[key]
      bias = 0
      if 'heatmaps' in key:
        bias = self._heatmap_bias

      self.out_layers[key] = CenterNetDecoderConv(output_filters=num_filters,
        name=key, bias_init=bias)

    super().build(input_shape)

  def call(self, x):
    outputs = {}
    for key in self.out_layers:
      outputs[key] = self.out_layers[key](x)

    return outputs

  def get_config(self):
    layer_config = {
      'task_outputs': self._task_outputs,
      'heatmap_bias': self._heatmap_bias
    }

    #layer_config.update(super().get_config())
    return layer_config

def build_centernet_decoder(input_specs, task_config):
  # NOTE: For now just support the default config
  
  # model specific 
  heatmap_bias = task_config.model.base.decoder.heatmap_bias
  
  # task specific
  task_outputs = task_config._get_output_length_dict()
  model = CenterNetDecoder(
      task_outputs=task_outputs,
      heatmap_bias=heatmap_bias)
  
  model.build(input_specs)
  return model
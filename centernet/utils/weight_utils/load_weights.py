import tensorflow as tf
import numpy as np

from centernet.modeling.CenterNet import build_centernet
from centernet.configs.centernet import CenterNetTask
from centernet.utils.weight_utils.config_classes import convBnCFG, residualBlockCFG, hourglassCFG, decoderConvCFG

from centernet.modeling.layers.nn_blocks import ConvBN, HourglassBlock, CenterNetDecoderConv
from official.vision.beta.modeling.layers.nn_blocks import ResidualBlock

CKPT_PATH = 'D:\\weights\centernet_hg104_512x512_coco17_tpu-8\checkpoint'
SAVED_MODEL_PATH = 'D:\\weights\centernet_hg104_512x512_coco17_tpu-8\saved_model'

def print_dict_as_tree(dictionary, spaces):
  if type(dictionary) is not dict:
    return 

  else:
    for key in dictionary.keys():
      print(" " * spaces + key)
      print_dict_as_tree(dictionary[key], spaces + 2)

def print_layer_weights_and_shape(layer):
  weights = layer.get_weights()
  variables = layer.variables

  for i in range(len(weights)):
    tf.print(np.shape(weights[i]), variables[i].name)

def update_weights_dict(weights_dict, variable_key, value):
  """ Inserts weight value into a weight dictionary
    Args:
        weights_dict:
        variable_key:
        value:
    """
  current_dict = weights_dict
  variable_key_list = variable_key.split('/')

  key = variable_key_list.pop(0)
  while len(variable_key_list):
    if variable_key_list[0] == '.ATTRIBUTES':
      current_dict[key] = value
      return

    if key not in current_dict.keys():
      current_dict[key] = {}
    current_dict = current_dict[key]
    key = variable_key_list.pop(0)

def load_weights_model(model, weights_dict):
  print("Loading model weights\n")
  load_weights_backbone(model.backbone, 
    weights_dict['model']['_feature_extractor']['_network'])
  
  load_weights_decoder(model.decoder,
    weights_dict['model']['_prediction_head_dict'])
  print("Successfully loaded model weights\n")

def get_backbone_layer_cfgs(weights_dict):
  print("Fetching backbone config classes\n")
  cfgs = [
    # Downsampling Layers
    convBnCFG(weights_dict=weights_dict['downsample_input']['conv_block']),
    residualBlockCFG(weights_dict=weights_dict['downsample_input']['residual_block']),
    # Hourglass
    hourglassCFG(weights_dict=weights_dict['hourglass_network']['0']),
    convBnCFG(weights_dict=weights_dict['output_conv']['0']),
    # Intermediate
    convBnCFG(weights_dict=weights_dict['intermediate_conv1']['0']),
    convBnCFG(weights_dict=weights_dict['intermediate_conv2']['0']),
    residualBlockCFG(weights_dict=weights_dict['intermediate_residual']['0']),
    # Hourglass
    hourglassCFG(weights_dict=weights_dict['hourglass_network']['1']),
    convBnCFG(weights_dict=weights_dict['output_conv']['1']),
    ]

  return cfgs

def load_weights_backbone(backbone, weights_dict):
  print("Loading backbone weights\n")
  backbone_layers = backbone.layers
  cfgs = get_backbone_layer_cfgs(weights_dict)

  cfg = cfgs.pop(0)
  for i in range(len(backbone_layers)):
    layer = backbone_layers[i]    
    if isinstance(layer, (ConvBN, HourglassBlock, ResidualBlock)):
      print("Loading weights for: {}".format(cfg))
      cfg.load_weights(layer)
      if len(cfgs) == 0:
        print("Weights have been loaded for {} / {} layers\n".format(i+1, len(backbone_layers)))
        return
      cfg = cfgs.pop(0)

def get_decoder_layer_cfgs(weights_dict):
  print("Fetching decoder config classes\n")
  cfgs = [
    decoderConvCFG(weights_dict=weights_dict['object_center']['0']),
    decoderConvCFG(weights_dict=weights_dict['object_center']['1']),
    decoderConvCFG(weights_dict=weights_dict['box.Soffset']['0']),
    decoderConvCFG(weights_dict=weights_dict['box.Soffset']['1']),
    decoderConvCFG(weights_dict=weights_dict['box.Sscale']['0']),
    decoderConvCFG(weights_dict=weights_dict['box.Sscale']['1'])
  ]
  return cfgs

def load_weights_decoder(decoder, weights_dict):
  print("Loading decoder weights\n")
  decoder_layers = decoder.layers
  cfgs = get_decoder_layer_cfgs(weights_dict)

  cfg = cfgs.pop(0)
  for i in range(len(decoder_layers)):
    layer = decoder_layers[i]
    if isinstance(layer, CenterNetDecoderConv):
      print("Loading weights for: {}".format(cfg))
      cfg.load_weights(layer)
      if len(cfgs) == 0:
        print("Weights have been loaded for {} / {} layers\n".format(i+1, len(decoder_layers)))
        return
      cfg = cfgs.pop(0)

def get_model_weights_as_dict(ckpt_path):
  print("\nConverting model checkpoint from {} to weights dictionary\n".format(ckpt_path))
  reader = tf.train.load_checkpoint(ckpt_path)
  shape_from_key = reader.get_variable_to_shape_map()
  dtype_from_key = reader.get_variable_to_dtype_map()

  variable_keys = shape_from_key.keys()
  weights_dict = {}

  for key in variable_keys:
    shape = shape_from_key[key]
    dtype = dtype_from_key[key]
    value = reader.get_tensor(key)
    update_weights_dict(weights_dict, key, value)
  
  print("Successfully read checkpoint weights\n")
  return weights_dict

if __name__ == '__main__':
  input_specs = tf.keras.layers.InputSpec(shape=[None, 512, 512, 3])
  config = CenterNetTask()
  
  model, loss = build_centernet(input_specs=input_specs,
      task_config=config, l2_regularization=0)
      
  weights_dict = get_model_weights_as_dict(CKPT_PATH)
  load_weights_model(model, weights_dict)
  
  # print_dict_as_tree(weights_dict, 0)
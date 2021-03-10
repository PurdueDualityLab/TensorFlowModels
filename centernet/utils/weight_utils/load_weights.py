import tensorflow as tf
import numpy as np

from centernet.modeling.CenterNet import build_centernet
from centernet.configs.centernet import CenterNetTask

from centernet.utils.weight_utils.tf_to_dict import get_model_weights_as_dict, write_dict_as_tree
from centernet.utils.weight_utils.config_classes import convBnCFG, residualBlockCFG, hourglassCFG, decoderConvCFG
from centernet.utils.weight_utils.config_data import BackboneConfigData, DecoderConfigData

from centernet.modeling.layers.nn_blocks import ConvBN, HourglassBlock, CenterNetDecoderConv

from official.vision.beta.modeling.layers.nn_blocks import ResidualBlock

CKPT_PATH = 'D:\\weights\centernet_hg104_512x512_coco17_tpu-8\checkpoint'
SAVED_MODEL_PATH = 'D:\\weights\centernet_hg104_512x512_coco17_tpu-8\saved_model'

def load_weights_model(model, weights_dict, backbone_name, decoder_name):
  """ Loads weights into the model.

  Args:
    model: keras.Model to load weights into
    weights_dict: Dictionary that stores the weights of the model
    backbone_name: String, indicating the desired backbone configuration
    decoder_name: String, indicating the desired decoder configuration
  """
  print("Loading model weights\n")
  n_weights = 0
  n_weights += load_weights_backbone(model.backbone, 
    weights_dict['model']['_feature_extractor']['_network'],
    backbone_name)
  
  n_weights += load_weights_decoder(model.decoder,
    weights_dict['model']['_prediction_head_dict'],
    decoder_name)
  print("Successfully loaded {} model weights.\n".format(n_weights))

def get_backbone_layer_cfgs(weights_dict, backbone_name):
  """ Fetches the config classes for the backbone.

  This function generates a list of config classes corresponding to
  each building block in the backbone.

  Args:
    weights_dict: Dictionary that stores the backbone model weights
    backbone_name: String, indicating the desired backbone configuration
  Returns:
    A list containing the config classe of the backbone building block
  """

  print("Fetching backbone config classes for {}\n".format(backbone_name))
  cfgs = BackboneConfigData(weights_dict=weights_dict).get_cfg_list(backbone_name)
  return cfgs

def load_weights_backbone(backbone, weights_dict, backbone_name):
  """ Loads the weights defined in the weights_dict into the backbone.

  This function loads the backbone weights by first fetching the necesary
  config classes for the backbone, then loads them in one by one for 
  each layer that has weights associated with it.

  Args:
    backbone: keras.Model backbone
    weights_dict: Dictionary that stores the backbone model weights
    backbone_name: String, indicating the desired backbone configuration
  Returns:
    Number of weights loaded in 
  """
  print("Loading backbone weights\n")
  backbone_layers = backbone.layers
  cfgs = get_backbone_layer_cfgs(weights_dict, backbone_name)
  n_weights_total = 0

  cfg = cfgs.pop(0)
  for i in range(len(backbone_layers)):
    layer = backbone_layers[i]    
    if isinstance(layer, (ConvBN, HourglassBlock, ResidualBlock)):
      n_weights = cfg.load_weights(layer)
      print("Loading weights for: {}, weights loaded: {}".format(cfg, n_weights))
      n_weights_total += n_weights
      if len(cfgs) == 0:
        print("{} Weights have been loaded for {} / {} layers\n".format(n_weights_total, i+1, len(backbone_layers)))
        return n_weights_total
      cfg = cfgs.pop(0)

def get_decoder_layer_cfgs(weights_dict, decoder_name):
  """ Fetches the config classes for the decoder.

  This function generates a list of config classes corresponding to
  each building block in the decoder.

  Args:
    weights_dict: Dictionary that stores the decoder model weights
    decoder_name: String, indicating the desired decoder configuration
  Returns:
    A list containing the config classe of the backbone building block
  """
  print("Fetching decoder config classes\n")

  cfgs = DecoderConfigData(weights_dict=weights_dict).get_cfg_list(decoder_name)
  return cfgs

def load_weights_decoder(decoder, weights_dict, decoder_name):
  """ Loads the weights defined in the weights_dict into the decoder.

  This function loads the decoder weights by first fetching the necesary
  config classes for the decoder, then loads them in one by one for 
  each layer that has weights associated with it.

  Args:
    decoder: keras.Model decoder
    weights_dict: Dictionary that stores the decoder model weights
    decoder_name: String, indicating the desired decoder configuration
  Returns:
    Number of weights loaded in 
  """
  print("Loading decoder weights\n")
  decoder_layers = decoder.layers
  cfgs = get_decoder_layer_cfgs(weights_dict, decoder_name)
  n_weights_total = 0

  cfg = cfgs.pop(0)
  for i in range(len(decoder_layers)):
    layer = decoder_layers[i]
    if isinstance(layer, CenterNetDecoderConv):
      n_weights = cfg.load_weights(layer)
      print("Loading weights for: {}, weights loaded: {}".format(cfg, n_weights))
      n_weights_total += n_weights
      if len(cfgs) == 0:
        print("{} Weights have been loaded for {} / {} layers\n".format(n_weights_total, i+1, len(decoder_layers)))
        return n_weights_total
      cfg = cfgs.pop(0)

if __name__ == '__main__':
  input_specs = tf.keras.layers.InputSpec(shape=[1, 512, 512, 3])
  config = CenterNetTask()
  
  model, loss = build_centernet(input_specs=input_specs,
      task_config=config, l2_regularization=0)

  weights_dict, n_weights = get_model_weights_as_dict(CKPT_PATH)
  load_weights_model(model, weights_dict, 'hourglass104_512', 'detection_2d')
  
  # Note number of weights read and loaded differ by two because
  # we also read in the checkpoint save_counter and object_graph
  # that are not weights to the model

  # Uncomment line below to write weights dict key names to a file
  # write_dict_as_tree(weights_dict, filename="centernet/utils/weight_utils/MODEL_VARS.txt")
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import ClassVar, Dict, List, Tuple

import numpy as np


class Config(ABC):
  def get_weights(self):
    return None
  
  def load_weights(self, layer):
    weights = self.get_weights() 
    layer.set_weights(weights)
    n_weights = 0

    for w in weights:
      n_weights += w.size
    return n_weights

@dataclass
class convBnCFG(Config):
  weights_dict: Dict = field(repr=False, default=None)
  weights: np.array = field(repr=False, default=None)
  beta: np.array = field(repr=False, default=None)
  gamma: np.array = field(repr=False, default=None)
  moving_mean: np.array = field(repr=False, default=None)
  moving_variance: np.array = field(repr=False, default=None)

  def __post_init__(self):
    conv_weights_dict = self.weights_dict['conv'] 
    norm_weights_dict = self.weights_dict['norm']

    self.weights = conv_weights_dict['kernel']
    self.beta = norm_weights_dict['beta']
    self.gamma = norm_weights_dict['gamma']
    self.moving_mean = norm_weights_dict['moving_mean']
    self.moving_variance = norm_weights_dict['moving_variance']
  
  def get_weights(self):
    return [
          self.weights,
          self.gamma,
          self.beta,
          self.moving_mean,
          self.moving_variance
      ]

@dataclass
class residualBlockCFG(Config):
  weights_dict: Dict = field(repr=False, default=None)
  
  skip_weights: np.array = field(repr=False, default=None)
  skip_beta: np.array = field(repr=False, default=None)
  skip_gamma: np.array = field(repr=False, default=None)
  skip_moving_mean: np.array = field(repr=False, default=None)
  skip_moving_variance: np.array = field(repr=False, default=None)

  conv_weights: np.array = field(repr=False, default=None)
  norm_beta: np.array = field(repr=False, default=None)
  norm_gamma: np.array = field(repr=False, default=None)
  norm_moving_mean: np.array = field(repr=False, default=None)
  norm_moving_variance: np.array = field(repr=False, default=None)

  conv_block_weights: np.array = field(repr=False, default=None)
  conv_block_beta: np.array = field(repr=False, default=None)
  conv_block_gamma: np.array = field(repr=False, default=None)
  conv_block_moving_mean: np.array = field(repr=False, default=None)
  conv_block_moving_variance: np.array = field(repr=False, default=None)

  def __post_init__(self):
    conv_weights_dict = self.weights_dict['conv'] 
    norm_weights_dict = self.weights_dict['norm'] 
    conv_block_weights_dict = self.weights_dict['conv_block']

    if 'skip' in self.weights_dict:
      skip_weights_dict = self.weights_dict['skip'] 
      self.skip_weights = skip_weights_dict['conv']['kernel']
      self.skip_beta = skip_weights_dict['norm']['beta']
      self.skip_gamma = skip_weights_dict['norm']['gamma']
      self.skip_moving_mean = skip_weights_dict['norm']['moving_mean']
      self.skip_moving_variance = skip_weights_dict['norm']['moving_variance']

    self.conv_weights = conv_weights_dict['kernel']
    self.norm_beta = norm_weights_dict['beta']
    self.norm_gamma = norm_weights_dict['gamma']
    self.norm_moving_mean = norm_weights_dict['moving_mean']
    self.norm_moving_variance = norm_weights_dict['moving_variance']

    self.conv_block_weights = conv_block_weights_dict['conv']['kernel']
    self.conv_block_beta = conv_block_weights_dict['norm']['beta']
    self.conv_block_gamma = conv_block_weights_dict['norm']['gamma']
    self.conv_block_moving_mean = conv_block_weights_dict['norm']['moving_mean']
    self.conv_block_moving_variance = conv_block_weights_dict['norm']['moving_variance']

  def get_weights(self):
    weights = [
      self.skip_weights,
      self.skip_gamma,
      self.skip_beta,

      self.conv_block_weights,
      self.conv_block_gamma,
      self.conv_block_beta,

      self.conv_weights,
      self.norm_gamma,
      self.norm_beta,

      self.skip_moving_mean,
      self.skip_moving_variance,
      self.conv_block_moving_mean,
      self.conv_block_moving_variance,
      self.norm_moving_mean,
      self.norm_moving_variance,
    ]

    weights = [x for x in weights if x is not None]
    return weights

@dataclass
class hourglassCFG(Config):
  weights_dict: Dict = field(repr=False, default=None)
  is_last_stage: bool = field(repr=False, default=None)

  def __post_init__(self):
    self.is_last_stage = False if 'inner_block' in self.weights_dict else True
  
  def generate_block_weights(self, weights_dict):
    reps = len(weights_dict.keys())
    weights = []
    n_weights = 0

    for i in range(reps):
      res_config = residualBlockCFG(weights_dict=weights_dict[str(i)])
      res_weights = res_config.get_weights()
      weights += res_weights

      for w in res_weights:
        n_weights += w.size
    
    return weights, n_weights
    
  def load_block_weights(self, layer, weight_dict):
    block_weights, n_weights = self.generate_block_weights(weight_dict)
    layer.set_weights(block_weights)
    return n_weights

  def load_weights(self, layer):
    n_weights = 0

    if not self.is_last_stage:
      enc_dec_layers = [
        layer.submodules[0], 
        layer.submodules[1], 
        layer.submodules[3]
      ]
      enc_dec_weight_dicts = [
        self.weights_dict['encoder_block1'], 
        self.weights_dict['encoder_block2'],
        self.weights_dict['decoder_block']
      ]

      for l, weights_dict in zip(enc_dec_layers, enc_dec_weight_dicts):
        n_weights += self.load_block_weights(l, weights_dict)

      if len(self.weights_dict['inner_block']) == 1: # still in an outer hourglass
        inner_weights_dict = self.weights_dict['inner_block']['0']
      else:
        inner_weights_dict = self.weights_dict['inner_block'] # inner residual block chain

      inner_hg_layer = layer.submodules[2]
      inner_hg_cfg = type(self)(weights_dict=inner_weights_dict)
      n_weights += inner_hg_cfg.load_weights(inner_hg_layer)
    
    else:
      inner_layer = layer.submodules[0]
      n_weights += self.load_block_weights(inner_layer, self.weights_dict)
    
    return n_weights
  
@dataclass
class decoderConvCFG(Config):
  weights_dict: Dict = field(repr=False, default=None)

  conv_1_weights: np.array = field(repr=False, default=None)
  conv_2_bias: np.array = field(repr=False, default=None)

  conv_2_weights: np.array = field(repr=False, default=None)
  conv_1_bias: np.array = field(repr=False, default=None)

  def __post_init__(self):
    conv_1_weights_dict = self.weights_dict['layer_with_weights-0'] 
    conv_2_weights_dict = self.weights_dict['layer_with_weights-1'] 

    self.conv_1_weights = conv_1_weights_dict['kernel']
    self.conv_1_bias = conv_1_weights_dict['bias']
    self.conv_2_weights = conv_2_weights_dict['kernel']
    self.conv_2_bias = conv_2_weights_dict['bias']
  
  def get_weights(self):
    return [
      self.conv_1_weights,
      self.conv_1_bias,
      self.conv_2_weights,
      self.conv_2_bias
    ]

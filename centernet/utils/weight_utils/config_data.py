from dataclasses import dataclass, field
from typing import Dict

from centernet.utils.weight_utils.config_classes import (convBnCFG,
                                                         decoderConvCFG,
                                                         hourglassCFG,
                                                         residualBlockCFG)


@dataclass
class BackboneConfigData():
  weights_dict: Dict = field(repr=False, default=None)

  def get_cfg_list(self, name):
    if name == 'hourglass104_512':
      return [
        # Downsampling Layers
        convBnCFG(weights_dict=self.weights_dict['downsample_input']['conv_block']),
        residualBlockCFG(weights_dict=self.weights_dict['downsample_input']['residual_block']),
        # Hourglass
        hourglassCFG(weights_dict=self.weights_dict['hourglass_network']['0']),
        convBnCFG(weights_dict=self.weights_dict['output_conv']['0']),
        # Intermediate
        convBnCFG(weights_dict=self.weights_dict['intermediate_conv1']['0']),
        convBnCFG(weights_dict=self.weights_dict['intermediate_conv2']['0']),
        residualBlockCFG(weights_dict=self.weights_dict['intermediate_residual']['0']),
        # Hourglass
        hourglassCFG(weights_dict=self.weights_dict['hourglass_network']['1']),
        convBnCFG(weights_dict=self.weights_dict['output_conv']['1']),
      ]

    elif name == 'extremenet':
      return [
        # Downsampling Layers
        convBnCFG(weights_dict=self.weights_dict['downsample_input']['conv_block']),
        residualBlockCFG(weights_dict=self.weights_dict['downsample_input']['residual_block']),
        # Hourglass
        hourglassCFG(weights_dict=self.weights_dict['hourglass_network']['0']),
        convBnCFG(weights_dict=self.weights_dict['output_conv']['0']),
        # Intermediate
        convBnCFG(weights_dict=self.weights_dict['intermediate_conv1']['0']),
        convBnCFG(weights_dict=self.weights_dict['intermediate_conv2']['0']),
        residualBlockCFG(weights_dict=self.weights_dict['intermediate_residual']['0']),
        # Hourglass
        hourglassCFG(weights_dict=self.weights_dict['hourglass_network']['1']),
        convBnCFG(weights_dict=self.weights_dict['output_conv']['1']),
      ]

@dataclass
class DecoderConfigData():
  weights_dict: Dict = field(repr=False, default=None)    
  
  def get_cfg_list(self, name):
    if name == 'detection_2d':
      return [
        decoderConvCFG(weights_dict=self.weights_dict['object_center']['0']),
        decoderConvCFG(weights_dict=self.weights_dict['object_center']['1']),
        decoderConvCFG(weights_dict=self.weights_dict['box.Soffset']['0']),
        decoderConvCFG(weights_dict=self.weights_dict['box.Soffset']['1']),
        decoderConvCFG(weights_dict=self.weights_dict['box.Sscale']['0']),
        decoderConvCFG(weights_dict=self.weights_dict['box.Sscale']['1'])
      ]

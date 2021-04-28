import numpy as np
import tensorflow as tf

from centernet.configs.centernet import CenterNetTask
from centernet.modeling.CenterNet import build_centernet
from centernet.utils.weight_utils.load_weights import (
    get_model_weights_as_dict, load_weights_backbone, load_weights_model)
from centernet.validation.config_classes import (ODAPIconvBnCFG,
                                                 ODAPIhourglassCFG,
                                                 ODAPIresidualBlockCFG)
from centernet.validation.odapi_backbone import hourglass_104

CKPT_PATH = 'D:\\weights\centernet_hg104_512x512_coco17_tpu-8\checkpoint'

def load_weights_odapi_backbone(weights_dict):
  odapi_backbone = hourglass_104()
  odapi_backbone.build(input_shape=[1, 512, 512, 3])

  # load weights for downsample block (comprised of a conv and res block)
  downsample_conv_cfg = ODAPIconvBnCFG(weights_dict=weights_dict['downsample_input']['conv_block'])
  downsample_res_cfg = ODAPIresidualBlockCFG(weights_dict=weights_dict['downsample_input']['residual_block'])
  downsample_conv_cfg.load_weights(odapi_backbone.downsample_input.conv_block)
  downsample_res_cfg.load_weights(odapi_backbone.downsample_input.residual_block)

  # load weights for encoder/decoder blocks
  hourglass_0_cfg = ODAPIhourglassCFG(weights_dict=weights_dict['hourglass_network']['0'])
  hourglass_1_cfg = ODAPIhourglassCFG(weights_dict=weights_dict['hourglass_network']['1'])
  hourglass_0_cfg.load_weights(odapi_backbone.hourglass_network[0])
  hourglass_1_cfg.load_weights(odapi_backbone.hourglass_network[1])

  # load weights for hourglass output conv blocks
  out_conv_0_cfg = ODAPIconvBnCFG(weights_dict=weights_dict['output_conv']['0'])
  out_conv_1_cfg = ODAPIconvBnCFG(weights_dict=weights_dict['output_conv']['1'])
  out_conv_0_cfg.load_weights(odapi_backbone.output_conv[0])
  out_conv_1_cfg.load_weights(odapi_backbone.output_conv[1])

  # load weights for intermediate conv and residual blocks
  inter_conv_1_cfg = ODAPIconvBnCFG(weights_dict=weights_dict['intermediate_conv1']['0'])
  inter_conv_2_cfg = ODAPIconvBnCFG(weights_dict=weights_dict['intermediate_conv2']['0'])
  inter_res_2_cfg = ODAPIresidualBlockCFG(weights_dict=weights_dict['intermediate_residual']['0'])
  inter_conv_1_cfg.load_weights(odapi_backbone.intermediate_conv1[0])
  inter_conv_2_cfg.load_weights(odapi_backbone.intermediate_conv2[0])
  inter_res_2_cfg.load_weights(odapi_backbone.intermediate_residual[0])

  return odapi_backbone

@tf.function
def compare(model_1, model_2, input):
  out_1 = model_1(input)
  out_2 = model_2(input)
  return out_1, out_2

if __name__ == '__main__':
  # testing if the output between our backbone and the ODAPI backbone matches
  weights_dict, n_weights = get_model_weights_as_dict(CKPT_PATH)
  backbone_weights_dict = weights_dict['model']['_feature_extractor']['_network']
  
  # load weights into odapi backbone
  odapi_backbone = load_weights_odapi_backbone(backbone_weights_dict)
  
  # load weights into tfmg centernet model
  input_specs = tf.keras.layers.InputSpec(shape=[1, 512, 512, 3])
  config = CenterNetTask()
  model, loss = build_centernet(input_specs=input_specs,
      task_config=config, l2_regularization=0)
  load_weights_model(model, weights_dict, 'hourglass104_512', 'detection_2d')

  test_input = tf.random.uniform(
    shape=[1, 512, 512, 3], minval=0, maxval=1, dtype=tf.dtypes.float32, seed=45
  )

  tfmg_output, odapi_output = compare(odapi_backbone, model.backbone, test_input)
  tfmg_hm1 = tfmg_output[0]
  tfmg_hm2 = tfmg_output[1]
  odapi_hm1 = odapi_output[0]
  odapi_hm2 = odapi_output[1]
  print(tf.math.reduce_sum(odapi_hm2-tfmg_hm2))
  print(tf.math.reduce_sum(odapi_hm1-tfmg_hm1))

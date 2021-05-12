import numpy as np
import tensorflow as tf

from centernet.configs.centernet import CenterNetTask
from centernet.differential_testing.config_classes import (
    ODAPIconvBnCFG, ODAPIdecoderConvCFG, ODAPIhourglassCFG,
    ODAPIresidualBlockCFG)
from centernet.differential_testing.odapi_backbone import hourglass_104
from centernet.differential_testing.odapi_decoder import \
    _construct_prediction_heads
from centernet.differential_testing.odapi_detection_generator import \
    ODAPIDetectionGenerator
from centernet.modeling.CenterNet import build_centernet
from centernet.utils.weight_utils.load_weights import (
    get_model_weights_as_dict, load_weights_backbone, load_weights_model)

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

def load_weights_odapi_decoder(weights_dict):
  odapi_decoder = _construct_prediction_heads(90, 2, -2.19)
  for key in odapi_decoder.keys():
    for head in odapi_decoder[key]:
      head.build(input_shape=[1, 128, 128, 256])

  # load weights for downsample block (comprised of a conv and res block)
  center_0 = ODAPIdecoderConvCFG(weights_dict=weights_dict['object_center']['0'])
  center_1 = ODAPIdecoderConvCFG(weights_dict=weights_dict['object_center']['1'])
  offset_0 = ODAPIdecoderConvCFG(weights_dict=weights_dict['box.Soffset']['0'])
  offset_1 = ODAPIdecoderConvCFG(weights_dict=weights_dict['box.Soffset']['1'])
  scale_0 = ODAPIdecoderConvCFG(weights_dict=weights_dict['box.Sscale']['0'])
  scale_1 = ODAPIdecoderConvCFG(weights_dict=weights_dict['box.Sscale']['1'])

  center_0.load_weights(odapi_decoder['ct_heatmaps'][0])
  center_1.load_weights(odapi_decoder['ct_heatmaps'][1])
  offset_0.load_weights(odapi_decoder['ct_offset'][0])
  offset_1.load_weights(odapi_decoder['ct_offset'][1])
  scale_0.load_weights(odapi_decoder['ct_size'][0])
  scale_1.load_weights(odapi_decoder['ct_size'][1])

  return odapi_decoder

@tf.function 
def compare_det_gen(model_1, model_2, input):
  out_1 = model_1(input)
  out_2 = model_2(input)

  return out_1, out_2

@tf.function
def compare_decoder(model_1, prediction_heads, input):
  out_1 = model_1(input)
  out_2 = dict()

  for key in prediction_heads.keys():
    out_2[key] = [prediction_heads[key][i](input[i]) for i in range(len(input))]

  return out_1, out_2

@tf.function
def compare_backbone(model_1, model_2, input):
  out_1 = model_1(input)
  out_2 = model_2(input)
  return out_1, out_2

def differential_test_backbone(tfmg_backbone, odapi_backbone):
  print("\n Differential test for backbone \n")
  test_input = tf.random.uniform(
    shape=[1, 512, 512, 3], minval=0, maxval=1, dtype=tf.dtypes.float32, seed=1439
  )

  tfmg_output, odapi_output = compare_backbone(odapi_backbone, tfmg_backbone, test_input)
  tfmg_hm1 = tfmg_output[0]
  tfmg_hm2 = tfmg_output[1]
  odapi_hm1 = odapi_output[0]
  odapi_hm2 = odapi_output[1]
  print("Difference between final output feature maps: ", tf.math.reduce_sum(odapi_hm2-tfmg_hm2))
  print("Difference between intermediate hourglass feature maps: ", tf.math.reduce_sum(odapi_hm1-tfmg_hm1))

def differential_test_decoder(tfmg_decoder, odapi_decoder):
  print("\n Differential test for decoder \n")
  test_input = [
    tf.random.uniform(shape=[1, 128, 128, 256], minval=0, maxval=1, 
      dtype=tf.dtypes.float32, seed=41965) for _ in range(2)]

  tfmg_output, odapi_output = compare_decoder(tfmg_decoder, odapi_decoder, test_input)

  for key in tfmg_output.keys():
    print("For key: {}, difference between first decoded map: ".format(key))
    print(tf.math.reduce_sum(tfmg_output[key][0] - odapi_output[key][0]))
    print("For key: {}, difference between second decoded map: ".format(key))
    print(tf.math.reduce_sum(tfmg_output[key][1] - odapi_output[key][1]))

def differential_test_det_gen(tfmg_det_gen, odapi_det_gen):
  print("\n Differential test for detection generator \n")
  ct_heatmaps = [
    tf.random.uniform(shape=[1, 128, 128, 90], minval=0, maxval=1, 
      dtype=tf.dtypes.float32, seed=475) for _ in range(2)]
  offset_heatmaps = [
    tf.random.uniform(shape=[1, 128, 128, 2], minval=0, maxval=1, 
      dtype=tf.dtypes.float32, seed=425) for _ in range(2)]
  size_heatmaps = [
    tf.random.uniform(shape=[1, 128, 128, 2], minval=0, maxval=1, 
      dtype=tf.dtypes.float32, seed=145) for _ in range(2)]
  
  test_input = {
    'ct_heatmaps': ct_heatmaps,
    'ct_offset': offset_heatmaps,
    'ct_size': size_heatmaps
  }

  tfmg_output, odapi_output = compare_det_gen(model.filter, odapi_det_gen, test_input)
  print("all bounding box coodinate differences:", tf.math.reduce_sum(tfmg_output['bbox'] - odapi_output['bbox']))
  print("all class prediction difference: ", tf.math.reduce_sum(tfmg_output['classes'] - odapi_output['classes']))
  print("confidence score prediction difference: ", tf.math.reduce_sum(tfmg_output['confidence'] - odapi_output['confidence']))
  print("number detection difference: ", tf.math.reduce_sum(tfmg_output['num_dets'] - odapi_output['num_dets']))

if __name__ == '__main__':
  # testing if the output between our backbone and the ODAPI backbone matches
  weights_dict, n_weights = get_model_weights_as_dict(CKPT_PATH)
  backbone_weights_dict = weights_dict['model']['_feature_extractor']['_network']
  decoder_weights_dict = weights_dict['model']['_prediction_head_dict']
  
  # load weights into odapi backbone
  odapi_backbone = load_weights_odapi_backbone(backbone_weights_dict)

  # load weights into odapi decoder
  odapi_decoder = load_weights_odapi_decoder(decoder_weights_dict)

  # create odapi detection generator
  odapi_det_gen = ODAPIDetectionGenerator()
  
  # load weights into tfmg centernet model
  input_specs = tf.keras.layers.InputSpec(shape=[1, 512, 512, 3])
  config = CenterNetTask()
  model, loss = build_centernet(input_specs=input_specs,
      task_config=config, l2_regularization=0)
  load_weights_model(model, weights_dict, 'hourglass104_512', 'detection_2d')

  differential_test_backbone(model.backbone, odapi_backbone)
  differential_test_decoder(model.decoder, odapi_decoder)
  differential_test_det_gen(model.filter, odapi_det_gen)

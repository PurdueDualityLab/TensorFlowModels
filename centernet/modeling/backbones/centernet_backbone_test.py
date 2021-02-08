import tensorflow as tf
from centernet.modeling.layers.nn_blocks import HourglassBlock
from centernet.modeling.backbones.centernet_backbone import CenterNetBackbone

from centernet.modeling.backbones.backbone_builder import buildCenterNetBackbone
from centernet.modeling.backbones.config import CENTERNET_HG104_CFG

if __name__ == '__main__':
  hg_test_input_shape = (1, 512, 512, 256)
  bb_test_input_shape = (1, 512, 512, 3)
  x_hg = tf.ones(shape=hg_test_input_shape)
  x_bb = tf.ones(shape=bb_test_input_shape)

  tf.print("Testing Single hourglass")
  hg = HourglassBlock(filter_sizes=filter_sizes, rep_sizes=rep_sizes)

  hg.build(input_shape=hg_test_input_shape)
  tf.print('Made an hourglass!')
  out = hg(x_hg)
  tf.print('Hourglass module output shape:{} Expected shape:{}'.format(
      tf.shape(out), hg_test_input_shape))

  tf.print("Testing backbone")

  backbone = buildCenterNetBackbone(CENTERNET_HG104_CFG)

  backbone.build(input_shape=bb_test_input_shape)  

  # Backbone summary shows output shape to be multiple for hg modules
  # Maybe this is because the hg call method has a conditional?
  backbone.summary() 
  
  out_final, all_outs = backbone(x_bb)
  tf.print('Made backbone!')

  expected_out_filters = CENTERNET_HG104_CFG[-1][1][0] # last hourglass filters
  tf.print('Backbone output shape: {} Expected shape: {}'.format(
      tf.shape(out_final), (bb_test_input_shape[0], bb_test_input_shape[1]//8, 
                      bb_test_input_shape[2]//8, expected_out_filters)))

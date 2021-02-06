import tensorflow as tf
from centernet.modeling.layers.nn_blocks import HourglassBlock
from centernet.modeling.backbones.centernet_backbone import CenterNetBackbone

if __name__ == '__main__':
  filter_sizes = [256, 256, 384, 384, 384, 512]
  rep_sizes = [2, 2, 2, 2, 2, 4]
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
      hg.output_shape, hg_test_input_shape))

  tf.print("Testing backbone")
  backbone = CenterNetBackbone(
      n_stacks=2,
      downsample=True,
      filter_sizes=filter_sizes,
      rep_sizes=rep_sizes)

  backbone.build(input_shape=bb_test_input_shape)  

  # Backbone summary shows output shape to be multiple for hg modules
  # Maybe this is because the hg call method has a conditional?
  # This is also causing the final print statement to error.
  backbone.summary() 
  
  out = backbone(x_bb)
  tf.print(tf.shape(out))

  tf.print('Made backbone!')
  tf.print('Backbone output shape: {} Expected shape: {}'.format(
      backbone.output_shape, test_input_shape
  ))

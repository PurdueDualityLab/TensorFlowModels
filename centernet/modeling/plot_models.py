import tensorflow as tf

from centernet.configs import centernet as cfg
from centernet.modeling.backbones import hourglass
from centernet.modeling.CenterNet import build_centernet
from centernet.modeling.decoders import centernet_decoder
from centernet.modeling.layers import nn_blocks

BACKBONE_OUT = "./centernet/modeling/backbone.png"
DECODER_OUT = "./centernet/modeling/decoder.png"
MODEL_OUT = "./centernet/modeling/model.png"



if __name__ == '__main__':
  dims = [256, 256, 384, 384, 384, 512]
  modules = [2, 2, 2, 2, 2, 4]

  backbone = hourglass.build_hourglass(
        tf.keras.layers.InputSpec(shape=[None, 512, 512, 3]), cfg.CenterNetBase())

  hg_block = nn_blocks.HourglassBlock(dims, modules)
  hg_block.build((1, 512, 512, 256))
  
  decoder = centernet_decoder.build_centernet_decoder(
    task_config=cfg.CenterNetTask(), 
    input_specs=(None, 128, 128, 256))
  
  input_specs = tf.keras.layers.InputSpec(shape=[None, 512, 512, 3])

  config = cfg.CenterNetTask()
  model, loss = build_centernet(input_specs=input_specs,
    task_config=config, l2_regularization=0)
  
  tf.keras.utils.plot_model(backbone, to_file=BACKBONE_OUT, show_shapes=True, dpi=300)
  tf.keras.utils.plot_model(decoder, to_file=DECODER_OUT, show_shapes=True, dpi=300)

  hg_block.summary()
  model.summary()

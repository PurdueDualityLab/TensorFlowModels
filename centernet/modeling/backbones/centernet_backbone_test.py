import tensorflow as tf
from centernet.modeling.layers.nn_blocks   import HourglassBlock
from centernet.modeling.backbones.centernet_backbone import CenterNetBackbone

if __name__ == '__main__':
    order = 5
    filter_sizes = [256, 256, 384, 384, 384, 512]
    rep_sizes = [2, 2, 2, 2, 2, 4]

    # Testing single hourglass
    hg = HourglassBlock(order=order,
                   filter_sizes=filter_sizes,
                   rep_sizes=rep_sizes)

    test_input_shape = (1, 512, 512, 256)

    hg.build(input_shape=test_input_shape)
    tf.print("Made an hourglass!")

    x = tf.keras.Input(shape=(512, 512, 256), batch_size=1)
    out = hg(x)
    tf.print("Output shape of hourglass module is {} should be {}".format(hg.output_shape, test_input_shape))

    # Testing backbone
    backbone = CenterNetBackbone(n_stacks=2,
                                 pre_layers=None,
                                 order=order,
                                 filter_sizes=filter_sizes,
                                 rep_sizes=rep_sizes)
    backbone.build(input_shape=(1, 512, 512, 3))
    backbone.summary()
    tf.print("Made backbone!")

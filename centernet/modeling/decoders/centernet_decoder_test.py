import numpy as np
import tensorflow as tf
from absl.testing import parameterized

from centernet.configs import centernet as cfg
from centernet.modeling.decoders import centernet_decoder
from centernet.modeling.CenterNet import build_centernet_decoder


class CenterNetDecoderTest(tf.test.TestCase, parameterized.TestCase):

  def test_create_decoder(self):
    decoder = build_centernet_decoder(
      task_config=cfg.CenterNetTask(), 
      input_specs=[(None, 128, 128, 256), (None, 128, 128, 256)],
      num_inputs=2)

    config = decoder.get_config()
    self.assertEqual(len(config), 2)
    self.assertEqual(config['heatmap_bias'], -2.19)

  def test_decoder_shape(self):
    decoder = build_centernet_decoder(
      task_config=cfg.CenterNetTask(),
      input_specs=[(None, 128, 128, 256), (None, 128, 128, 256)],
      num_inputs=2)

    # Output shape tests
    outputs = decoder([np.zeros((2, 128, 128, 256), dtype=np.float32),
                       np.zeros((2, 128, 128, 256), dtype=np.float32)])
    self.assertEqual(len(outputs), 3)
    self.assertEqual(outputs['ct_heatmaps'][0].shape, (2, 128, 128, 90))
    self.assertEqual(outputs['ct_offset'][0].shape, (2, 128, 128, 2))
    self.assertEqual(outputs['ct_size'][0].shape, (2, 128, 128, 2))

    # Weight initialization tests
    tf.print("\n\n{}\n\n".format(decoder.layers))
    hm_bias_vector = np.asarray(decoder.layers[2].weights[-1])
    off_bias_vector = np.asarray(decoder.layers[4].weights[-1])
    size_bias_vector = np.asarray(decoder.layers[6].weights[-1])

    self.assertArrayNear(hm_bias_vector,
      np.repeat(-2.19, repeats=90), err=1.00e-6)
    self.assertArrayNear(off_bias_vector,
      np.repeat(0, repeats=2), err=1.00e-6)
    self.assertArrayNear(size_bias_vector,
      np.repeat(0, repeats=2), err=1.00e-6)

if __name__ == '__main__':
  tf.test.main()

import numpy as np
import tensorflow as tf
from absl.testing import parameterized

from yolo.ops import preprocessing_ops


class InputUtilsTest(parameterized.TestCase, tf.test.TestCase):
  @parameterized.parameters(([1, 2], 20, 0), ([13, 2, 4], 15, 0))
  def testPadMaxInstances(self,
                          input_shape,
                          instances,
                          pad_axis):
    expected_output_shape = input_shape
    expected_output_shape[pad_axis] = instances
    output = preprocessing_ops.pad_max_instances(
        np.ones(input_shape), instances, pad_axis=pad_axis)
    self.assertAllEqual(expected_output_shape,
                        tf.shape(output).numpy())

  @parameterized.parameters((100, 200))
  def testGetImageShape(self, image_height, image_width):
    image = tf.convert_to_tensor(np.random.rand(image_height, image_width, 3))
    image_shape = preprocessing_ops.get_image_shape(image)
    self.assertAllEqual((image_height, image_width),
                        image_shape)

  @parameterized.parameters((400, 600, .5, .5, .0, True), (100, 200, .5, .5, .5))
  def testImageRandHSV(self,
                       image_height,
                       image_width,
                       rh,
                       rs,
                       rv,
                       is_darknet = False):
    image = tf.convert_to_tensor(np.random.rand(image_height, image_width, 3))
    processed_image = preprocessing_ops.image_rand_hsv(image,
                                                       rh,
                                                       rs,
                                                       rv,
                                                       darknet= is_darknet)
    processed_image_shape = tf.shape(processed_image)
    self.assertAllEqual([image_height, image_width, 3],
                        processed_image_shape.numpy())

  @parameterized.parameters((100, 200, 50, 50), (100, 200, 50, 50, 0.5))
  def testRandomWindowCrop(self,
                           image_height,
                           image_width,
                           target_height,
                           target_width,
                           translate=0.0):
    image = tf.convert_to_tensor(np.random.rand(image_height, image_width, 3))
    processed_image, _ = preprocessing_ops.random_window_crop(image,
                                                              target_height,
                                                              target_width,
                                                              translate)
    processed_image_shape = tf.shape(processed_image)
    self.assertAllEqual([target_height, target_width, 3],
                        processed_image_shape.numpy())

  @parameterized.parameters((100, 200, [50, 100]))
  def testResizeAndJitterImage(self,
                               image_height,
                               image_width,
                               desired_size):
    image = tf.convert_to_tensor(np.random.rand(image_height, image_width, 3))
    processed_image, _, _ = preprocessing_ops.resize_and_jitter_image(image, desired_size)
    processed_image_shape = tf.shape(processed_image)
    self.assertAllEqual([desired_size[0], desired_size[1], 3],
                        processed_image_shape.numpy())

  @parameterized.parameters((400, 600, [200, 300]))
  def testAffineWarpImage(self,
                          image_height,
                          image_width,
                          desired_size,
                          degrees=7.0,
                          scale_min=0.1,
                          scale_max=1.9):
    image = tf.convert_to_tensor(np.random.rand(image_height, image_width, 3))
    processed_image, _, _ = preprocessing_ops.affine_warp_image(image,
                                                                desired_size,
                                                                degrees=degrees,
                                                                scale_min=scale_min,
                                                                scale_max=scale_max)
    processed_image_shape = tf.shape(processed_image)
    self.assertAllEqual([desired_size[0], desired_size[1], 3],
                        processed_image_shape.numpy())


if __name__ == '__main__':
  tf.test.main()

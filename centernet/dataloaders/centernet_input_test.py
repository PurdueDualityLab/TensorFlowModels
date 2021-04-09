import tensorflow as tf
from absl.testing import parameterized

from centernet.dataloaders.centernet_input import CenterNetParser


class CenterNetInputTest(tf.test.TestCase, parameterized.TestCase):
  def check_labels_correct(self, boxes, classes, output_size, input_size):
    parser = CenterNetParser()
    labels = parser._build_labels(
      boxes=tf.constant(boxes, dtype=tf.float32), 
      classes=tf.constant(classes, dtype=tf.float32), 
      output_size=output_size, input_size=input_size)
    
    tl_heatmaps = labels['tl_heatmaps']
    br_heatmaps = labels['br_heatmaps']
    ct_heatmaps = labels['ct_heatmaps']
    tl_offset = labels['tl_offset']
    br_offset = labels['br_offset']
    ct_offset = labels['ct_offset']
    size = labels['size']
    mask_indices = labels['mask_indices']
    box_indices = labels['box_indices']
    
    boxes = tf.cast(boxes, tf.float32)
    classes = tf.cast(classes, tf.float32)
    height_ratio = output_size[0] / input_size[0]
    width_ratio = output_size[1] / input_size[1]
    
    # Shape checks
    self.assertEqual(tl_heatmaps.shape, (512, 512, 90))
    self.assertEqual(br_heatmaps.shape, (512, 512, 90))
    self.assertEqual(ct_heatmaps.shape, (512, 512, 90))

    self.assertEqual(tl_offset.shape, (parser._max_num_instances, 2))
    self.assertEqual(br_offset.shape, (parser._max_num_instances, 2))
    self.assertEqual(ct_offset.shape, (parser._max_num_instances, 2))

    self.assertEqual(size.shape, (parser._max_num_instances, 2))
    self.assertEqual(mask_indices.shape, (parser._max_num_instances))
    self.assertEqual(box_indices.shape, (parser._max_num_instances, 3))
    
    # Not checking heatmaps, but we can visually validate them
    
    for i in range(len(boxes)):
      # Check sizes
      self.assertAllEqual(size[i], [boxes[i][3] - boxes[i][1], boxes[i][2] - boxes[i][0]])

      # Check box indices
      y = tf.math.floor((boxes[i][0] + boxes[i][2]) / 2 * height_ratio)
      x = tf.math.floor((boxes[i][1] + boxes[i][3]) / 2 * width_ratio)
      self.assertAllEqual(box_indices[i], [classes[i], y, x])

      # check offsets
      true_y = (boxes[i][0] + boxes[i][2]) / 2 * height_ratio
      true_x = (boxes[i][1] + boxes[i][3]) / 2 * width_ratio
      self.assertAllEqual(ct_offset[i], [true_x - x, true_y - y])
    
    for i in range(len(boxes), parser._max_num_instances):
      # Make sure rest are zero
      self.assertAllEqual(size[i], [0, 0])
      self.assertAllEqual(box_indices[i], [0, 0, 0])
      self.assertAllEqual(ct_offset[i], [0, 0])
    
    # Check mask indices
    self.assertAllEqual(tf.cast(mask_indices[3:], tf.int32), 
      tf.repeat(0, repeats=parser._max_num_instances-3))
    self.assertAllEqual(tf.cast(mask_indices[:3], tf.int32), 
      tf.repeat(1, repeats=3))


  def test_generate_heatmap_no_scale(self):
    boxes = [
      (10, 300, 15, 370),
      (100, 300, 150, 370),
      (15, 100, 200, 170),
    ]
    classes = (0, 1, 2)
    sizes = [512, 512]

    self.check_labels_correct(boxes=boxes, classes=classes, 
      output_size=sizes, input_size=sizes)

if __name__ == '__main__':
  tf.test.main()

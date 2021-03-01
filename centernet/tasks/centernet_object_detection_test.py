import tensorflow as tf
import numpy as np
import centernet.tasks as tasks
import centernet.utils as utils

def gaussian2D(shape, sigma=1):
  m, n = [(ss - 1.) / 2. for ss in shape]
  y, x = np.ogrid[-m:m+1,-n:n+1]

  h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
  h[h < np.finfo(h.dtype).eps * h.max()] = 0
  return h

def draw_gaussian(heatmap, center, radius, k=1, delte=6):
  diameter = 2 * radius + 1
  gaussian = gaussian2D((diameter, diameter), sigma=diameter / delte)

  x, y = center

  height, width = heatmap.shape[0:2]

  left, right = min(x, radius), min(width - x, radius + 1)
  top, bottom = min(y, radius), min(height - y, radius + 1)

  masked_heatmap  = heatmap[y - top:y + bottom, x - left:x + right]
  masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
  np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)

def gaussian_radius(det_size, min_overlap):
  height, width = det_size

  a1  = 1
  b1  = (height + width)
  c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
  sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
  r1  = (b1 + sq1) / 2

  a2  = 4
  b2  = 2 * (height + width)
  c2  = (1 - min_overlap) * width * height
  sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
  r2  = (b2 + sq2) / 2

  a3  = 4 * min_overlap
  b3  = -2 * min_overlap * (height + width)
  c3  = (min_overlap - 1) * width * height
  sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
  r3  = (b3 + sq3) / 2
  return min(r1, r2, r3)

def generate_heatmaps(batch_size, categories, output_size, detections, gaussian_iou=0.7):
  tl_heatmaps = np.zeros((batch_size, categories, output_size[0], output_size[1]), dtype=np.float32)
  br_heatmaps = np.zeros((batch_size, categories, output_size[0], output_size[1]), dtype=np.float32)
  ct_heatmaps = np.zeros((batch_size, categories, output_size[0], output_size[1]), dtype=np.float32)

  for b_ind, detection_batches in enumerate(detections):
    for ind, detection in enumerate(detection_batches):
      category = int(detection[-1])
      #category = 0

      xtl, ytl = detection[0], detection[1]
      xbr, ybr = detection[2], detection[3]
      xct, yct = (detection[2] + detection[0])/2., (detection[3]+detection[1])/2.

      xtl = int(xtl)
      ytl = int(ytl)
      xbr = int(xbr)
      ybr = int(ybr)
      xct = int(xct)
      yct = int(yct)

      width  = detection[2] - detection[0]
      height = detection[3] - detection[1]

      radius = gaussian_radius((height, width), gaussian_iou)
      radius = max(0, int(radius))

      draw_gaussian(tl_heatmaps[b_ind, category], [xtl, ytl], radius)
      draw_gaussian(br_heatmaps[b_ind, category], [xbr, ybr], radius)
      draw_gaussian(ct_heatmaps[b_ind, category], [xct, yct], radius, delte = 5)

  return tl_heatmaps, br_heatmaps, ct_heatmaps

class ObjectDetectionTest(tf.test.TestCase):
    def generate_heatmaps(self, dectections):
      detections = [[
        (10, 30, 15, 17, 0)
      ]]
      tl_heatmaps, br_heatmaps, ct_heatmaps = generate_heatmaps(1, 2, (416, 416), detections)
      pass

if __name__ == '__main__':
  # This code is for visualization
  import matplotlib.pyplot as plt
  detections = [[
    (10, 300, 15, 370, 0),
    (100, 300, 150, 370, 0),
    (200, 100, 15, 170, 0),
  ],
  # more images can go here if you like
  ]
  tl_heatmaps, br_heatmaps, ct_heatmaps = generate_heatmaps(1, 2, (416, 416), detections)
  # ct_heatmaps[batch_id, class_id, ...]
  plt.imshow(ct_heatmaps[0, 0, ...])
  plt.show()
  # This is to run the test
  # tf.test.main()

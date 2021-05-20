import tensorflow as tf
import numpy as np

from yolo.ops.box_ops import compute_iou
from yolo.ops.box_ops import yxyx_to_xcycwh
from official.core import input_reader

# https://github.com/AlexeyAB/darknet/blob/master/scripts/gen_anchors.py

[[15.0, 23.0], [38.0, 57.0], [119.0, 67.0], [57.0, 141.0], [164.0, 156.0],
 [97.0, 277.0], [371.0, 184.0], [211.0, 352.0], [428.0, 419.0]]


def IOU(X, centroids):
  w, h = tf.split(X, 2, axis=-1)
  c_w, c_h = tf.split(centroids, 2, axis=-1)

  similarity = (c_w * c_h) / (w * h)
  similarity = tf.where(
      tf.logical_and(c_w >= w, c_h >= h), w * h / (c_w * c_h), similarity)
  similarity = tf.where(
      tf.logical_and(c_w >= w, c_h <= h), w * c_h / (w * h + (c_w - w) * c_h),
      similarity)
  similarity = tf.where(
      tf.logical_and(c_w <= w, c_h >= h), c_w * h / (w * h + c_w * (c_h - h)),
      similarity)
  return tf.squeeze(similarity, axis=-1)


# def write_anchors_to_file(centroids,X,anchor_file):
#     f = open(anchor_file,'w')

#     anchors = centroids.copy()
#     print(anchors.shape)

#     for i in range(anchors.shape[0]):
#         anchors[i][0]*=width_in_cfg_file/32.
#         anchors[i][1]*=height_in_cfg_file/32.

#     widths = anchors[:,0]
#     sorted_indices = np.argsort(widths)

#     print('Anchors = ', anchors[sorted_indices])

#     for i in sorted_indices[:-1]:
#         f.write('%0.2f,%0.2f, '%(anchors[i,0],anchors[i,1]))

#     #there should not be comma after last anchor, that's why
#     f.write('%0.2f,%0.2f\n'%(anchors[sorted_indices[-1:],0],anchors[sorted_indices[-1:],1]))

#     f.write('%f\n'%(avg_IOU(X,centroids)))
#     print()


class AnchorKMeans:
  """K-means for YOLO anchor box priors
    Args:
      boxes(np.ndarray): a matrix containing image widths and heights
      k(int): number of clusters
      with_color(bool): color map
    To use:
      km = YoloKmeans(boxes = np.random.rand(20, 2), k = 3, with_color = True)
      centroids, map = km.run_kmeans()

      km = YoloKmeans()
      km.load_voc_boxes()
      centroids = km.run_kmeans()
      km = YoloKmeans()
      km.get_box_from_file("voc_boxes.pkl")
      centroids = km.run_kmeans()
      km = YoloKmeans()
      km.get_box_from_dataset(tfds.load('voc', split=['train', 'test', 'validation']))
      centroids = km.run_kmeans()
    """

  def __init__(self, boxes=None, k=9, with_color=False):
    assert isinstance(k, int)
    assert isinstance(with_color, bool)

    self._k = k
    self._boxes = boxes
    self._clusters = None
    self._with_color = with_color

  def iou(self, boxes, clusters):
    n = tf.shape(boxes)[0]
    boxes = tf.repeat(boxes, self._k, axis=0)
    boxes = tf.reshape(boxes, (n, self._k, -1))
    boxes = tf.cast(boxes, tf.float32)

    clusters = tf.tile(clusters, [n, 1])
    clusters = tf.reshape(clusters, (n, self._k, -1))
    clusters = tf.cast(clusters, tf.float32)

    # zeros = tf.cast(tf.zeros(boxes.shape), dtype=tf.float32)

    # boxes = tf.concat([zeros, boxes], axis=-1)
    # clusters = tf.concat([zeros, clusters], axis=-1)
    return IOU(boxes, clusters)

  def get_box_from_dataset(self, dataset, image_w=512):
    box_ls = None
    if not isinstance(dataset, list):
      dataset = [dataset]
    for ds in dataset:
      for el in ds:
        if type(box_ls) == type(None):
          box_ls = yxyx_to_xcycwh(el['groundtruth_boxes'])[..., 2:]
        else:
          box_ls = tf.concat(
              [box_ls, yxyx_to_xcycwh(el['groundtruth_boxes'])[..., 2:]],
              axis=0)
    self._boxes = box_ls

  @property
  def boxes(self):
    return self._boxes.numpy()

  def kmeans(self, max_iter, box_num, clusters, k):
    dists = tf.zeros((box_num, k))
    last = tf.zeros((box_num,), dtype=tf.int64) - 1
    old_d = dists

    tf.print(tf.shape(clusters))
    num_iters = 0

    while tf.math.less(num_iters, max_iter):
      dists = 1 - self.iou(self._boxes, clusters)
      curr = tf.math.argmin(dists, axis=-1)
      if tf.math.reduce_all(curr == last):
        break
      for i in range(k):
        hold = tf.math.reduce_mean(self._boxes[curr == i], axis=0)
        clusters = tf.tensor_scatter_nd_update(clusters, [[i]], [hold])
        tf.print(dists * 512, summarize=-1)

      last = curr
      num_iters += 1
      old_d = dists
      tf.print('k-Means box generation iteration: ', num_iters, end='\r')
    return clusters

  def run_kmeans(self, max_iter=300):
    box_num = tf.shape(self._boxes)[0]
    cluster_select = tf.convert_to_tensor(
        np.random.choice(box_num, self._k, replace=False))
    clusters = tf.gather(self._boxes, cluster_select, axis=0)
    clusters = self.kmeans(max_iter, box_num, clusters, self._k)

    clusters = clusters.numpy()
    clusters = np.array(sorted(clusters, key=lambda x: x[0]))
    return clusters, None

  def __call__(self, dataset, max_iter=300, image_width=416):
    if image_width is None:
      raise Warning('Using default width of 416 to generate bounding boxes')
      image_width = 416
    self.get_box_from_dataset(dataset)
    clusters, _ = self.run_kmeans(max_iter=max_iter)
    clusters = np.floor(clusters * image_width)
    return clusters.tolist()


class BoxGenInputReader(input_reader.InputReader):
  """Input reader that returns a tf.data.Dataset instance."""

  def read(self,
           k=None,
           image_width=416,
           input_context=None):  # -> tf.data.Dataset:

    self._is_training = False
    dataset = super().read(input_context=input_context)

    kmeans_gen = AnchorKMeans(k=k)
    boxes = kmeans_gen(dataset, image_width=image_width)
    del kmeans_gen  # free the memory
    del dataset

    print('clusting complete -> default boxes used ::')
    print(boxes)
    return boxes

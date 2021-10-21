from numpy.core.defchararray import center
import tensorflow as tf
import numpy as np

from yolo.ops.box_ops import compute_iou, compute_diou, compute_giou, compute_ciou
from yolo.ops.box_ops import yxyx_to_xcycwh
from official.core import input_reader
import matplotlib.patches as patches

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
# https://github.com/AlexeyAB/darknet/blob/master/scripts/gen_anchors.py

[[15.0, 23.0], [38.0, 57.0], [119.0, 67.0], [57.0, 141.0], [164.0, 156.0],
 [97.0, 277.0], [371.0, 184.0], [211.0, 352.0], [428.0, 419.0]]


def IOU(X, centroids_X):
  x = tf.concat([tf.zeros_like(X), X], axis = -1)
  centroids = tf.concat([tf.zeros_like(centroids_X), centroids_X], axis = -1)

  iou2, iou = compute_giou(x, centroids)

  # iou = (iou + 1)/2
  iou, _ = compute_giou(x, centroids)

  # x_area = tf.reduce_prod(X, axis = -1)
  # centroids_area = tf.reduce_prod(centroids_X, axis = -1)
  # mse = (x_area - centroids_area) ** 2

  # x_ar = X[..., 0]/X[..., 1]
  # centroids_ar = centroids_X[..., 0]/centroids_X[..., 1]
  # mse = (x_ar - centroids_ar) ** 2
  return iou #+ mse

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

  def __init__(self, boxes=None, with_color=False):
    assert isinstance(with_color, bool)

    self._boxes = boxes
    self._clusters = None
    self._with_color = with_color

  def iou(self, boxes, clusters):
    n = tf.shape(boxes)[0]
    k = tf.shape(clusters)[0]
    boxes = tf.repeat(boxes, k, axis=0)
    boxes = tf.reshape(boxes, (n, k, -1))
    boxes = tf.cast(boxes, tf.float32)

    clusters = tf.tile(clusters, [n, 1])
    clusters = tf.reshape(clusters, (n, k, -1))
    clusters = tf.cast(clusters, tf.float32)
    return IOU(boxes, clusters)

  def metric(self, wh, k):  # compute metrics
    x = self.iou(wh, tf.convert_to_tensor(k))  # iou metric
    return x, tf.reduce_max(x, axis=1)[0]  # x, best_x

  def fitness(self, wh, k, thr):  # mutation fitness
    _, best = self.metric(wh, k)
    return tf.reduce_mean((best * tf.cast(best > thr, tf.float32)))  # fitness

  
  def get_box_from_dataset(self, dataset, image_w=512):
    box_ls = []
    if not isinstance(dataset, list):
      dataset = [dataset]
    for ds in dataset:
      for i, el in enumerate(ds):
        width = el["width"]
        height = el["height"]
        boxes = el['groundtruth_boxes']

        scale = tf.maximum(width, height)
        boxes = yxyx_to_xcycwh(boxes)[..., 2:] * tf.cast([width, height], boxes.dtype)
        boxes = boxes[tf.reduce_max(boxes, axis = -1) >= 0] / tf.cast([width, height], boxes.dtype)
        box_ls.append(boxes)
        tf.print('loading sample: ', i, end='\r')

    box_ls = tf.concat(box_ls,axis=0)

    inds = tf.argsort(tf.reduce_prod(box_ls, axis = -1), axis = 0)

    box_ls = tf.gather(box_ls, inds, axis = 0)
    self._boxes = box_ls

  @property
  def boxes(self):
    return self._boxes.numpy()

  def dist(self, x, y, n):
    mw = min(x[0], y[0])
    mh = min(x[1], y[1])
    inter = mw * mh 

    wsum = x[0]*x[1] + y[0]*y[1]
    un = wsum - inter 
    iou = inter/(un + 0.00001)
    return 1 - iou

  def closes_center(self, box, cluster, i):
    best = 0
    best_dist = self.dist(box, cluster[0], i)
    for j in range(i):
      new_dist = self.dist(box, cluster[j], i)
      if new_dist < best_dist:
        best_dist = new_dist
        best = j
    return best

  def dist_to_closest_center(self, box, cluster, i):
    ci = self.closes_center(box, cluster, i)
    return self.dist(box, cluster[ci], i)

  def smart_centers(self, clusters):
    clusters = tf.zeros_like(clusters).numpy()
    boxes = self._boxes.numpy()

    weights = np.zeros(shape = [boxes.shape[0]], dtype = clusters.dtype)
    for i in range(clusters.shape[0]):
      wsum = 0
      for j in range(boxes.shape[0]):
        weights[j] = self.dist_to_closest_center(boxes[j], clusters, i)
        wsum += weights[j]
      
      r = wsum * np.random.uniform(0, 1, size = [])
      for j in range(boxes.shape[0]):
        r -= weights[j]
        if (r <= 0):
          clusters[i] = boxes[j]
          break
        wsum += weights[j]
    return clusters

  def maximization(self, boxes, clusters, assignments):
    for i in range(clusters.shape[0]):
      hold = tf.math.reduce_mean(boxes[assignments == i], axis=0)
      clusters = tf.tensor_scatter_nd_update(clusters, [[i]], [hold])
    return clusters

  def avg_iou(self, boxes, clusters, assignments):
    ious = []
    num_boxes = []
    clusters1 = tf.split(clusters, clusters.shape[0], axis = 0)
    for i, c in enumerate(clusters1):
      hold = boxes[assignments == i]
      iou = tf.reduce_mean(self.iou(hold, c)).numpy()
      ious.append(iou)
      num_boxes.append(hold.shape[0])
    
    print(self.floor_cluster(clusters).tolist())
    print(ious)
    print(num_boxes)
    return ious

  def get_init_centroids(self, boxes, k, type = "smart"):
    box_num = tf.shape(boxes)[0]

    def take_split(x, n):
      split = x.shape[0] // n
      bn2 = split * n 
      x = x[:bn2, :]
      return tf.split(x, n, axis = 0)


    # fixed_means
    if type == "split_means":
      split = box_num // k
      bn2 = split * k 
      boxes = boxes[:bn2, :]
      cluster_groups = tf.split(boxes, k, axis = 0)
      clusters = []
      for c in cluster_groups:
        clusters.append(tf.reduce_mean(c, axis = 0))
      print(clusters)
      clusters = tf.convert_to_tensor(clusters).numpy()
    elif type == "over_split_means":
      cluster_groups = take_split(boxes, k)
      clusters = []
      for i, c in enumerate(cluster_groups):
        if i - 1 > 0:
          c_minus = cluster_groups[i - 1]
          c_minus = take_split(c_minus, k + 1)[-1]
          c = tf.concat([c_minus, c], axis = 0)
        if i + 1 < len(cluster_groups):
          c_plus = cluster_groups[i + 1]
          c_plus = take_split(c_plus, k + 1)[0]
          c = tf.concat([c, c_plus], axis = 0)
        print(c.shape)
        clusters.append(tf.reduce_mean(c, axis = 0))
      print(clusters)
      clusters = tf.convert_to_tensor(clusters).numpy()
    else:
      cluster_select = tf.convert_to_tensor(np.random.choice(box_num, k, replace=False))
      clusters = tf.gather(boxes, cluster_select, axis=0)
      
      if type == "smart":
        clusters = self.smart_centers(clusters)

      if hasattr(clusters, "numpy"):
        clusters.numpy()
      clusters = np.array(sorted(clusters, key=lambda x: x[0] * x[1]))

    print(np.floor(clusters).tolist())
    return clusters

  def kmeans(self, boxes, clusters, k):

    boxes = boxes
    clusters = clusters
    assignments = tf.zeros((boxes.shape[0]), dtype=tf.int64) - 1
    dists = tf.zeros((boxes.shape[0], k))
    num_iters = 0

    dists = 1 - self.iou(boxes, clusters)
    curr = tf.math.argmin(dists, axis=-1)
    clusters = self.maximization(boxes, clusters, curr)
    while not tf.math.reduce_all(curr == assignments):
      # get the distiance
      assignments = curr
      dists = 1 - self.iou(boxes, clusters)
      curr = tf.math.argmin(dists, axis=-1)
      clusters = self.maximization(boxes, clusters, curr)
      tf.print('k-Means box generation iteration: ', num_iters, end='\r')
      num_iters += 1

    tf.print('k-Means box generation iteration: ', num_iters, end='\n')
    assignments = curr
    
    clusters = tf.convert_to_tensor(np.array(sorted(clusters.numpy(), key=lambda x: x[0] * x[1])))
    dists = 1 - self.iou(boxes, clusters)
    assignments = tf.math.argmin(dists, axis=-1)
    return clusters, assignments

  def floor_cluster(self, clusters):
    return np.floor(np.array(sorted(clusters, key=lambda x: x[0] * x[1])))

  def run_kmeans(self, k, boxes, clusters = None, type = "smart"):
    if clusters is None:
      clusters = self.get_init_centroids(boxes, k, type = type)
    clusters, assignments = self.kmeans(boxes, clusters, k)
    return clusters.numpy(), assignments.numpy()
  
  def get_boxes(self, boxes_, clusters, assignments = None):
    if assignments is None:
      dists = 1 - self.iou(boxes_, np.array(clusters))
      assignments = tf.math.argmin(dists, axis=-1)
    boxes = []
    clusters = tf.split(clusters, clusters.shape[0], axis = 0)
    for i, c in enumerate(clusters):
      hold = boxes_[assignments == i]
      if hasattr(hold, "numpy"):
        hold = hold.numpy()
      boxes.append(hold)    
    return boxes

  def plot_boxes(self, boxes, centroids, assignments, image_resolution):
    
    boxes = self.get_boxes(boxes, centroids, assignments)
    if hasattr(centroids, "numpy"):
      centroids = centroids.numpy()

    color = list(iter(cm.rainbow(np.linspace(0, 1, len(boxes)))))

    fig, ax = plt.subplots(2)
    for i, box_set in enumerate(boxes):
      ax[0].scatter(box_set[:, 0], box_set[:, 1], label = i, color = color[i])
    ax[0].scatter(centroids[:, 0], centroids[:, 1], s = 80, color = 'k')

    image = tf.ones(image_resolution, dtype = tf.float32).numpy()
    ax[1].imshow(image)
    for i in range(centroids.shape[0]):
      rect = patches.Rectangle((image_resolution[0]//2 - centroids[i, 0]//2, image_resolution[1]//2 - centroids[i, 1]//2), centroids[i, 0], centroids[i, 1], linewidth=1, edgecolor=color[i], facecolor='none')
      ax[1].add_patch(rect)
    plt.show()

    return

  def avg_iou_total(self, boxes, clusters):
    clusters = tf.convert_to_tensor(clusters)
    dists = 1 - self.iou(boxes, clusters)
    assignments = tf.math.argmin(dists, axis=-1)
    ious = self.avg_iou(boxes, clusters, assignments)
    print()
    print()
    print()
    return clusters, assignments, ious
  
  def evolve(self, boxes, clusters, k = 3):
      boxes_set1 = self.get_boxes(self._boxes, clusters)
      clusters = []
      for boxes in boxes_set1:
        cluster_set, assignments = self.run_kmeans(2, boxes, type = "random")
        cluster_set, assignments, ious = self.avg_iou_total(boxes, cluster_set)

        ind = np.argmax(ious)
        # clusters.extend(cluster_set)
        # clusters.append(cluster_set[ind]) #
        clusters.append(np.mean(cluster_set, axis = 0))
      cluster_set = self.floor_cluster(cluster_set)
      clusters, assignments, ious = self.avg_iou_total(self._boxes, clusters)
      print()
      print()
      print()

      return clusters, assignments

  def __call__(self, dataset, k, anchors = None, anchors_per_scale = None, image_resolution=512):
    self.get_box_from_dataset(dataset)
    self._boxes *= tf.convert_to_tensor(image_resolution[:2], self._boxes.dtype)



    if anchors_per_scale is None:
      clusters, assignments = self.run_kmeans(k, self._boxes)
      clusters, assignments, iou = self.avg_iou_total(self._boxes, clusters)
      print(np.mean(iou))
      self.plot_boxes(self._boxes, clusters, assignments, image_resolution)
    else:
      boxes_ls = self._boxes.numpy()

      clustersp, assignments = self.run_kmeans(anchors_per_scale, boxes_ls, type = "split_means")
      clustersp += np.roll(clustersp, 1, axis = -1)
      clustersp /= 2

      clusters1 = self.get_init_centroids(boxes_ls, anchors_per_scale, type = "split_means")
      clusters1 += np.roll(clusters1, 1, axis = -1)
      clusters1 /= 2
      
      clusters1 = (clustersp + clusters1*4)/5

      boxes_set1 = self.get_boxes(boxes_ls, clusters1)
      clusters = []
      for boxes in boxes_set1:
        cluster_set, assignments = self.run_kmeans(k//anchors_per_scale, boxes, type = "split_means")
        #cluster_set, assignments, ious = self.avg_iou_total(boxes, cluster_set)
        clusters.extend(cluster_set)
      clusters, assignments, iou = self.avg_iou_total(boxes_ls, clusters)
      print(np.mean(iou))
      self.plot_boxes(boxes_ls, clusters, assignments, image_resolution)

        

    clusters, assignments, iou = self.avg_iou_total(self._boxes, anchors)
    print(np.mean(iou))
    self.plot_boxes(self._boxes, clusters, assignments, image_resolution)
    clusters = np.floor(clusters)
    return clusters.tolist()


class BoxGenInputReader(input_reader.InputReader):
  """Input reader that returns a tf.data.Dataset instance."""

  def read(self,
           k=None,
           anchors_per_scale = None,
           anchors=None,
           image_resolution=416,
           input_context=None):  # -> tf.data.Dataset:

    self._is_training = False
    dataset = super().read(input_context=input_context)
    dataset = dataset.unbatch()

    kmeans_gen = AnchorKMeans()
    boxes = kmeans_gen(
      dataset, k, 
      anchors_per_scale = anchors_per_scale, 
      anchors = anchors,  image_resolution=image_resolution)
    del kmeans_gen  # free the memory
    del dataset

    print('clusting complete -> default boxes used ::')
    print(boxes)
    return boxes
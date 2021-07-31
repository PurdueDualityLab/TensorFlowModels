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


# def kmean_anchors(path='./data/coco128.yaml', n=9, img_size=640, thr=4.0, gen=1000, verbose=True):
#     """ Creates kmeans-evolved anchors from training dataset

#         Arguments:
#             path: path to dataset *.yaml, or a loaded dataset
#             n: number of anchors
#             img_size: image size used for training
#             thr: anchor-label wh ratio threshold hyperparameter hyp['anchor_t'] used for training, default=4.0
#             gen: generations to evolve anchors using genetic algorithm

#         Return:
#             k: kmeans evolved anchors

#         Usage:
#             from utils.utils import *; _ = kmean_anchors()
#     """
#     thr = 1. / thr

#     def metric(k, wh):  # compute metrics
#         r = wh[:, None] / k[None]
#         x = torch.min(r, 1. / r).min(2)[0]  # ratio metric
#         # x = wh_iou(wh, torch.tensor(k))  # iou metric
#         return x, x.max(1)[0]  # x, best_x

#     def fitness(k):  # mutation fitness
#         _, best = metric(torch.tensor(k, dtype=torch.float32), wh)
#         return (best * (best > thr).float()).mean()  # fitness

#     def print_results(k):
#         k = k[np.argsort(k.prod(1))]  # sort small to large
#         x, best = metric(k, wh0)
#         bpr, aat = (best > thr).float().mean(), (x > thr).float().mean() * n  # best possible recall, anch > thr
#         print('thr=%.2f: %.4f best possible recall, %.2f anchors past thr' % (thr, bpr, aat))
#         print('n=%g, img_size=%s, metric_all=%.3f/%.3f-mean/best, past_thr=%.3f-mean: ' %
#               (n, img_size, x.mean(), best.mean(), x[x > thr].mean()), end='')
#         for i, x in enumerate(k):
#             print('%i,%i' % (round(x[0]), round(x[1])), end=',  ' if i < len(k) - 1 else '\n')  # use in *.cfg
#         return k

#     if isinstance(path, str):  # *.yaml file
#         with open(path) as f:
#             data_dict = yaml.load(f, Loader=yaml.FullLoader)  # model dict
#         from utils.datasets import LoadImagesAndLabels
#         dataset = LoadImagesAndLabels(data_dict['train'], augment=True, rect=True)
#     else:
#         dataset = path  # dataset

#     # Get label wh
#     shapes = img_size * dataset.shapes / dataset.shapes.max(1, keepdims=True)
#     wh0 = np.concatenate([l[:, 3:5] * s for s, l in zip(shapes, dataset.labels)])  # wh

#     # Filter
#     i = (wh0 < 3.0).any(1).sum()
#     if i:
#         print('WARNING: Extremely small objects found. '
#               '%g of %g labels are < 3 pixels in width or height.' % (i, len(wh0)))
#     wh = wh0[(wh0 >= 2.0).any(1)]  # filter > 2 pixels

#     # Kmeans calculation
#     print('Running kmeans for %g anchors on %g points...' % (n, len(wh)))
#     s = wh.std(0)  # sigmas for whitening
#     k, dist = kmeans(wh / s, n, iter=30)  # points, mean distance
#     k *= s
#     wh = torch.tensor(wh, dtype=torch.float32)  # filtered
#     wh0 = torch.tensor(wh0, dtype=torch.float32)  # unflitered
#     k = print_results(k)

#     # Plot
#     # k, d = [None] * 20, [None] * 20
#     # for i in tqdm(range(1, 21)):
#     #     k[i-1], d[i-1] = kmeans(wh / s, i)  # points, mean distance
#     # fig, ax = plt.subplots(1, 2, figsize=(14, 7))
#     # ax = ax.ravel()
#     # ax[0].plot(np.arange(1, 21), np.array(d) ** 2, marker='.')
#     # fig, ax = plt.subplots(1, 2, figsize=(14, 7))  # plot wh
#     # ax[0].hist(wh[wh[:, 0]<100, 0],400)
#     # ax[1].hist(wh[wh[:, 1]<100, 1],400)
#     # fig.tight_layout()
#     # fig.savefig('wh.png', dpi=200)

#     # Evolve
#     npr = np.random
#     f, sh, mp, s = fitness(k), k.shape, 0.9, 0.1  # fitness, generations, mutation prob, sigma
#     pbar = tqdm(range(gen), desc='Evolving anchors with Genetic Algorithm')  # progress bar
#     for _ in pbar:
#         v = np.ones(sh)
#         while (v == 1).all():  # mutate until a change occurs (prevent duplicates)
#             v = ((npr.random(sh) < mp) * npr.random() * npr.randn(*sh) * s + 1).clip(0.3, 3.0)
#         kg = (k.copy() * v).clip(min=2.0)
#         fg = fitness(kg)
#         if fg > f:
#             f, k = fg, kg.copy()
#             pbar.desc = 'Evolving anchors with Genetic Algorithm: fitness = %.4f' % f
#             if verbose:
#                 print_results(k)

#     return print_results(k)


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

  def metric(self, wh, k):  # compute metrics
    # n = tf.shape(wh)[0]
    # wh = tf.repeat(wh, self._k, axis=0)
    # wh = tf.reshape(wh, (n, self._k, -1))
    # wh = tf.cast(wh, tf.float32)

    # k = tf.tile(k, [n, 1])
    # k = tf.reshape(k, (n, self._k, -1))
    # k = tf.cast(k, tf.float32)
    # r = wh / k

    # tf.print(r)
    # x = tf.reduce_min(tf.minimum(r, 1. / r), axis = 2) # ratio metric
    # tf.print(x)
    # # x = x[0]

    x = self.iou(wh, tf.convert_to_tensor(k))  # iou metric
    return x, tf.reduce_max(x, axis=1)[0]  # x, best_x

  def fitness(self, wh, k, thr):  # mutation fitness
    _, best = self.metric(wh, k)
    return (best * tf.cast(best > thr, tf.float32))  # fitness

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
          # tf.print(yxyx_to_xcycwh(el['groundtruth_boxes'])[..., 2:])
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
        # tf.print(dists * 512, summarize=-1)

      last = curr
      num_iters += 1
      old_d = dists
      tf.print('k-Means box generation iteration: ', num_iters, end='\r')

    f = self.fitness(self._boxes, clusters, 0.213)
    sh = tf.shape(clusters)
    mp = 0.9
    s = 0.1
    c = clusters

    npr = np.random
    number_of_generations = 1000
    for k in range(number_of_generations):
      v = tf.ones_like(clusters)
      while tf.reduce_all(v == 1):
        v = ((npr.random(sh) < mp) * npr.random() * npr.randn(*sh) * s +
             1).clip(0.3, 3.0)
        v = tf.convert_to_tensor(v)
      kg = clusters * tf.cast(v, clusters.dtype)
      fg = self.fitness(self._boxes, kg, 0.213)
      tf.print(fg)
      if fg > f:
        f = fg
        clusters = kg
    # tf.print(c * 512, clusters * 512)
    return c, clusters

  def run_kmeans(self, max_iter=300):
    box_num = tf.shape(self._boxes)[0]
    cluster_select = tf.convert_to_tensor(
        np.random.choice(box_num, self._k, replace=False))
    clusters = tf.gather(self._boxes, cluster_select, axis=0)
    c, clusters = self.kmeans(max_iter, box_num, clusters, self._k)

    # fitness(clusters, self._boxes, )
    clusters = clusters.numpy()
    c = c.numpy()
    clusters = np.array(sorted(clusters, key=lambda x: x[0] * x[1]))
    c = np.array(sorted(c, key=lambda x: x[0] * x[1]))
    return c, clusters, None

  def __call__(self, dataset, max_iter=300, image_width=416):
    if image_width is None:
      raise Warning('Using default width of 416 to generate bounding boxes')
      image_width = 416
    self.get_box_from_dataset(dataset)
    self._boxes *= image_width
    c, clusters, _ = self.run_kmeans(max_iter=max_iter)
    clusters = np.floor(clusters)
    c = np.floor(c)
    return clusters.tolist(), c.tolist()


class BoxGenInputReader(input_reader.InputReader):
  """Input reader that returns a tf.data.Dataset instance."""

  def read(self,
           k=None,
           image_width=416,
           input_context=None):  # -> tf.data.Dataset:

    self._is_training = False
    dataset = super().read(input_context=input_context)

    kmeans_gen = AnchorKMeans(k=k)
    boxes, ogb = kmeans_gen(dataset, image_width=image_width)
    del kmeans_gen  # free the memory
    del dataset

    print('clusting complete -> default boxes used ::')
    print(ogb)
    return boxes

import numpy as np
 
class _YoloKmeans:
  def __init__(self, boxes = None, k = 9, with_color = False):
        self._k = k
        self._boxes = boxes
        self._with_color = with_color
  def iou(self, boxes, clusters):
    n = boxes.shape[0]
    k = self._k
 
    box_area = boxes[:, 0] * boxes[:, 1]
    box_area = box_area.repeat(k)
    box_area = np.reshape(box_area, (n, k))
 
    cluster_area = clusters[:, 0] * clusters[:, 1]
    cluster_area = np.tile(cluster_area, [1, n])
    cluster_area = np.reshape(cluster_area, (n, k))
 
    box_w = np.reshape(boxes[:, 0].repeat(k), (n, k))
    cluster_w = np.reshape(np.tile(clusters[:, 0], [1, n]), (n, k))
    min_w = np.minimum(box_w, cluster_w)
 
    box_h = np.reshape(boxes[:, 1].repeat(k), (n, k))
    cluster_h = np.reshape(np.tile(clusters[:, 1], [1, n]), (n, k))
    min_h = np.minimum(box_h, cluster_h)
 
    intersection = np.multiply(min_w, min_h) # element-wise
 
    return intersection / (box_area + cluster_area - intersection)
  
  def _run_kmeans(self, max_iter = 300):
    if not isinstance(self._boxes, np.ndarray):
        raise Exception('Box Not found')
    box_num = self._boxes.shape[0]
    k = self._k
    dists = np.zeros((box_num, k))
    last = np.zeros((box_num,))
    np.random.seed()
    clusters = self._boxes[np.random.choice(box_num, k, replace=False)]
    num_iters = 0
 
    while num_iters < max_iter:
      dists = 1 - self.iou(self._boxes, clusters)
      curr = np.argmin(dists, axis = -1) 
      if (curr == last).all():
        break
      for i in range(k):
        clusters[i] = np.median(self._boxes[curr == i], axis=0)
      last = curr
      num_iters += 1
    print(f'num_iters = {num_iters}')
    if self._with_color:
        return clusters, last
    else:
        return clusters

if __name__ == "__main__":
    import time
    import matplotlib.pyplot as plt

    boxes = 416 * np.random.rand(2000, 2)
    print(boxes.shape)

    km = _YoloKmeans(boxes, with_color = True)
    centroids, last = km._run_kmeans()

    plt.scatter(boxes[:, 0], boxes[:, 1], c = last)
    plt.scatter(centroids[:,0], centroids[:,1], c = 'black')
    plt.show()

    km = _YoloKmeans()
    km._run_kmeans()
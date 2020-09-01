import numpy as np
 
class YoloKmeans:
    """K-means for YOLO anchor box priors
    Args:
        boxes(np.ndarray): a matrix containing image widths and heights
        k(int): number of clusters
        with_color(bool): color map
    To use:
        km = YoloKmeans(boxes = np.random.rand(20, 2), k = 3, with_color = True)
        centroids, map = km.run_kmeans()
    """
    def __init__(self, boxes, k = 9, with_color = False):

        assert isinstance(boxes, np.ndarray)
        assert isinstance(k, int)
        assert isinstance(with_color, bool)
        assert boxes.shape[-1] == 2

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

    def run_kmeans(self, max_iter = 300):
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
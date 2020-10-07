import numpy as np
import pickle
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from yolo.utils.iou_utils import compute_iou
from yolo.utils.box_utils import _yxyx_to_xcycwh
import tensorflow as tf


class YoloKmeans:
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
        self._with_color = with_color

    def iou(self, boxes, clusters):
        n = boxes.shape[0]
        boxes = tf.repeat(boxes, self._k, axis=0)
        boxes = tf.reshape(boxes, (n, self._k, -1))
        boxes = tf.cast(boxes, tf.float32)

        clusters = tf.tile(clusters, [n, 1])
        clusters = tf.reshape(clusters, (n, self._k, -1))
        clusters = tf.cast(clusters, tf.float32)

        zeros = tf.cast(tf.zeros(boxes.shape), dtype=tf.float32)

        boxes = tf.concat([zeros, boxes], axis=-1)
        clusters = tf.concat([zeros, clusters], axis=-1)
        return compute_iou(boxes, clusters)

    def get_box_from_file(self, filename):
        try:
            f = open(filename, 'rb')
        except IOError:
            pass
        self._boxes = pickle.load(f)

    def get_box_from_dataset(self, dataset):
        box_ls = []
        if not isinstance(dataset, list):
            dataset = [dataset]
        for ds in dataset:
            for el in ds:
                for box in list(el['objects']['bbox']):
                    box_ls.append(_yxyx_to_xcycwh(box).numpy()[..., 2:])
        self._boxes = np.array(box_ls)

    def load_voc_boxes(self):
        self.get_box_from_dataset(
            tfds.load('voc', split=['train', 'test', 'validation']))

    def load_coco_boxes(self):
        self.get_box_from_dataset(
            tfds.load('coco',
                      split=['test', 'test2015', 'train', 'validation']))

    def get_boxes(self):
        return self._boxes

    def run_kmeans(self, max_iter=300):
        if not isinstance(self._boxes, np.ndarray):
            raise Exception('Box Not found')

        box_num = self._boxes.shape[0]
        k = self._k
        dists = np.zeros((box_num, k))
        last = np.zeros((box_num, ))
        np.random.seed()
        clusters = self._boxes[np.random.choice(box_num, k, replace=False)]
        num_iters = 0

        while num_iters < max_iter:
            dists = 1 - self.iou(self._boxes, clusters)
            curr = np.argmin(dists, axis=-1)
            if (curr == last).all():
                break
            for i in range(k):
                clusters[i] = np.mean(self._boxes[curr == i], axis=0)
            last = curr
            num_iters += 1
        print(f'num_iters = {num_iters}')
        clusters = np.array(sorted(clusters, key=lambda x: x[0] * x[1]))
        if self._with_color:
            return clusters, last
        else:
            return clusters


if __name__ == '__main__':
    km = YoloKmeans(with_color=True)
    km.load_coco_boxes()
    centroids, cmap = km.run_kmeans()
    boxes = km.get_boxes()
    plt.scatter(boxes[:, 0], boxes[:, 1], c=cmap)
    plt.scatter(centroids[:, 0], centroids[:, 1], c='b')
    plt.show()
    print((centroids * 416).astype(int))

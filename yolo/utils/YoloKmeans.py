import numpy as np
import pickle
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from yolo.utils.iou_utils import compute_iou
from yolo.utils.box_utils import yxyx_to_xcycwh
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
        self._clusters = None
        self._with_color = with_color

    @tf.function
    def iou(self, boxes, clusters):
        n = tf.shape(boxes)[0]
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

    # slow as fuck
    # load via threads
    def get_box_from_dataset(self, dataset):
        box_ls = None
        if not isinstance(dataset, list):
            dataset = [dataset]
        for ds in dataset:
            for el in ds:  
                if type(box_ls) == type(None):
                    #box_ls = yxyx_to_xcycwh(el['objects']['bbox'])[..., 2:]
                    box_ls = yxyx_to_xcycwh(el["groundtruth_boxes"])[..., 2:]
                else:
                    box_ls = tf.concat([box_ls, yxyx_to_xcycwh(el["groundtruth_boxes"])[..., 2:]], axis = 0)
        self._boxes = box_ls

    def load_voc_boxes(self):
        self.get_box_from_dataset(
            tfds.load('voc', split=['train', 'test', 'validation']))

    def load_coco_boxes(self):
        self.get_box_from_dataset(
            tfds.load('coco',
                      split=['test', 'test2015', 'train', 'validation']))

    @property
    def boxes(self):
        return self._boxes.numpy()

    @tf.function
    def kmeans(self, max_iter, box_num, clusters, k):
        dists = tf.zeros((box_num, k))
        last = tf.zeros((box_num, ), dtype=tf.int64)
        
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
            last = curr
            num_iters += 1
            tf.print('k-Means box generation iteration: ', num_iters , end = "\r")
        return clusters

    def run_kmeans(self, max_iter=300):
        box_num = tf.shape(self._boxes)[0]
        cluster_select = tf.convert_to_tensor(np.random.choice(box_num, self._k, replace=False))
        clusters = tf.gather(self._boxes, cluster_select, axis = 0)
        clusters = self.kmeans(max_iter, box_num, clusters, self._k)
        
        clusters = clusters.numpy()
        clusters = np.array(sorted(clusters, key=lambda x: x[0] * x[1]))
        if self._with_color:
            return clusters, last
        else:
            return clusters, None
    
    def __call__(self, dataset, max_iter = 300, image_width = 416):
        if image_width == None:
            raise Warning("Using default width of 416 to generate bounding boxes")
            image_width = 416
        self.get_box_from_dataset(dataset)
        clusters, _  = self.run_kmeans(max_iter=max_iter)
        clusters = np.floor(clusters * image_width)
        return clusters.tolist()


    



if __name__ == '__main__':
    import tensorflow_datasets as tfds

    coco = tfds.load("coco", split = "validation", shuffle_files = True)
    coco = coco.take(40000)
    coco = coco.shuffle(10000).prefetch(10000)

    km2 = YoloKmeans(k = 9)
    print(km2(coco, image_width=608))

    # km = MiniBatchKMeansNN(k = 9, box_width = 608)
    # print(km(coco))




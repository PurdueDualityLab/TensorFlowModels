import numpy as np
import pickle
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
 
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
    def __init__(self, boxes = None, k = 9, with_color = False):

        assert isinstance(k, int)
        assert isinstance(with_color, bool)
        if boxes:
            assert isinstance(boxes, np.ndarray)
            assert boxes.shape[-1] == 2

        self._k = k
        self._boxes = boxes if boxes else None
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
                    box_ls.append(box.numpy()[2:])
        self._boxes = np.array(box_ls)

    def load_voc_boxes(self):
        self.get_box_from_dataset(tfds.load('voc', split=['train', 'test', 'validation']))

    def load_coco_boxes(self):
        self.get_box_from_dataset(tfds.load('coco', split=['test', 'test2015', 'train', 'validation']))

    def get_boxes(self):
        return self._boxes

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

if __name__ == '__main__':
    km = YoloKmeans(with_color= True)
    km.load_voc_boxes()
    centroids, cmap = km.run_kmeans()
    boxes = km.get_boxes()
    plt.scatter(boxes[:, 0], boxes[:, 1], c = cmap)
    plt.scatter(centroids[:, 0], centroids[:, 1], c = 'b')
    plt.show()
    print((centroids * 416).astype(int))
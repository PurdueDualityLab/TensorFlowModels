from yolo.dataloaders.YoloKmeans import YoloKmeans
import tensorflow_datasets as tfds
import numpy as np


def build_anchor_boxes(dataset, num_boxes=9, width=416., height=416.):
    if not isinstance(num_boxes, int):
        raise ValueError('num_boxes should be an Integer')
    km = YoloKmeans(k=num_boxes)
    km.get_box_from_dataset(dataset)
    centroids = km.run_kmeans()
    centroids = np.multiply(centroids,
                            np.tile(np.array([width, height]), [num_boxes, 1]))
    return centroids.astype(int)


if __name__ == '__main__':
    print(build_anchor_boxes(tfds.load('coco', split=['validation'])))

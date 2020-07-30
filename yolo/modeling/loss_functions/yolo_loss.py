import tensorflow as tf
import tensorflow.keras as ks
from tensorflow.keras import backend as K
import tensorflow_datasets as tfds
from yolo.modeling.yolo_v3 import Yolov3
from yolo.modeling.loss_functions.voc_test import *

class Yolo_Loss(ks.losses.Loss):
    def __init__(self, mask, anchors, classes, num, ignore_thresh, truth_thresh, random, fixed_dims, batch_size = None, reduction = tf.keras.losses.Reduction.AUTO, name=None, **kwargs):
        self._anchors = [anchors[i] for i in mask]
        self._classes = classes
        self._num = len(mask)
        self._ignore_thresh = ignore_thresh
        self._truth_thresh = truth_thresh
        self._iou_thresh = 1 # recomended use = 0.213 in [yolo]
        self._random = 0 if random == None else random
        self._net_fixed_dims = fixed_dims
        self._batch_size = batch_size
        super(Yolo_Loss, self).__init__(reduction = reduction, name = name)
        return

    @tf.function
    def _get_splits(self, tensor):
        splits = list(tf.split(tensor, [(self._classes + 5)] * self._num, axis = -1))
        for i in range(len(splits)):
            splits[i] = list(tf.split(splits[i], [self._classes, 1, 4], axis = -1))
        return splits

    @tf.function
    def _get_yolo_box(self, box, width, height, grid_points, anchor_box):
        box_split = tf.split(box, [1,1,1,1], axis = -1)
        x = K.sigmoid(box_split[0]) + grid_points[0]
        y = K.sigmoid(box_split[1]) + grid_points[1]
        w = K.exp(box_split[2]) * anchor_box[0]/self._net_fixed_dims
        h = K.exp(box_split[3]) * anchor_box[1]/self._net_fixed_dims
        return (x, y, w, h)

    @tf.function
    def _split_truth_box(self, box):
        return tuple(tf.split(box, [1,1,1,1], axis = -1))

    def _get_centers(self, lwidth, lheight, batch_size):
        x_left = K.expand_dims(tf.linspace(start = 0.0, stop = (lwidth - 1)/lwidth, num = lwidth))
        y_left = K.expand_dims(tf.linspace(start = 0.0, stop = (lheight - 1)/lheight, num = lheight))
        x_left = K.concatenate([x_left] * lheight)
        y_left = K.transpose(K.concatenate([y_left] * lwidth))
        x_left = K.expand_dims(x_left)
        y_left = K.expand_dims(y_left)
        x_left = K.stack([x_left]*batch_size, axis = 0)
        y_left = K.stack([y_left]*batch_size, axis = 0)
        return K.constant(x_left, dtype=tf.float32), K.constant(y_left, dtype=tf.float32)
    
    #errors
    def _generate_truth(self, y_true, batch_size, width, height):
        gt = y_true.to_list()
        box_list = [[[[] for j in range(height)] for i in range(width)] for _ in range(len(gt))]

        # gt = y_true.to_list()
        for i in range(len(gt)):
            for j in range(len(gt[i])):
                x = tf.cast(y_true[i, j, -4] * width, dtype=tf.int32)
                y = tf.cast(y_true[i, j, -3] * height, dtype=tf.int32)
                box_list[i][x][y].extend(gt[i][j])
        concat = tf.ragged.constant(box_list).to_tensor()
        return concat


    def call(self, y_pred, y_true):
        width = y_pred.shape[1]
        height = y_pred.shape[2]
        batch_size = y_pred.shape[0] if self._batch_size == None else self._batch_size 
        print(batch_size, width, height)
        #y_true = self._generate_truth(y_true, batch_size, width, height)
        #print(len(y_true))
        #grid_points = self._get_centers(width, height, batch_size)
        #print(y_true.shape)
        # temp_out = y_true - y_pred
        # y_true = self._get_splits(y_true)
        #y_pred = self._get_splits(y_pred)
        
        # for i, (truth, pred) in enumerate(zip(y_true, y_pred)):
        #     box_pred = self._get_yolo_box(pred[2], width, height, grid_points, self._anchors[i])
        #     box_truth = self._split_truth_box(truth[2])
        #     truth[1] = iou(box_pred, box_truth)
        # print(len(y_true), len(y_pred))
        return 10#K.sum(y_true - y_pred)

    def get_config(self):
        pass


def load_dataset(skip = 0, batch_size = 100):
    dataset,info = tfds.load('voc', split='train', with_info=True, shuffle_files=False)
    dataset = dataset.skip(skip).take(batch_size)
    return dataset.map(preprocess).batch(batch_size)

def load_model():
    model = Yolov3(dn2tf_backbone = True, 
                   dn2tf_head = False,
                   input_shape= (None, 416, 416, 3), 
                   config_file="yolov3.cfg", 
                   weights_file='yolov3_416.weights', 
                   classes = 20, 
                   boxes = 9)
    model.build(input_shape = (1, 416, 416, 3))
    model.summary()
    
    return model

def load_loss(batch_size, n = 3, classes = 20):
    depth = n * (classes + 5)
    anchors = [(10,13),  (16,30),  (33,23),  (30,61),  (62,45),  (59,119),  (116,90),  (156,198),  (373,326)]
    outtop = Yolo_Loss(mask = [6, 7, 8], 
                anchors = anchors, 
                classes = classes, 
                num = 3, 
                ignore_thresh = 0.7,
                truth_thresh = 1, 
                random = 1, 
                fixed_dims = 416, 
                batch_size = batch_size)
    outmid = Yolo_Loss(mask = [3, 4, 5], 
                anchors = anchors, 
                classes = classes, 
                num = 3, 
                ignore_thresh = 0.7,
                truth_thresh = 1, 
                random = 1, 
                fixed_dims = 416, 
                batch_size = batch_size)
    outbot = Yolo_Loss(mask = [1, 2, 3], 
                anchors = anchors, 
                classes = classes, 
                num = 3, 
                ignore_thresh = 0.7,
                truth_thresh = 1, 
                random = 1, 
                fixed_dims = 416, 
                batch_size = batch_size)
    loss_dict = {256: outtop, 512: outmid, 1024: outbot}
    return loss_dict

def main():
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    batch = 20
    dataset = load_dataset(batch_size=batch)
    model = load_model()
    loss_fns = load_loss(batch)

    print(dataset)
    print(loss_fns)
    import time
    with tf.device("/GPU:0"):
    #     with tf.GradientTape() as tape:
    #         for image, label in dataset:
    #             y_pred = model(image)      
    #             total_loss = 0 
    # with tf.device("/CPU:0"):
    #     start = time.time()
    #     for key in y_pred.keys():
    #         print(f"{key}:{y_pred[key].shape}")
    #         loss = loss_fns[key](y_pred[key], label)
    #         total_loss += loss
    #     end = time.time() - start
        model.compile(loss=loss_fns)
        model.fit(dataset)
    # print(end)  
        
    return
    

if __name__ == "__main__":
    
    main()

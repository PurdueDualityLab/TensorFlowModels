import tensorflow as tf
import tensorflow.keras as ks
from tensorflow.keras import backend as K
import tensorflow_datasets as tfds
from yolo.modeling.yolo_v3 import Yolov3
from yolo.modeling.loss_functions.voc_test import *

class Yolo_Loss(ks.losses.Loss):
    def __init__(self, mask, anchors, classes, num, ignore_thresh, truth_thresh, random, fixed_dims, batch_size = None, reduction = tf.keras.losses.Reduction.AUTO, name=None, **kwargs):
        self._anchors = K.expand_dims(tf.convert_to_tensor([anchors[i] for i in mask], dtype= tf.float32), axis = 0)/416
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
    def _get_splits(self, tensor, batch_size, width, height):
        # split by anchor
        # splits = list(tf.split(tensor, self._num, axis = -1))
        splits = tf.reshape(tensor, [batch_size, width, height, self._num, (self._classes + 5)])
        # [(self._classes + 5)] * 
        # for i in range(len(splits)):
        #     #split by attribute
        #     splits[i] = list(tf.split(splits[i], [4, 1, self._classes], axis = -1))
        return splits

    @tf.function
    def _get_centers(self, lwidth, lheight, batch_size):
        x_left = tf.linspace(start = 0.0, stop = K.cast((lwidth - 1)/lwidth, dtype = tf.float32), num = lwidth)
        y_left = tf.linspace(start = 0.0, stop = K.cast((lheight - 1)/lheight, dtype = tf.float32), num = lheight)

        x_left, y_left = tf.meshgrid(x_left, y_left)
        x_left = tf.repeat(K.expand_dims(tf.repeat(K.expand_dims(x_left, axis=0), batch_size, axis = 0), axis = -1), self._num, axis = -1)
        y_left = tf.repeat(K.expand_dims(tf.repeat(K.expand_dims(y_left, axis=0), batch_size, axis = 0), axis = -1), self._num, axis = -1)

        #tf.print(tf.shape(x_left), tf.shape(y_left))
        return K.stack([x_left, y_left], axis = -1)
    
    @tf.function
    def _get_anchor_grid(self, width, height, batch_size):
        anchors = tf.repeat(self._anchors, width*height, axis = 0)
        anchors = K.expand_dims(tf.reshape(anchors, [width, height, self._num, -1]), axis = 0)
        anchors = tf.repeat(anchors, batch_size, axis = 0)
        return anchors

    def call(self, y_pred, y_true):
        width = tf.shape(y_pred)[1]
        height = tf.shape(y_pred)[2]
        batch_size = tf.shape(y_pred)[0]

        mse = ks.losses.MSE

        grid_points = self._get_centers(width, height, batch_size)  
        anchor_grid = self._get_anchor_grid(width, height, batch_size)
        
        tf.print(tf.shape(y_true), tf.shape(y_pred))
        tf.print(tf.shape(self._anchors))
        
        y_pred = self._get_splits(y_pred, batch_size, width, height)
        y_true = self._get_splits(y_true, batch_size, width, height)

        pred_xy = tf.math.sigmoid(y_pred[..., 0:2]) + grid_points
        pred_wh = tf.math.exp(y_pred[..., 2:4]) * anchor_grid
        pred_conf = tf.math.sigmoid(y_pred[..., 4])
        pred_class = tf.math.sigmoid(y_pred[..., 5:])

        # tf.print(anchor_grid[1,..., 0, 1])
        # tf.print(pred_wh[1,..., 0, 1])
        
        true_xy = y_true[..., 0:2]
        true_wh = y_true[..., 2:4]
        true_conf = y_true[..., 4]
        true_class = y_true[..., 5:]

        temp_out = K.mean(mse(pred_xy,true_xy)) + K.mean(mse(pred_wh,true_wh)) + K.mean(mse(pred_conf, true_conf)) + K.mean(mse(pred_class, true_class))

        tf.print(tf.shape(pred_xy), tf.shape(pred_conf), tf.shape(pred_class))

        return temp_out

    def get_config(self):
        pass

def load_model():
    model = Yolov3(dn2tf_backbone = True, 
                   dn2tf_head = False,
                   input_shape= (None, 416, 416, 3), 
                   config_file="yolov3.cfg", 
                   weights_file='yolov3_416.weights', 
                   classes = 20, 
                   boxes = 9)
    # model.build(input_shape = (1, 416, 416, 3))
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
    outbot = Yolo_Loss(mask = [0, 1, 2], 
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

def print_out(tensor):
    for key in tensor.keys():
        print(f"{key}, {tf.shape(tensor[key])}")
    return

def main():
    # physical_devices = tf.config.list_physical_devices('GPU')
    # tf.config.experimental.set_memory_growth(physical_devices[0], True)
    #strategy = tf.distribute.MirroredStrategy()
    batch = 2
    dataset = load_dataset(batch_size=batch)
    model = load_model()
    loss_fns = load_loss(batch)

    print(dataset)
    # print(loss_fns)
    import time
    #with strategy.scope():
    with tf.device("/GPU:0"):
        for image, label in dataset:
            with tf.GradientTape() as tape:
                y_pred = model(image)
                print_out(y_pred)
                print_out(label)      
                total_loss = 0 
                start = time.time()
                for key in y_pred.keys():
                    print(f"{key}:{y_pred[key].shape}")
                    loss = loss_fns[key](y_pred[key], label[key])
                    tf.print(loss)
                    total_loss += loss
                end = time.time() - start
                grad = tape.gradient(loss, model.trainable_variables)
        # model.compile(loss=loss_fns)
        # model.fit(dataset)
    # print(end)  
        
    return
    

if __name__ == "__main__":
    main()

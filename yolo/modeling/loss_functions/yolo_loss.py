import tensorflow as tf
import tensorflow.keras as ks
from tensorflow.keras import backend as K
import tensorflow_datasets as tfds
from yolo.modeling.yolo_v3 import Yolov3
from yolo.modeling.loss_functions.voc_test import *

class Yolo_Loss(ks.losses.Loss):
    def __init__(self, mask, anchors, classes, num, ignore_thresh, truth_thresh, random, fixed_dims, reduction = tf.keras.losses.Reduction.AUTO, name=None, **kwargs):
        self._anchors = K.expand_dims(tf.convert_to_tensor([anchors[i] for i in mask], dtype= tf.float32), axis = 0)/416
        self._classes = tf.cast(classes, dtype = tf.int32)
        self._num = tf.cast(len(mask), dtype = tf.int32)
        self._ignore_thresh = tf.cast(ignore_thresh, dtype = tf.float32)
        # self._truth_thresh = truth_thresh
        self._iou_thresh = 1 # recomended use = 0.213 in [yolo]
        self._random = 0 if random == None else random
        self._net_fixed_dims = fixed_dims
        super(Yolo_Loss, self).__init__(reduction = reduction, name = name)

        self._loss_type = "mse"
        self._iou_normalizer= tf.cast(0.5, dtype = tf.float32)
        return

    @tf.function
    def _get_centers(self, lwidth, lheight, batch_size):
        x_left = tf.linspace(start = 0.0, stop = K.cast((lwidth - 1)/lwidth, dtype = tf.float32), num = lwidth)
        y_left = tf.linspace(start = 0.0, stop = K.cast((lheight - 1)/lheight, dtype = tf.float32), num = lheight)

        x_left, y_left = tf.meshgrid(x_left, y_left)
        x_y = tf.transpose(K.stack([x_left, y_left], axis = -1), perm = [1, 0, 2])
        x_y = tf.repeat(K.expand_dims(tf.repeat(K.expand_dims(x_y, axis = -2), self._num, axis = -2), axis = 0), batch_size, axis = 0)
        return x_y
    
    @tf.function
    def _get_anchor_grid(self, width, height, batch_size):
        anchors = tf.repeat(self._anchors, width*height, axis = 0)
        anchors = K.expand_dims(tf.reshape(anchors, [width, height, self._num, -1]), axis = 0)
        anchors = tf.repeat(anchors, batch_size, axis = 0)
        return anchors

    def call(self, y_true, y_pred):
        #1. generate and store constants and format output
        width = tf.cast(tf.shape(y_pred)[1], dtype = tf.int32)
        height = tf.cast(tf.shape(y_pred)[2], dtype = tf.int32)
        batch_size = tf.cast(tf.shape(y_pred)[0], dtype = tf.int32)
        grid_points = self._get_centers(width, height, batch_size)  
        anchor_grid = self._get_anchor_grid(width, height, batch_size)
        
        y_pred = tf.reshape(y_pred, [batch_size, width, height, self._num, (self._classes + 5)])
        #y_true = tf.reshape(y_true, [batch_size, width, height, self._num, (self._classes + 5)])

        fwidth = tf.cast(width, dtype = tf.float32)
        fheight = tf.cast(height, dtype = tf.float32)

        #2. split up layer output into components, xy, wh, confidence, class -> then apply activations to the correct items
        pred_xy = tf.math.sigmoid(y_pred[..., 0:2])
        pred_wh = y_pred[..., 2:4]
        pred_conf = tf.math.sigmoid(y_pred[..., 4])
        pred_class = tf.math.sigmoid(y_pred[..., 5:])

        #3. split up ground_truth into components, xy, wh, confidence, class -> apply calculations to acchive safe format as predictions
        true_xy = (y_true[..., 0:2] - grid_points)
        true_wh = tf.math.log(y_true[..., 2:4]/anchor_grid)
        true_wh = tf.where(tf.math.is_nan(true_wh), 0.0, true_wh)
        true_wh = tf.where(tf.math.is_inf(true_wh), 0.0, true_wh)
        true_conf = y_true[..., 4]
        true_class = y_true[..., 5:]
        true_xy = K.concatenate([K.expand_dims(true_xy[..., 0] * true_conf * fwidth, axis = -1), K.expand_dims(true_xy[..., 1] * true_conf * fheight, axis = -1)], axis = -1)

        #4. use iou to calculate the mask of where the network belived there to be an object -> used to penelaize the network for wrong predictions
        temp = K.concatenate([K.expand_dims(pred_xy[..., 0]/fwidth, axis = -1), K.expand_dims(pred_xy[..., 1]/fheight, axis = -1)], axis = -1) + grid_points
        pred_box = K.concatenate([temp, tf.math.exp(pred_wh) * anchor_grid], axis = -1)        
        true_box = y_true[..., 0:4]
        mask_iou = box_iou(true_box, pred_box) 
        mask_iou = tf.cast(mask_iou < self._ignore_thresh, dtype = tf.float32)

        #5. apply generalized IOU or mse to the box predictions -> only the indexes where an object exists will affect the total loss -> found via the true_confidnce in ground truth 
        if self._loss_type == "mse":
            #yolo_layer.c: scale = (2-truth.w*truth.h)
            scale = (2 - true_box[...,2] * true_box[...,3])
            loss_xy = tf.reduce_sum(K.square(true_xy - pred_xy), axis = -1)
            loss_wh = tf.reduce_sum(K.square(true_wh - pred_wh), axis = -1)
            loss_box = (loss_wh + loss_xy) * true_conf * scale
        else:
            loss_box = (1 - giou(true_box, pred_box)) * self._iou_normalizer * true_conf

        #6. apply binary cross entropy(bce) to class attributes -> only the indexes where an object exists will affect the total loss -> found via the true_confidnce in ground truth 
        class_loss = tf.reduce_sum(ks.losses.binary_crossentropy(K.expand_dims(true_class, axis = -1), K.expand_dims(pred_class, axis = -1)), axis= -1) * true_conf
        
        #7. apply bce to confidence at all points and then strategiacally penalize the network for making predictions of objects at locations were no object exists
        conf_loss = ks.losses.binary_crossentropy(K.expand_dims(true_conf, axis = -1), K.expand_dims(pred_conf, axis = -1))
        conf_loss = true_conf * (conf_loss) + (1 - true_conf) * conf_loss * mask_iou
        
        #8. take the sum of all the dimentions and reduce the loss such that each batch has a unique loss value
        loss_box = tf.reduce_sum(loss_box, axis=(1, 2, 3))
        conf_loss = tf.reduce_sum(conf_loss*conf_loss, axis=(1, 2, 3))
        class_loss = tf.reduce_sum(class_loss, axis=(1, 2, 3))

        #9. i beleive tensorflow will take the average of all the batches loss, so add them and let TF do its thing
        return (class_loss + conf_loss + loss_box)

    def get_config(self):
        pass

def load_model():
    model = Yolov3(dn2tf_backbone = True, 
                   dn2tf_head = True,
                   input_shape= (None, None, None, 3), 
                   config_file="yolov3.cfg", 
                   weights_file='yolov3_416.weights', 
                   classes = 80, 
                   boxes = 9)
    model.build(input_shape = (None, None, None, 3))
    model.summary()
    return model

def load_loss(batch_size, n = 3, classes = 80):
    depth = n * (classes + 5)
    anchors = [(10,13),  (16,30),  (33,23),  (30,61),  (62,45),  (59,119),  (116,90),  (156,198),  (373,326)]
    outtop = Yolo_Loss(mask = [6, 7, 8], 
                anchors = anchors, 
                classes = classes, 
                num = 3, 
                ignore_thresh = 0.7,
                truth_thresh = 1, 
                random = 1, 
                fixed_dims = 416)
    outmid = Yolo_Loss(mask = [3, 4, 5], 
                anchors = anchors, 
                classes = classes, 
                num = 3, 
                ignore_thresh = 0.7,
                truth_thresh = 1, 
                random = 1, 
                fixed_dims = 416)
    outbot = Yolo_Loss(mask = [0, 1, 2], 
                anchors = anchors, 
                classes = classes, 
                num = 3, 
                ignore_thresh = 0.7,
                truth_thresh = 1, 
                random = 1, 
                fixed_dims = 416)
    loss_dict = {256: outtop, 512: outmid, 1024: outbot}
    return loss_dict

def print_out(tensor):
    for key in tensor.keys():
        print(f"{key}, {tf.shape(tensor[key])}")
    return

def main():
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    #strategy = tf.distribute.MirroredStrategy()
    batch = 10
    dataset = load_dataset(skip = 0, batch_size=batch)
    optimizer = ks.optimizers.SGD(learning_rate=0.001)
    model = load_model()
    loss_fns = load_loss(batch)

    print(dataset)
    # print(loss_fns)
    import time
    with tf.device("/GPU:0"):
        model.compile(loss=loss_fns, optimizer=optimizer)
        model.fit(dataset)
    return
    

if __name__ == "__main__":
    main()

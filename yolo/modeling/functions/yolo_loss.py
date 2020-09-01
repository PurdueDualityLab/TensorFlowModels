import tensorflow as tf
import tensorflow.keras as ks
from tensorflow.keras import backend as K
import tensorflow_datasets as tfds
from yolo.modeling.yolo_v3 import Yolov3
from yolo.modeling.functions.iou import *


class Yolo_Loss(ks.losses.Loss):
    def __init__(self, 
                 mask, 
                 anchors, 
                 classes,
                 num_extras = 0, 
                 ignore_thresh = 0.7, 
                 truth_thresh = 1, 
                 loss_type = "mse", 
                 iou_normalizer = 1.0,
                 cls_normalizer = 1.0, 
                 scale_x_y = 1.0,
                 nms_kind = "greedynms",
                 beta_nms = 0.6,
                 reduction = tf.keras.losses.Reduction.AUTO, 
                 name=None, 
                 dtype = tf.float16, # change to tf.float32
                 **kwargs):

        """
        parameters for the loss functions used at each detection head output

        Args: 
            mask: list of indexes for which anchors in the anchors list should be used in prediction
            anchors: list of tuples (w, h) representing the anchor boxes to be used in prediction 
            classes: the number of classes that can be predicted 
            num_extras: number of indexes predicted in addition to 4 for the box and N + 1 for classes 
            ignore_thresh: float for the threshold for if iou > threshold the network has made a prediction, 
                           and should not be penealized for p(object) prediction if an object exists at this location
            truth_thresh: float thresholding the groud truth to get the true mask 
            loss_type: string for the key of the loss to use, 
                       options -> mse, giou, ciou
            iou_normalizer: float used for appropriatly scaling the iou or the loss used for the box prediction error 
            cls_normalizer: float used for appropriatly scaling the classification error
            scale_x_y: float used to scale the predictied x and y outputs
            nms_kind: string used for filtering the output and ensuring each object ahs only one prediction
            beta_nms: float for the thresholding value to apply in non max supression(nms) -> not yet implemented

        call Return: 
            float: for the average loss 
        """
        self._dtype = dtype
        self._anchors = K.expand_dims(tf.convert_to_tensor([anchors[i] for i in mask], dtype= self._dtype), axis = 0)/416 #<- division done for testing
        self._classes = tf.cast(classes, dtype = tf.int32)
        self._num = tf.cast(len(mask), dtype = tf.int32)
        self._num_extras = tf.cast(num_extras, dtype = self._dtype)
        self._truth_thresh = tf.cast(truth_thresh, dtype = self._dtype) 
        self._ignore_thresh = tf.cast(ignore_thresh, dtype = self._dtype)

        # used (mask_n >= 0 && n != best_n && l.iou_thresh < 1.0f) for id n != nest_n
        # checks all anchors to see if another anchor was used on this ground truth box to make a prediction
        # if iou > self._iou_thresh then the network check the other anchors, so basically 
        # checking anchor box 1 on prediction for anchor box 2
        self._iou_thresh = tf.cast(0.213, dtype = self._dtype) # recomended use = 0.213 in [yolo]
        
        self._loss_type = loss_type
        self._iou_normalizer= tf.cast(iou_normalizer, dtype = self._dtype)
        self._cls_normalizer = tf.cast(cls_normalizer, dtype = self._dtype)
        self._scale_x_y = tf.cast(scale_x_y, dtype = self._dtype)

        #used in detection filtering
        self._beta_nms = tf.cast(beta_nms, dtype = self._dtype)
        self._nms_kind = nms_kind
        
        super(Yolo_Loss, self).__init__(reduction = reduction, name = name)
        return

    @tf.function
    def _get_centers(self, lwidth, lheight, batch_size):
        """ generate a grid that is used to detemine the relative centers of the bounding boxs """
        #x_left = tf.linspace(start = 0.0, stop = K.cast((lwidth - 1)/lwidth, dtype = tf.float32), num = lwidth)
        #y_left = tf.linspace(start = 0.0, stop = K.cast((lheight - 1)/lheight, dtype = tf.float32), num = lheight)

        x_left = tf.linspace(start = K.cast(tf.cast(1, dtype = tf.int32)/lwidth, dtype = tf.float32), stop = K.cast((lwidth)/lwidth, dtype = tf.float32), num = lheight)
        y_left = tf.linspace(start = K.cast(tf.cast(1, dtype = tf.int32)/lheight, dtype = tf.float32), stop = K.cast((lheight)/lheight, dtype = tf.float32), num = lwidth)
        x_left, y_left = tf.meshgrid(x_left, y_left)

        x_y = K.stack([x_left, y_left], axis = -1)
        x_y = tf.cast(x_y, dtype = self._dtype)

        x_y = tf.repeat(K.expand_dims(tf.repeat(K.expand_dims(x_y, axis = -2), self._num, axis = -2), axis = 0), batch_size, axis = 0)
        return x_y
    
    @tf.function
    def _get_anchor_grid(self, width, height, batch_size):
        """ get the transformed anchor boxes for each dimention """
        # need to make sure this is correct
        # anchors = tf.cast(self._anchors, dtype = self._dtype)
        # anchors = tf.reshape(anchors, [1, -1])
        # anchors = tf.repeat(anchors, width*height, axis = 0)
        # anchors = tf.reshape(anchors, [1, width, height, self._num, -1])
        # anchors = tf.repeat(anchors, batch_size, axis = 0)

        anchors = tf.repeat(self._anchors, width*height, axis = 0)
        anchors = K.expand_dims(tf.reshape(anchors, [width, height, self._num, -1]), axis = 0)
        anchors = tf.repeat(anchors, batch_size, axis = 0)
        # anchors = tf.cast(anchors, dtype = self._dtype)
        return anchors

    def call(self, y_true, y_pred):
        #1. generate and store constants and format output

        # y_true = tf.cast(y_true, dtype = tf.float32)
        # y_pred = tf.cast(y_pred, dtype = tf.float32)
        # tf.print(tf.shape(y_true), tf.shape(y_pred))

        width = tf.cast(tf.shape(y_pred)[1], dtype = tf.int32)
        height = tf.cast(tf.shape(y_pred)[2], dtype = tf.int32)
        batch_size = tf.cast(tf.shape(y_pred)[0], dtype = tf.int32)

        grid_points = self._get_centers(width, height, batch_size)  
        anchor_grid = self._get_anchor_grid(width, height, batch_size)
        
        y_pred = tf.reshape(y_pred, [batch_size, width, height, self._num, (self._classes + 5)])

        fwidth = tf.cast(width, self._dtype)
        fheight = tf.cast(height, self._dtype)

        #2. split up layer output into components, xy, wh, confidence, class -> then apply activations to the correct items
        pred_xy = tf.math.sigmoid(y_pred[..., 0:2]) * self._scale_x_y - 0.5 * (self._scale_x_y - 1)
        pred_wh = y_pred[..., 2:4]
        pred_conf = tf.math.sigmoid(y_pred[..., 4])
        pred_class = tf.math.sigmoid(y_pred[..., 5:])

        #3. split up ground_truth into components, xy, wh, confidence, class -> apply calculations to acchive safe format as predictions
        true_xy = tf.nn.relu(y_true[..., 0:2] - grid_points)
        #true_xy = y_true[..., 0:2] - grid_points
        true_xy = K.concatenate([K.expand_dims(true_xy[..., 0] * fwidth, axis = -1), K.expand_dims(true_xy[..., 1] * fheight, axis = -1)], axis = -1)
        true_wh = tf.math.log(y_true[..., 2:4]/anchor_grid)
        true_wh = tf.where(tf.math.is_nan(true_wh), tf.cast(0.0, dtype = self._dtype), true_wh)
        true_wh = tf.where(tf.math.is_inf(true_wh), tf.cast(0.0, dtype = self._dtype), true_wh)
        true_conf = y_true[..., 4]
        true_class = y_true[..., 5:]
        #true_xy = K.concatenate([K.expand_dims(true_xy[..., 0] * true_conf * fwidth, axis = -1), K.expand_dims(true_xy[..., 1] * true_conf * fheight, axis = -1)], axis = -1)

        #4. use iou to calculate the mask of where the network belived there to be an object -> used to penelaize the network for wrong predictions
        temp = K.concatenate([K.expand_dims(pred_xy[..., 0]/fwidth, axis = -1), K.expand_dims(pred_xy[..., 1]/fheight, axis = -1)], axis = -1) + grid_points
        pred_box = K.concatenate([temp, tf.math.exp(pred_wh) * anchor_grid], axis = -1)        
        true_box = y_true[..., 0:4]
        mask_iou = box_iou(true_box, pred_box, dtype = self._dtype) 
        mask_iou = tf.cast(mask_iou < self._ignore_thresh, dtype = self._dtype)

        #5. apply generalized IOU or mse to the box predictions -> only the indexes where an object exists will affect the total loss -> found via the true_confidnce in ground truth 
        if self._loss_type == "mse":
            #yolo_layer.c: scale = (2-truth.w*truth.h)
            scale = (2 - true_box[...,2] * true_box[...,3])
            loss_xy = tf.reduce_sum(K.square(true_xy - pred_xy), axis = -1)
            loss_wh = tf.reduce_sum(K.square(true_wh - pred_wh), axis = -1)
            loss_box = (loss_wh + loss_xy) * true_conf * scale * self._iou_normalizer 
        else:
            loss_box = (1 - giou(true_box, pred_box, dtype = self._dtype)) * self._iou_normalizer * true_conf

        #6. apply binary cross entropy(bce) to class attributes -> only the indexes where an object exists will affect the total loss -> found via the true_confidnce in ground truth 
        class_loss = self._cls_normalizer * tf.reduce_sum(ks.losses.binary_crossentropy(K.expand_dims(true_class, axis = -1), K.expand_dims(pred_class, axis = -1)), axis= -1) * true_conf
        
        #7. apply bce to confidence at all points and then strategiacally penalize the network for making predictions of objects at locations were no object exists
        conf_loss = ks.losses.binary_crossentropy(K.expand_dims(true_conf, axis = -1), K.expand_dims(pred_conf, axis = -1))
        conf_loss = true_conf * (conf_loss) + (1 - true_conf) * conf_loss * mask_iou
        
        #8. take the sum of all the dimentions and reduce the loss such that each batch has a unique loss value
        loss_box = tf.cast(tf.reduce_sum(loss_box, axis=(1, 2, 3)), dtype = self._dtype)
        conf_loss = tf.cast(tf.reduce_sum(conf_loss*conf_loss, axis=(1, 2, 3)), dtype = self._dtype)
        class_loss = tf.cast(tf.reduce_sum(class_loss, axis=(1, 2, 3)), dtype = self._dtype)

        #9. i beleive tensorflow will take the average of all the batches loss, so add them and let TF do its thing
        return (class_loss + conf_loss + loss_box)

    def get_config(self):
        """save all loss attributes"""
        layer_config = {
                "anchors": self._anchors, 
                "classes": self._classes,
                "ignore_thresh": self._ignore_thresh, 
                "truth_thresh": self._truth_thresh, 
                "iou_thresh": self._iou_thresh, 
                "loss_type": self._loss_type, 
                "iou_normalizer": self._iou_normalizer,
                "cls_normalizer": self._cls_normalizer, 
                "scale_x_y": self._scale_x_y, 
        }
        layer_config.update(super().get_config())
        return layer_config

        pass

# def load_model():
#     model = Yolov3(classes = 20, boxes = 9)
#     model.build(input_shape = (None, None, None, 3))
#     model.load_weights_from_dn(dn2tf_backbone = True, dn2tf_head = True, config_file=None, weights_file="yolov3-regular.weights")
#     model.summary()
#     return model

# def load_loss(batch_size, n = 3, classes = 20):
#     depth = n * (classes + 5)
#     anchors = [(10,13),  (16,30),  (33,23),  (30,61),  (62,45),  (59,119),  (116,90),  (156,198),  (373,326)]
#     outtop = Yolo_Loss(mask = [6, 7, 8], 
#                 anchors = anchors, 
#                 classes = classes, 
#                 ignore_thresh = 0.7,
#                 truth_thresh = 1, 
#                 random = 1)
#     outmid = Yolo_Loss(mask = [3, 4, 5], 
#                 anchors = anchors, 
#                 classes = classes, 
#                 ignore_thresh = 0.7,
#                 truth_thresh = 1, 
#                 random = 1)
#     outbot = Yolo_Loss(mask = [0, 1, 2], 
#                 anchors = anchors, 
#                 classes = classes, 
#                 ignore_thresh = 0.7,
#                 truth_thresh = 1, 
#                 random = 1)
#     loss_dict = {256: outtop, 512: outmid, 1024: outbot}
#     return loss_dict

# def print_out(tensor):
#     for key in tensor.keys():
#         print(f"{key}, {tf.shape(tensor[key])}")
#     return

# def main():
#     batch = 10
#     dataset = load_dataset(skip = 0, batch_size=batch)
#     optimizer = ks.optimizers.SGD(learning_rate=0.001)
#     model = load_model()
#     loss_fns = load_loss(batch)

#     print(dataset)
#     import time
#     with tf.device("/GPU:0"):
#         model.compile(loss=loss_fns, optimizer=optimizer)
#         model.fit(dataset)
#     return
    

if __name__ == "__main__":
    main()

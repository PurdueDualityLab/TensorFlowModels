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
                 scale_anchors = 1, 
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
                 dtype = tf.float32,
                 **kwargs):

        """
        parameters for the loss functions used at each detection head output

        Args: 
            mask: list of indexes for which anchors in the anchors list should be used in prediction
            anchors: list of tuples (w, h) representing the anchor boxes to be used in prediction 
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
        super(Yolo_Loss, self).__init__(reduction = reduction, name = name, **kwargs)
        self.dtype = dtype
        self._anchors = tf.convert_to_tensor([anchors[i] for i in mask], dtype= self.dtype)/scale_anchors #<- division done for testing

        self._num = tf.cast(len(mask), dtype = tf.int32)
        self._num_extras = tf.cast(num_extras, dtype = self.dtype)
        self._truth_thresh = tf.cast(truth_thresh, dtype = self.dtype) 
        self._ignore_thresh = tf.cast(ignore_thresh, dtype = self.dtype)

        # used (mask_n >= 0 && n != best_n && l.iou_thresh < 1.0f) for id n != nest_n
        # checks all anchors to see if another anchor was used on this ground truth box to make a prediction
        # if iou > self._iou_thresh then the network check the other anchors, so basically 
        # checking anchor box 1 on prediction for anchor box 2
        self._iou_thresh = tf.cast(0.213, dtype = self.dtype) # recomended use = 0.213 in [yolo]
        
        self._loss_type = loss_type
        self._iou_normalizer= tf.cast(iou_normalizer, dtype = self.dtype)
        self._cls_normalizer = tf.cast(cls_normalizer, dtype = self.dtype)
        self._scale_x_y = tf.cast(scale_x_y, dtype = self.dtype)

        #used in detection filtering
        self._beta_nms = tf.cast(beta_nms, dtype = self.dtype)
        self._nms_kind = nms_kind
        return

    @tf.function
    def _get_centers(self, lwidth, lheight, batch_size):
        """ generate a grid that is used to detemine the relative centers of the bounding boxs """
        x_left, y_left = tf.meshgrid(tf.range(0, lheight), tf.range(0, lwidth))
        x_y = K.stack([x_left, y_left], axis = -1)
        x_y = tf.cast(x_y, dtype = self.dtype)/tf.cast(lwidth, dtype = self.dtype)
        x_y = tf.repeat(tf.expand_dims(tf.repeat(tf.expand_dims(x_y, axis = -2), self._num, axis = -2), axis = 0), batch_size, axis = 0)
        return x_y
    
    @tf.function
    def _get_anchor_grid(self, width, height, batch_size):
        """ get the transformed anchor boxes for each dimention """
        anchors = tf.cast(self._anchors, dtype = self.dtype)
        anchors = tf.reshape(anchors, [1, -1])
        anchors = tf.repeat(anchors, width*height, axis = 0)
        anchors = tf.reshape(anchors, [1, width, height, self._num, -1])
        anchors = tf.repeat(anchors, batch_size, axis = 0)
        return anchors

    @tf.function
    def print_error(self, pred_conf):
        if tf.reduce_any(tf.math.is_nan(pred_conf)) == tf.convert_to_tensor([True]):
            tf.print("\nerror")

    def call(self, y_true, y_pred):
        #1. generate and store constants and format output
        batch_size = tf.cast(tf.shape(y_pred)[0], dtype = tf.int32)
        width = tf.cast(tf.shape(y_pred)[1], dtype = tf.int32)
        height = tf.cast(tf.shape(y_pred)[2], dtype = tf.int32)
        grid_points = self._get_centers(width, height, batch_size)
        anchor_grid = self._get_anchor_grid(width, height, batch_size)

        y_pred = tf.reshape(y_pred, [batch_size, width, height, self._num, -1])
        y_pred = tf.cast(y_pred, dtype = self.dtype)

        fwidth = tf.cast(width, self.dtype)
        fheight = tf.cast(height, self.dtype)

        #2. split up layer output into components, xy, wh, confidence, class -> then apply activations to the correct items
        pred_xy = tf.math.sigmoid(y_pred[..., 0:2]) * self._scale_x_y - 0.5 * (self._scale_x_y - 1)
        pred_wh = y_pred[..., 2:4]
        pred_conf = tf.expand_dims(tf.math.sigmoid(y_pred[..., 4]), axis = -1)
        pred_class = tf.math.sigmoid(y_pred[..., 5:])
        self.print_error(pred_conf)

        #3. split up ground_truth into components, xy, wh, confidence, class -> apply calculations to acchive safe format as predictions
        true_xy = tf.nn.relu(y_true[..., 0:2] - grid_points)
        true_xy = K.concatenate([K.expand_dims(true_xy[..., 0] * fwidth, axis = -1), K.expand_dims(true_xy[..., 1] * fheight, axis = -1)], axis = -1)
        true_wh = tf.math.log(y_true[..., 2:4]/anchor_grid)
        true_wh = tf.where(tf.math.is_nan(true_wh), tf.cast(0.0, dtype = self.dtype), true_wh)
        true_wh = tf.where(tf.math.is_inf(true_wh), tf.cast(0.0, dtype = self.dtype), true_wh)
        true_conf = y_true[..., 4]
        true_class = y_true[..., 5:]

        #4. use iou to calculate the mask of where the network belived there to be an object -> used to penelaize the network for wrong predictions
        box_xy = pred_xy[..., 0:2]/fwidth + grid_points
        box_wh = tf.math.exp(pred_wh) * anchor_grid
        pred_box = K.concatenate([box_xy, box_wh], axis = -1)        
        true_box = y_true[..., 0:4]
        iou = box_iou(true_box, pred_box, dtype = self.dtype) 
        iou = tf.where(tf.math.is_nan(iou), tf.cast(0.0, dtype = self.dtype), iou)
        iou = tf.where(tf.math.is_inf(iou), tf.cast(0.0, dtype = self.dtype), iou)
        mask_iou = tf.cast(iou < self._ignore_thresh, dtype = self.dtype)

        #5. apply generalized IOU or mse to the box predictions -> only the indexes where an object exists will affect the total loss -> found via the true_confidnce in ground truth 
        if self._loss_type == "mse":
            #yolo_layer.c: scale = (2-truth.w*truth.h)
            #error exists here
            scale = (2 - true_box[...,2] * true_box[...,3]) * self._iou_normalizer 
            loss_xy = tf.reduce_sum(K.square(true_xy - pred_xy), axis = -1)
            loss_wh = tf.reduce_sum(K.square(true_wh - pred_wh), axis = -1)
            loss_box = (loss_wh + loss_xy) * true_conf * scale 
        else:
            giou_loss = giou(true_box, pred_box, dtype = self.dtype)
            giou_loss = tf.where(tf.math.is_nan(giou_loss), tf.cast(0.0, dtype = self.dtype), giou_loss)
            giou_loss = tf.where(tf.math.is_inf(giou_loss), tf.cast(0.0, dtype = self.dtype), giou_loss)
            loss_box = (1 - giou_loss) * self._iou_normalizer * true_conf

        #6. apply binary cross entropy(bce) to class attributes -> only the indexes where an object exists will affect the total loss -> found via the true_confidnce in ground truth 
        class_loss = self._cls_normalizer * tf.reduce_sum(ks.losses.binary_crossentropy(K.expand_dims(true_class, axis = -1), K.expand_dims(pred_class, axis = -1)), axis= -1) * true_conf
        
        #7. apply bce to confidence at all points and then strategiacally penalize the network for making predictions of objects at locations were no object exists
        bce = ks.losses.binary_crossentropy(K.expand_dims(true_conf, axis = -1), pred_conf)
        conf_loss = (true_conf + (1 - true_conf) * mask_iou) * bce

        #8. take the sum of all the dimentions and reduce the loss such that each batch has a unique loss value
        loss_box = tf.cast(tf.reduce_sum(loss_box, axis=(1, 2, 3)), dtype = self.dtype)
        conf_loss = tf.cast(tf.reduce_sum(conf_loss, axis=(1, 2, 3)), dtype = self.dtype)
        # conf_loss = tf.cast(tf.reduce_sum(conf_loss2, axis=(1, 2)), dtype = self.dtype)
        class_loss = tf.cast(tf.reduce_sum(class_loss, axis=(1, 2, 3)), dtype = self.dtype)

        #9. i beleive tensorflow will take the average of all the batches loss, so add them and let TF do its thing
        return class_loss + conf_loss + loss_box

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


import tensorflow as tf
import tensorflow.keras as ks
from tensorflow.keras import backend as K
import tensorflow_datasets as tfds
from yolo.modeling.yolo_v3 import Yolov3
from yolo.utils.iou_utils import *
from yolo.utils.loss_utils import GridGenerator


class Yolo_Loss(ks.losses.Loss):
    def __init__(self, 
                 mask, 
                 anchors, 
                 scale_anchors = 1, 
                 num_extras = 0, 
                 ignore_thresh = 0.7, 
                 truth_thresh = 1, 
                 loss_type = "ciou", 
                 iou_normalizer = 1.0,
                 cls_normalizer = 1.0, 
                 scale_x_y = 1.0,
                 nms_kind = "greedynms",
                 beta_nms = 0.6,
                 reduction = tf.keras.losses.Reduction.NONE, 
                 path_key = None,
                 max_val = 5, 
                 name=None, 
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
        # match dtype to back end
        self.dtype = tf.keras.backend.floatx()
        #self._anchors = tf.convert_to_tensor([anchors[i] for i in mask], dtype= self.dtype)/scale_anchors #<- division done for testing

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
        self._max_value = tf.cast(max_val, dtype = self.dtype)

        # used in detection filtering
        self._beta_nms = tf.cast(beta_nms, dtype = self.dtype)
        self._nms_kind = nms_kind

        # grid comp
        self._anchor_generator = GridGenerator.get_generator_from_key(path_key)
        if self._anchor_generator == None:
            self._anchor_generator = GridGenerator(masks = mask, anchors = anchors, scale_anchors=scale_anchors, name = path_key)

        # metric struff
        self._loss_box = 0.0
        self._conf_loss = 0.0
        self._class_loss = 0.0
        self._iou = 0.0
        self._avg_iou = 0.0
        self._count = 0.0
        return

    @tf.function
    def print_error(self, pred_conf):
        if tf.reduce_any(tf.math.is_nan(pred_conf)):
            tf.print("\nerror")

    def call(self, y_true, y_pred):
        #1. generate and store constants and format output
        batch_size = tf.cast(tf.shape(y_pred)[0], dtype = tf.int32)
        width = tf.cast(tf.shape(y_pred)[1], dtype = tf.int32)
        height = tf.cast(tf.shape(y_pred)[2], dtype = tf.int32)
        grid_points, anchor_grid = self._anchor_generator(width, height, batch_size)
        y_pred = tf.reshape(y_pred, [batch_size, width, height, self._num, -1])

        fwidth = tf.cast(width, self.dtype)
        fheight = tf.cast(height, self.dtype)
        y_true = tf.cast(y_true, dtype = self.dtype)

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

        #5. apply generalized IOU or mse to the box predictions -> only the indexes where an object exists will affect the total loss -> found via the true_confidnce in ground truth 
        if self._loss_type == "giou":
            iou, giou = compute_giou(true_box, pred_box)
            mask_iou = tf.cast(iou < self._ignore_thresh, dtype = self.dtype)
            loss_box = (1 - giou) * self._iou_normalizer * true_conf
            loss_box = tf.math.minimum(loss_box, self._max_value)
        elif self._loss_type == "ciou":
            iou, ciou = compute_ciou(true_box, pred_box)
            mask_iou = tf.cast(iou < self._ignore_thresh, dtype = self.dtype)
            loss_box = (1 - ciou) * self._iou_normalizer * true_conf
            loss_box = tf.math.minimum(loss_box, self._max_value)
        else:
            # iou mask computation 
            iou = compute_iou(true_box, pred_box) 
            mask_iou = tf.cast(iou < self._ignore_thresh, dtype = self.dtype)
            
            # mse loss computation :: yolo_layer.c: scale = (2-truth.w*truth.h)
            scale = (2 - true_box[...,2] * true_box[...,3]) * self._iou_normalizer 
            loss_xy = tf.reduce_sum(K.square(true_xy - pred_xy), axis = -1)
            loss_wh = tf.reduce_sum(K.square(true_wh - pred_wh), axis = -1)
            loss_box = (loss_wh + loss_xy) * true_conf * scale 

        #6. apply binary cross entropy(bce) to class attributes -> only the indexes where an object exists will affect the total loss -> found via the true_confidnce in ground truth 
        class_loss = self._cls_normalizer * tf.reduce_sum(ks.losses.binary_crossentropy(K.expand_dims(true_class, axis = -1), K.expand_dims(pred_class, axis = -1)), axis= -1) * true_conf
        
        #7. apply bce to confidence at all points and then strategiacally penalize the network for making predictions of objects at locations were no object exists
        bce = ks.losses.binary_crossentropy(K.expand_dims(true_conf, axis = -1), pred_conf)
        conf_loss = (true_conf + (1 - true_conf) * mask_iou) * bce

        #8. take the sum of all the dimentions and reduce the loss such that each batch has a unique loss value
        loss_box = tf.reduce_mean(tf.cast(tf.reduce_sum(loss_box, axis=(1, 2, 3)), dtype = self.dtype))
        conf_loss = tf.reduce_mean(tf.cast(tf.reduce_sum(conf_loss, axis=(1, 2, 3)), dtype = self.dtype))
        class_loss = tf.reduce_mean(tf.cast(tf.reduce_sum(class_loss, axis=(1, 2, 3)), dtype = self.dtype))

        #9. i beleive tensorflow will take the average of all the batches loss, so add them and let TF do its thing
        loss = class_loss + conf_loss + loss_box

        #10. store values for use in metrics
        self._loss_box = loss_box
        self._conf_loss = conf_loss
        self._class_loss = class_loss

        # hits inf when all loss is neg or 0
        #self._avg_iou += tf.reduce_sum(iou) / tf.cast(tf.math.count_nonzero(iou), dtype=self.dtype)
        del grid_points
        del anchor_grid
        return loss

    def get_avg_iou():
        self._count += 1
        return self._avg_iou/self._count
    
    def get_classification_loss():
        return self._class_loss
    
    def get_box_loss():
        return self._loss_box

    def get_confidence_loss():
        return self._conf_loss

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





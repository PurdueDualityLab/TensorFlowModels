import tensorflow as tf
import tensorflow.keras as ks
from tensorflow.keras import backend as K

from yolo.utils.iou_utils import *

class Yolo_Loss_v1(ks.losses.Loss):
    def __init__(self,
                 coord_scale=5.0,
                 noobj_scale=0.5,
                 num_boxes=2,
                 num_classes=20,
                 ignore_thresh=0.7,
                 reduction=tf.keras.losses.Reduction.NONE,
                 name=None,
                 **kwargs):

        super(Yolo_Loss_v1, self).__init__(reduction=reduction,
                                           name=name,
                                           **kwargs)
        self._coord_scale = coord_scale
        self._noobj_scale = noobj_scale
        self._num_boxes = num_boxes
        self._num_classes = num_classes
        
    
    def call(self, y_true, y_pred):
        localization_loss = 0
        classification_loss = 0
        confidence_loss = 0
        
        class_start = self._num_boxes * 5

        # Seperate bounding box components from class probabilities
        pred_boxes = y_pred[..., :class_start]
        pred_class = y_pred[..., class_start:]

        true_boxes = y_true[..., :class_start]
        true_class = y_true[..., class_start:]

        # Get components from the box
        pred_boxes = tf.reshape(pred_boxes, [-1, self._num_boxes, 5])
        true_boxes = tf.reshape(true_boxes, [-1, self._num_boxes, 5])

        pred_boxes_xywh = pred_boxes[..., 0:4] 
        pred_xy = pred_boxes_xywh[..., 0:2]
        pred_wh = pred_boxes_xywh[..., 2:4]
        pred_confidence = pred_boxes[..., 4]

        true_boxes_xywh = true_boxes[..., 0:4] 
        true_xy = true_boxes_xywh[..., 0:2]
        true_wh = true_boxes_xywh[..., 2:4]
        true_confidence = true_boxes[..., 4]

        # Determine IOU of all predictor boxes vs gt boxes in each cell
        iou = compute_iou(true_boxes_xywh, pred_boxes_xywh)

        # Mask off the non-predictor bounding boxes based on iou
        predictor_mask = self.get_predictor_mask(iou)

        # Localization loss:
        loss_xy = tf.reduce_sum(K.square(true_xy - pred_xy), axis=-1) * predictor_mask * true_confidence
        loss_wh = tf.reduce_sum(K.square(tf.math.sqrt(true_wh) -
                                         tf.math.sqrt(pred_wh)), axis=-1) * predictor_mask * true_confidence

        localization_loss = self._coord_scale * (tf.reduce_sum(loss_xy + loss_wh))

        # Confidence loss:
        obj_loss = K.square(true_confidence - 
                                          pred_confidence) * predictor_mask * true_confidence
        noobj_loss = K.square(true_confidence - 
                                            pred_confidence) * (1 - predictor_mask) * (1 - true_confidence)
        
        confidence_loss = tf.reduce_sum(obj_loss + self._noobj_scale * noobj_loss)
        # Class Probability loss:
        # TODO: implement



        return localization_loss + classification_loss + confidence_loss
    
    def get_predictor_mask(self, iou, penalize_miss=True):
        # If penalize_miss is True, then if both bounding box iou's are 0
        # (ie. both bounding boxes completely miss the ground truth box),
        # then both boxes are considered in the loss

        highest_iou = tf.reduce_max(iou, axis=-1, keepdims=True)
        if penalize_miss:
            # The box with the highest iou is assigned 1, the other 0
            # If both boxes have iou 0, they are both assigned 1 
            highest_iou_mask = iou >= highest_iou
        else:
            # The box with the highest iou is assigned 1, the other 0
            # If both boxes have iou 0, they are both assigned 0
            highest_iou_mask = tf.math.divide_no_nan(iou, highest_iou)
        
        return tf.cast(highest_iou_mask, dtype=tf.float32)
        
    



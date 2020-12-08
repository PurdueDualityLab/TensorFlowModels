import tensorflow as tf
import tensorflow.keras as ks
from tensorflow.keras import backend as K

from yolo.ops import iou_ops
from yolo.ops.loss_utils import GridGenerator


class Yolo_Loss(object):
    def __init__(self,
                 classes,
                 mask,
                 anchors,
                 scale_anchors=1,
                 num_extras=0,
                 ignore_thresh=0.7,
                 truth_thresh=1,
                 loss_type="ciou",
                 iou_normalizer=1.0,
                 cls_normalizer=1.0,
                 scale_x_y=1.0,
                 nms_kind="greedynms",
                 beta_nms=0.6,
                 reduction=tf.keras.losses.Reduction.NONE,
                 path_key=None,
                 max_val=5,
                 use_tie_breaker=True,
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
        self._classes = tf.cast(classes, dtype=tf.int32)
        self._num = tf.cast(len(mask), dtype=tf.int32)
        self._num_extras = tf.cast(num_extras, dtype=tf.int32)
        self._truth_thresh = truth_thresh
        self._ignore_thresh = ignore_thresh
        self._masks = mask

        # used (mask_n >= 0 && n != best_n && l.iou_thresh < 1.0f) for id n != nest_n
        # checks all anchors to see if another anchor was used on this ground truth box to make a prediction
        # if iou > self._iou_thresh then the network check the other anchors, so basically
        # checking anchor box 1 on prediction for anchor box 2
        #self._iou_thresh = 0.213 # recomended use = 0.213 in [yolo]
        self._use_tie_breaker = tf.cast(use_tie_breaker, tf.bool)

        self._loss_type = tf.cast(loss_type, tf.string)
        self._iou_normalizer = iou_normalizer
        self._cls_normalizer = cls_normalizer
        self._scale_x_y = scale_x_y
        self._max_value = max_val

        # used in detection filtering
        self._beta_nms = beta_nms
        self._nms_kind = tf.cast(nms_kind, tf.string)

        # grid comp
        self._anchor_generator = GridGenerator(masks=mask,anchors=anchors,scale_anchors=scale_anchors)

        # metric struff
        self._path_key = path_key
        return

    @tf.function(experimental_relax_shapes=True)
    def print_error(self, pred_conf):
        if tf.stop_gradient(tf.reduce_any(tf.math.is_nan(pred_conf))):
            tf.print("\nerror: stop training")

    @tf.function(experimental_relax_shapes=True)
    def _get_label_attributes(self, width, height, batch_size, y_true, y_pred,
                              dtype):
        grid_points, anchor_grid = self._anchor_generator(width,
                                                          height,
                                                          batch_size,
                                                          dtype=dtype)
        y_true = build_grided_gt(
            y_true, tf.convert_to_tensor(self._masks, dtype=dtype), width,
            self._classes, tf.shape(y_pred), dtype, self._use_tie_breaker)
        return tf.stop_gradient(grid_points), tf.stop_gradient(
            anchor_grid), tf.stop_gradient(y_true)

    @tf.function(experimental_relax_shapes=True)
    def _get_predicted_box(self, width, height, unscaled_box, anchor_grid, grid_points):
        pred_xy = tf.math.sigmoid(unscaled_box[..., 0:2]) * self._scale_x_y - 0.5 * (self._scale_x_y - 1)
        pred_wh = unscaled_box[..., 2:4]
        box_xy = tf.stack([pred_xy[..., 0] / width, pred_xy[..., 1] / height], axis=-1) + grid_points
        box_wh = tf.math.exp(pred_wh) * anchor_grid
        pred_box = K.concatenate([box_xy, box_wh], axis=-1)
        return pred_xy, pred_wh, pred_box
        

    @tf.function(experimental_relax_shapes=True)
    def _scale_ground_truth_box(self, box, width, height, anchor_grid, grid_points, dtype):
        xy = tf.nn.relu(box[..., 0:2] - grid_points)
        xy = K.concatenate([
            K.expand_dims(xy[..., 0] * width, axis=-1),
            K.expand_dims(xy[..., 1] * height, axis=-1)
        ],
                           axis=-1)
        wh = tf.math.log(box[..., 2:4] / anchor_grid)
        wh = tf.where(tf.math.is_nan(wh), tf.cast(0.0, dtype=dtype), wh)
        wh = tf.where(tf.math.is_inf(wh), tf.cast(0.0, dtype=dtype), wh)
        return tf.stop_gradient(xy), tf.stop_gradient(wh)

    def rm_nan_inf(self, x):
        x = tf.where(tf.math.is_nan(x), tf.cast(0.0, dtype=x.dtype), x)
        x = tf.where(tf.math.is_inf(x), tf.cast(0.0, dtype=x.dtype), x)
        return x

    @tf.function(experimental_relax_shapes=True)
    def __call__(self, y_true, y_pred):
        #1. generate and store constants and format output
        shape = tf.shape(y_pred)
        batch_size, width, height = shape[0], shape[1], shape[2]
        y_pred = tf.cast(tf.reshape(y_pred, [batch_size, width, height, self._num, -1]), tf.float32)
        grid_points, anchor_grid, y_true = self._get_label_attributes(width, height, batch_size, y_true, y_pred, y_pred.dtype)

        fwidth = tf.cast(width, y_pred.dtype)
        fheight = tf.cast(height, y_pred.dtype)

        #2. split up layer output into components, xy, wh, confidence, class -> then apply activations to the correct items
        pred_xy, pred_wh, pred_box = self._get_predicted_box(
            fwidth, fheight, y_pred[..., 0:4], anchor_grid, grid_points)
        pred_conf = tf.expand_dims(tf.math.sigmoid(y_pred[..., 4]), axis=-1)
        pred_conf = self.rm_nan_inf(pred_conf)
        pred_class = tf.math.sigmoid(y_pred[..., 5:])
        self.print_error(pred_box)

        #3. split up ground_truth into components, xy, wh, confidence, class -> apply calculations to acchive safe format as predictions
        true_box = y_true[..., 0:4]
        true_conf = y_true[..., 4]
        true_class = y_true[..., 5:]

        #5. apply generalized IOU or mse to the box predictions -> only the indexes where an object exists will affect the total loss -> found via the true_confidnce in ground truth
        if self._loss_type == "giou":
            iou, giou = iou_ops.compute_giou(true_box, pred_box)
            mask_iou = tf.cast(iou < self._ignore_thresh, dtype=y_pred.dtype)
            loss_box = (1 - giou) * self._iou_normalizer * true_conf
            #loss_box = tf.math.minimum(loss_box, self._max_value)
        elif self._loss_type == "ciou":
            iou, ciou = iou_ops.compute_ciou(true_box, pred_box)
            mask_iou = tf.cast(iou < self._ignore_thresh, dtype=y_pred.dtype)
            loss_box = (1 - ciou) * self._iou_normalizer * true_conf
            #loss_box = tf.math.minimum(loss_box, self._max_value)
        else:
            # iou mask computation
            iou = iou_ops.compute_iou(true_box, pred_box)
            mask_iou = tf.cast(iou < self._ignore_thresh, dtype=y_pred.dtype)

            # mse loss computation :: yolo_layer.c: scale = (2-truth.w*truth.h)
            scale = (
                2 - true_box[..., 2] * true_box[..., 3]) * self._iou_normalizer
            true_xy, true_wh = self._scale_ground_truth_box(
                true_box, fwidth, fheight, anchor_grid, grid_points,
                y_pred.dtype)
            loss_xy = tf.reduce_sum(K.square(true_xy - pred_xy), axis=-1)
            loss_wh = tf.reduce_sum(K.square(true_wh - pred_wh), axis=-1)
            loss_box = (loss_wh + loss_xy) * true_conf * scale
            #loss_box = tf.math.minimum(loss_box, self._max_value)

        #6. apply binary cross entropy(bce) to class attributes -> only the indexes where an object exists will affect the total loss -> found via the true_confidnce in ground truth
        class_loss = self._cls_normalizer * tf.reduce_sum(
            ks.losses.binary_crossentropy(K.expand_dims(true_class, axis=-1),
                                          K.expand_dims(pred_class, axis=-1)),
            axis=-1) * true_conf

        #7. apply bce to confidence at all points and then strategiacally penalize the network for making predictions of objects at locations were no object exists
        bce = ks.losses.binary_crossentropy(K.expand_dims(true_conf, axis=-1),
                                            pred_conf)
        conf_loss = (true_conf + (1 - true_conf) * mask_iou) * bce

        #8. take the sum of all the dimentions and reduce the loss such that each batch has a unique loss value
        loss_box = tf.reduce_mean(
            tf.cast(tf.reduce_sum(loss_box, axis=(1, 2, 3)),
                    dtype=y_pred.dtype))
        conf_loss = tf.reduce_mean(
            tf.cast(tf.reduce_sum(conf_loss, axis=(1, 2, 3)),
                    dtype=y_pred.dtype))
        class_loss = tf.reduce_mean(
            tf.cast(tf.reduce_sum(class_loss, axis=(1, 2, 3)),
                    dtype=y_pred.dtype))

        #9. i beleive tensorflow will take the average of all the batches loss, so add them and let TF do its thing
        loss = class_loss + conf_loss + loss_box

        #10. store values for use in metrics
        recall50 = tf.reduce_mean(
            tf.math.divide_no_nan(
                tf.reduce_sum(tf.cast(tf.squeeze(pred_conf, axis=-1) > 0.5,
                                      dtype=true_conf.dtype) * true_conf,
                              axis=(1, 2, 3)),
                (tf.reduce_sum(true_conf, axis=(1, 2, 3)))))
        avg_iou = tf.math.divide_no_nan(
            tf.reduce_sum(iou),
            tf.cast(tf.math.count_nonzero(tf.cast(iou > 0,
                                                  dtype=y_pred.dtype)),
                    dtype=y_pred.dtype))
        return loss, loss_box, conf_loss, class_loss, avg_iou, recall50


@tf.function(experimental_relax_shapes=True)
def build_grided_gt(y_true, mask, size, classes, true_shape, dtype,
                    use_tie_breaker):
    """
    convert ground truth for use in loss functions
    Args: 
        y_true: tf.Tensor[] ground truth [box coords[0:4], classes_onehot[0:-1], best_fit_anchor_box]
        mask: list of the anchor boxes choresponding to the output, ex. [1, 2, 3] tells this layer to predict only the first 3 anchors in the total. 
        size: the dimensions of this output, for regular, it progresses from 13, to 26, to 52
    
    Return:
        tf.Tensor[] of shape [batch, size, size, #of_anchors, 4, 1, num_classes]
    """
    boxes = tf.cast(y_true["bbox"], dtype)
    classes = tf.one_hot(tf.cast(y_true["classes"], dtype=tf.int32),
                         depth=classes,
                         dtype=dtype)
    anchors = tf.cast(y_true["best_anchors"], dtype)

    batches = tf.shape(boxes)[0]
    num_boxes = tf.shape(boxes)[1]
    len_masks = tf.shape(mask)[0]

    full = tf.zeros([batches, size, size, len_masks, true_shape[-1]],
                    dtype=dtype)
    depth_track = tf.zeros((batches, size, size, len_masks), dtype=tf.int32)

    x = tf.cast(boxes[..., 0] * tf.cast(size, dtype=dtype), dtype=tf.int32)
    y = tf.cast(boxes[..., 1] * tf.cast(size, dtype=dtype), dtype=tf.int32)

    anchors = tf.repeat(tf.expand_dims(anchors, axis=-1), len_masks, axis=-1)

    update_index = tf.TensorArray(tf.int32, size=0, dynamic_size=True)
    update = tf.TensorArray(dtype, size=0, dynamic_size=True)
    const = tf.cast(tf.convert_to_tensor([1.]), dtype=dtype)
    mask = tf.cast(mask, dtype=dtype)

    i = 0
    anchor_id = 0
    for batch in range(batches):
        for box_id in range(num_boxes):
            if K.all(tf.math.equal(boxes[batch, box_id, 2:4], 0)):
                continue
            if K.any(tf.math.less(boxes[batch, box_id, 0:2], 0.0)) or K.any(
                    tf.math.greater_equal(boxes[batch, box_id, 0:2], 1.0)):
                continue
            if use_tie_breaker:
                for anchor_id in range(tf.shape(anchors)[-1]):
                    index = tf.math.equal(anchors[batch, box_id, anchor_id],
                                          mask)
                    if K.any(index):
                        #tf.print(anchor_id, anchors[batch, box_id, anchor_id])
                        p = tf.cast(K.argmax(tf.cast(index, dtype=tf.int32)),
                                    dtype=tf.int32)
                        uid = 1

                        used = depth_track[batch, y[batch, box_id],
                                           x[batch, box_id], p]
                        if anchor_id == 0:
                            # write the box to the update list
                            # the boxes output from yolo are for some reason have the x and y indexes swapped for some reason, I am not sure why
                            """peculiar"""
                            update_index = update_index.write(
                                i,
                                [batch, y[batch, box_id], x[batch, box_id], p])
                            value = K.concatenate([
                                boxes[batch, box_id], const, classes[batch,
                                                                     box_id]
                            ])
                            update = update.write(i, value)
                        elif tf.math.equal(used, 2) or tf.math.equal(used, 0):
                            uid = 2
                            # write the box to the update list
                            # the boxes output from yolo are for some reason have the x and y indexes swapped for some reason, I am not sure why
                            """peculiar"""
                            update_index = update_index.write(
                                i,
                                [batch, y[batch, box_id], x[batch, box_id], p])
                            value = K.concatenate([
                                boxes[batch, box_id], const, classes[batch,
                                                                     box_id]
                            ])
                            update = update.write(i, value)

                        depth_track = tf.tensor_scatter_nd_update(
                            depth_track,
                            [(batch, y[batch, box_id], x[batch, box_id], p)],
                            [uid])
                        i += 1
            else:
                index = tf.math.equal(anchors[batch, box_id, 0], mask)
                if K.any(index):
                    #tf.(0, anchors[batch, box_id, 0])
                    p = tf.cast(K.argmax(tf.cast(index, dtype=tf.int32)),
                                dtype=tf.int32)
                    update_index = update_index.write(
                        i, [batch, y[batch, box_id], x[batch, box_id], p])
                    value = K.concatenate(
                        [boxes[batch, box_id], const, classes[batch, box_id]])
                    update = update.write(i, value)
                    i += 1

    # if the size of the update list is not 0, do an update, other wise, no boxes and pass an empty grid
    if tf.math.greater(update_index.size(), 0):
        update_index = update_index.stack()
        update = update.stack()
        full = tf.tensor_scatter_nd_add(full, update_index, update)
    return full
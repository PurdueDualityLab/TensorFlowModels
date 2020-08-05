"""Contains common building blocks for yolo neural networks."""
import tensorflow as tf
import tensorflow.keras as ks
import tensorflow.keras.backend as K

import matplotlib.pyplot as plt
import matplotlib.patches as patches


#####NOT REAL ONLY FOR TESTING!!!!!!

@ks.utils.register_keras_serializable(package='yolo')
class YoloLayer(ks.layers.Layer):
    def __init__(self,
                 masks, 
                 anchors,
                 thresh, 
                 **kwargs):
        self._masks = masks
        self._anchors = list(tf.convert_to_tensor(anchors)/416)
        tf.print(self._anchors)
        self._thresh = tf.cast(thresh, dtype = tf.float32)
        super().__init__(**kwargs)
        return

    def _get_centers(self, lwidth, lheight, batch_size, num):
        """ generate a grid that is used to detemine the relative centers of the bounding boxs """
        x_left = tf.linspace(start = 0.0, stop = K.cast((lwidth - 1)/lwidth, dtype = tf.float32), num = lwidth)
        y_left = tf.linspace(start = 0.0, stop = K.cast((lheight - 1)/lheight, dtype = tf.float32), num = lheight)
        x_left, y_left = tf.meshgrid(x_left, y_left)

        x_y = tf.transpose(K.stack([x_left, y_left], axis = -1), perm = [1, 0, 2])
        x_y = tf.repeat(K.expand_dims(tf.repeat(K.expand_dims(x_y, axis = -2), num, axis = -2), axis = 0), batch_size, axis = 0)
        return x_y

    def _get_anchor_grid(self, width, height, batch_size, num, anchors):
        """ get the transformed anchor boxes for each dimention """
        # need to make sure this is correct
        # anchors = tf.reshape(anchors, -1)
        # anchors = tf.reshape(anchors, [1, tf.shape(anchors)[0]])
        # anchors = tf.repeat(anchors, width*height, axis = 0)
        # anchors = K.expand_dims(tf.reshape(anchors, [width, height, num, -1]), axis = 0)
        # anchors = tf.cast(tf.repeat(anchors, batch_size, axis = 0), dtype = tf.float32)
        anchors = tf.repeat(anchors, width*height, axis = 0)
        anchors = K.expand_dims(tf.reshape(anchors, [width, height, num, -1]), axis = 0)
        anchors = tf.cast(tf.repeat(anchors, batch_size, axis = 0), dtype = tf.float32)
        return anchors

    def _filter_output(self, data, mask):
        shape = tf.shape(data)
        anchors = tf.convert_to_tensor([self._anchors[j] for j in mask])

        data = tf.reshape(data, [shape[0], shape[1], shape[2], len(mask), -1])
        anchors = self._get_anchor_grid(shape[1], shape[2], shape[0], len(mask), anchors)
        centers = self._get_centers(shape[1], shape[2], shape[0], len(mask))

        box_xy = K.concatenate([K.expand_dims(tf.math.sigmoid(data[..., 0])/tf.cast(shape[1], dtype = tf.float32), axis = -1), K.expand_dims(tf.math.sigmoid(data[..., 1])/tf.cast(shape[2], dtype = tf.float32), axis = -1)], axis = -1) + centers
        box_wh = tf.math.exp(data[..., 2:4])*anchors
        box = K.concatenate([box_xy, box_wh], axis = -1)

        masked = tf.math.sigmoid(data[..., 4:]) > self._thresh
        objectness_mask = masked[..., 0] 
        classes = data[..., 5:]#masked[..., 1:]

        objectness = tf.math.sigmoid(data[..., 4])

        batched_boxes = []
        batched_scores = []
        batched_classes = []
        for batch in tf.range(shape[0]):
            tf.print(K.sum(tf.cast(objectness_mask[batch], dtype = tf.float32)))
            batched_boxes.append(tf.boolean_mask(box[batch], objectness_mask[batch]))
            batched_scores.append(tf.boolean_mask(objectness[batch], objectness_mask[batch]))
            batched_classes.append(tf.boolean_mask(classes[batch], objectness_mask[batch]))
        tf.print("\r")

        # tf.print(batched_boxes, batched_classes)
        return (batched_boxes, batched_scores, batched_classes)


    def _filter_truth(self, data, mask):
        shape = tf.shape(data)
        anchors = tf.convert_to_tensor([self._anchors[j] for j in mask])

        data = tf.reshape(data, [shape[0], shape[1], shape[2], len(mask), -1])

        box = data[..., 0:4]

        masked = data[..., 4:] > self._thresh
        objectness_mask = masked[..., 0] 
        classes = data[..., 5:]

        objectness = data[..., 4]

        batched_boxes = []
        batched_scores = []
        batched_classes = []
        for batch in tf.range(shape[0]):
            tf.print(K.sum(tf.cast(objectness_mask[batch], dtype = tf.float32)))
            batched_boxes.append(tf.boolean_mask(box[batch], objectness_mask[batch]))
            batched_scores.append(tf.boolean_mask(objectness[batch], objectness_mask[batch]))
            batched_classes.append(tf.boolean_mask(classes[batch], objectness_mask[batch]))
        tf.print("\r")

        # tf.print(batched_boxes, batched_classes)
        return (batched_boxes, batched_scores, batched_classes)

    def call(self, inputs, truth):
        keys = list(inputs.keys())
        batch_size = tf.shape(inputs[keys[0]])[0]
        tf.print(tf.shape(inputs[keys[0]])[1],"\n")
        outputs = []
        tests = []
        for key in keys:
            prediction = inputs[key]
            truth_b = truth[key]
            outputs.append(self._filter_output(prediction, self._masks[key]))
            tests.append(self._filter_truth(truth_b, self._masks[key]))
        
        box_concat = [[] for i in range(batch_size)]
        scores_concat = [[] for i in range(batch_size)]
        class_concat = [[] for i in range(batch_size)]
        for blist, slist, clist in outputs:
            for i in tf.range(batch_size):
                box_concat[i].extend(blist[i])
                scores_concat[i].extend(slist[i])
                class_concat[i].extend(clist[i])
        
        t_box_concat = [[] for i in range(batch_size)]
        t_scores_concat = [[] for i in range(batch_size)]
        t_class_concat = [[] for i in range(batch_size)]
        for blist, slist, clist in tests:
            for i in tf.range(batch_size):
                t_box_concat[i].extend(blist[i])
                t_scores_concat[i].extend(slist[i])
                t_class_concat[i].extend(clist[i])

        return box_concat, scores_concat, class_concat, t_box_concat, t_scores_concat, t_class_concat 

def dis_image(i, b):
    fig,ax = plt.subplots(1)
    ax.imshow(i)
    for box in b:
        tx = box[1] * i.shape[0]
        ty = box[0] * i.shape[1]
        rect = patches.Rectangle((tx, ty), box[3] * i.shape[1] - tx, box[2] * i.shape[0] - ty, linewidth=2,edgecolor='r',facecolor='none')
        ax.add_patch(rect)
    plt.show()

def yolo_to_tf(box):
    box = tf.convert_to_tensor(box)
    xmin = K.expand_dims(box[..., 0] - box[..., 2]/2, axis = -1)
    xmax = K.expand_dims(box[..., 0] + box[..., 2]/2, axis = -1)
    ymin = K.expand_dims(box[..., 1] - box[..., 3]/2, axis = -1)
    ymax = K.expand_dims(box[..., 1] + box[..., 3]/2, axis = -1)
    box_tf_form = K.concatenate([ymin, xmin, ymax, xmax], axis = -1)
    return box_tf_form

    

if __name__ == "__main__":
    from yolo.modeling.yolo_v3 import Yolov3
    from yolo.modeling.loss_functions.voc_test import *

    value = load_dataset(0, 3)
    model = Yolov3(classes = 80, boxes = 9, type = "spp")
    model.build(input_shape = (None, None, None, 3))
    model.load_weights_from_dn(dn2tf_backbone = True, dn2tf_head = True, config_file=None, weights_file=None)
    #model.load_weights_from_dn(dn2tf_backbone = True, dn2tf_head = False, config_file=None, weights_file="yolov3_416.weights")
    model.summary()
    # filter_l = YoloLayer(masks = {1024:[3,4,5], 256:[0,1,2]}, 
    #                      anchors = [(10,14),  (23,27),  (37,58),  (81,82),  (135,169),  (344,319)], 
    #                      thresh = 0.5)

    filter_l = YoloLayer(masks = {1024:[6, 7, 8], 512:[3,4,5] ,256:[0,1,2]}, 
                         anchors =[(10,13),  (16,30),  (33,23),  (30,61),  (62,45),  (59,119),  (116,90),  (156,198),  (373,326)], 
                         thresh = 0.5)

                         
    for image, label in value:
        pred = model(image)
        batched_boxes, batched_scores, batched_classes, t_boxes, t_scores, t_classes= filter_l(pred, label)

        for i, (box, scores, classif, t_box, t_score, t_class) in enumerate(zip(batched_boxes,batched_scores, batched_classes, t_boxes, t_scores, t_classes)):
            box = K.expand_dims(tf.convert_to_tensor(box), axis = 0)
            classif = K.expand_dims(tf.convert_to_tensor(classif), axis = 0)
            box_tf = yolo_to_tf(box)

            t_box = tf.convert_to_tensor(t_box)
            t_box_tf = yolo_to_tf(t_box)

            print(tf.shape(box))
            box_mask = tf.image.combined_non_max_suppression(tf.expand_dims(box_tf, axis=2), classif, 100, 100,  0.5, 0.05) # yeah this just sorts the scores, its not real
            print(box_mask)
            disp_box = box_mask.nmsed_boxes[0][:5]#tf.gather(box, box_mask)
            print(disp_box)

            #     print(i, len(box), len(classif), end = "\n")
            dis_image(image[i], disp_box)
            dis_image(image[i], t_box_tf)
            






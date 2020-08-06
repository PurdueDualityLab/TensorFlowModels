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
        # x_left = tf.linspace(start = K.cast(tf.cast(0, dtype = tf.int32)/lwidth, dtype = tf.float32), stop = K.cast((lwidth - 1)/lwidth, dtype = tf.float32), num = lheight)
        # y_left = tf.linspace(start = K.cast(tf.cast(0, dtype = tf.int32)/lheight, dtype = tf.float32), stop = K.cast((lheight - 1)/lheight, dtype = tf.float32), num = lwidth)
        
        # # idk why this works better
        x_left = tf.linspace(start = K.cast(tf.cast(1, dtype = tf.int32)/lwidth, dtype = tf.float32), stop = K.cast((lwidth)/lwidth, dtype = tf.float32), num = lheight)
        y_left = tf.linspace(start = K.cast(tf.cast(1, dtype = tf.int32)/lheight, dtype = tf.float32), stop = K.cast((lheight)/lheight, dtype = tf.float32), num = lwidth)
        x_left, y_left = tf.meshgrid(x_left, y_left)

        x_y = K.stack([x_left, y_left], axis = -1)
        #x_y = tf.transpose(x_y, perm = [1, 0, 2])
        x_y = tf.repeat(K.expand_dims(tf.repeat(K.expand_dims(x_y, axis = -2), num, axis = -2), axis = 0), batch_size, axis = 0)
        return x_y

    def _get_anchor_grid(self, width, height, batch_size, num, anchors):
        """ get the transformed anchor boxes for each dimention """
        #need to make sure this is correct
        anchors = tf.reshape(anchors, -1)
        anchors = tf.reshape(anchors, [1, tf.shape(anchors)[0]])
        anchors = tf.repeat(anchors, width*height, axis = 0)
        anchors = K.expand_dims(tf.reshape(anchors, [width, height, num, -1]), axis = 0)
        anchors = tf.cast(tf.repeat(anchors, batch_size, axis = 0), dtype = tf.float32)
        # anchors = tf.repeat(anchors, width*height, axis = 0)
        # anchors = K.expand_dims(tf.reshape(anchors, [width, height, num, -1]), axis = 0)
        # anchors = tf.cast(tf.repeat(anchors, batch_size, axis = 0), dtype = tf.float32)
        return anchors

    def _filter_output(self, data, mask):
        shape = tf.shape(data)
        anchors = tf.convert_to_tensor([self._anchors[j] for j in mask])

        data = tf.reshape(data, [shape[0], shape[1], shape[2], len(mask), -1])
        anchors = self._get_anchor_grid(shape[1], shape[2], shape[0], len(mask), anchors)
        centers = self._get_centers(shape[1], shape[2], shape[0], len(mask))
        
        # print(tf.math.sigmoid(data[..., 0:2]))

        box_xy = K.concatenate([K.expand_dims(tf.math.sigmoid(data[..., 0])/tf.cast(shape[1], dtype = tf.float32), axis = -1), K.expand_dims(tf.math.sigmoid(data[..., 1])/tf.cast(shape[2], dtype = tf.float32), axis = -1)], axis = -1) + centers
        box_wh = tf.math.exp(data[..., 2:4])*anchors
        box = K.concatenate([box_xy, box_wh], axis = -1)
        box = yolo_to_tf(box)

        probs = tf.math.sigmoid(data[..., 4:])

        objectness = probs[..., 0]
        classes = probs[..., 1:] 
        scaled = classes * tf.repeat(K.expand_dims(objectness, axis = -1), tf.shape(classes)[-1], axis = -1)
        mask = tf.cast(objectness > self._thresh, tf.float32)

        batched_boxes = []
        batched_scores = []
        batched_classes = []

        box = box * tf.repeat(K.expand_dims(mask, axis = -1), tf.shape(box)[-1], axis = -1)
        scaled = scaled * tf.repeat(K.expand_dims(mask, axis = -1), tf.shape(scaled)[-1], axis = -1)

        box = tf.reshape(box, [tf.shape(box)[0], -1, 4])
        scaled = tf.reshape(scaled, [tf.shape(scaled)[0], -1, tf.shape(scaled)[-1]])
        scaled = tf.where(scaled > self._thresh, scaled, 0.0)

        nms_boxes = tf.image.combined_non_max_suppression(tf.expand_dims(box, axis=2), scaled, 200, 200, 0.0) # yeah this just sorts the scores, its not real
        return nms_boxes


    def _filter_truth(self, data, mask):
        shape = tf.shape(data)
        anchors = tf.convert_to_tensor([self._anchors[j] for j in mask])

        data = tf.reshape(data, [shape[0], shape[1], shape[2], len(mask), -1])

        # print(tf.math.sigmoid(data[..., 0:2]))
        box = data[..., 0:4]
        box = yolo_to_tf(box)
        classes = data[..., 5:]

        box = tf.reshape(box, [tf.shape(box)[0], -1, 4])
        classes = tf.reshape(classes, [tf.shape(classes)[0], -1, tf.shape(classes)[-1]])
        nms_boxes = tf.image.combined_non_max_suppression(tf.expand_dims(box, axis=2), classes, 200, 200,  0.5, 0.05)
        return nms_boxes

    def merge_detections(self, labels):
        
        total_boxes = []
        total_classes = []
        total_conf = []
        for key in labels:
            total_boxes.append(key.nmsed_boxes)
            total_classes.append(key.nmsed_classes)
            total_conf.append(key.nmsed_scores)


        boxes = K.concatenate(total_boxes, axis = 1)
        classes = K.concatenate(total_classes, axis = 1)
        conf = K.concatenate(total_conf, axis = 1)
        batch_size = tf.shape(boxes)[0]

        temp = boxes > 0
        mask = tf.reduce_any(temp, axis= -1)
        mask = tf.reduce_any(mask, axis= 0)

        boxes = tf.boolean_mask(boxes, mask, axis = 1)
        classes = tf.boolean_mask(classes, mask,axis = 1)
        conf = tf.boolean_mask(conf, mask, axis = 1)
        #tf.print(boxes.shape, classes.shape, conf.shape, mask.shape)
        return boxes, classes, conf

    def call(self, inputs, truth):
        keys = list(inputs.keys())
        batch_size = tf.shape(inputs[keys[0]])[0]
        #tf.print(tf.shape(inputs[keys[0]])[1],"\n")
        outputs = []
        tests = []
        for key in keys:
            prediction = inputs[key]
            truth_b = truth[key]
            outputs.append(self._filter_output(prediction, self._masks[key]))
            # tests.append(self._filter_truth(truth_b, self._masks[key]))
        
        # for value, true in zip(outputs, tests):
        #     for j in range(1):
        outputs = self.merge_detections(outputs)
        # tests = self.merge_detections(tests)
        return outputs, tests

def dis_image(i, b, t = []):
    fig,ax = plt.subplots(1)
    ax.imshow(i)
    for box in b:
        tx = box[1] * i.shape[0]
        ty = box[0] * i.shape[1]
        rect = patches.Rectangle((tx, ty), box[3] * i.shape[1] - tx, box[2] * i.shape[0] - ty, linewidth=2,edgecolor='g',facecolor='none')
        ax.add_patch(rect)
    for box in t:
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

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    
    size = 10
    value = load_dataset(0, 1, size)
    model = Yolov3(classes = 80, boxes = 9, type = "regular")
    model.build(input_shape = (None, None, None, 3))
    #model.load_weights_from_dn(dn2tf_backbone = True, dn2tf_head = True, config_file=None, weights_file=None)
    model.load_weights_from_dn(dn2tf_backbone = True, dn2tf_head = True, config_file=None, weights_file="yolov3_416.weights")
    model.summary()
    # filter_l = YoloLayer(masks = {1024:[3,4,5], 256:[0,1,2]}, 
    #                      anchors = [(10,14),  (23,27),  (37,58),  (81,82),  (135,169),  (344,319)], 
    #                      thresh = 0.5)

    #predictions look like they are off center by a tad
    filter_l = YoloLayer(masks = {1024:[6, 7, 8], 512:[3,4,5] ,256:[0,1,2]}, 
                         anchors =[(10,13),  (16,30),  (33,23),  (30,61),  (62,45),  (59,119),  (116,90),  (156,198),  (373,326)], 
                         thresh = 0.5)

    import time   
    start = time.time()   
    with tf.device("/GPU:0"):       
        for image, label in value:       
            pred = model(image)
            #very slow
            outputs, tests = filter_l(pred, label)

        # for j in range(size):
            boxes = outputs[0][0]
        # #true_b = tests[0][0]
            dis_image(image[0], boxes)#, true_b)
    end = time.time() - start
    print(end)






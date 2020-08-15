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
        self._anchors = tf.convert_to_tensor(anchors)/416
        self._thresh = tf.cast(thresh, dtype = tf.float32)
        self._keys = list(masks.keys())
        self._len_keys = len(self._keys)
        super().__init__(**kwargs)
        return

    def _get_centers(self, lwidth, lheight, batch_size, num):
        """ generate a grid that is used to detemine the relative centers of the bounding boxs """
        x_left, y_left = tf.meshgrid(tf.range(1, lheight + 1), tf.range(1, lwidth + 1))
        x_y = K.stack([x_left, y_left], axis = -1)
        x_y = tf.cast(x_y, dtype = tf.float32)
        x_y = tf.repeat(tf.expand_dims(tf.repeat(tf.expand_dims(x_y, axis = -2), num, axis = -2), axis = 0), batch_size, axis = 0)
        return x_y 

    def _get_anchor_grid(self, width, height, batch_size, num, anchors):
        """ get the transformed anchor boxes for each dimention """
        anchors = tf.reshape(anchors, [1, -1])
        anchors = tf.repeat(anchors, width*height, axis = 0)
        anchors = tf.reshape(anchors, [1, width, height, num, -1])
        anchors = tf.cast(tf.repeat(anchors, batch_size, axis = 0), dtype = tf.float32)
        return anchors

    def _filter_output(self, data, mask):
        shape = tf.shape(data)
        anchors = tf.gather(self._anchors, mask)

        data = tf.reshape(data, [shape[0], shape[1], shape[2], len(mask), -1])
        anchors = self._get_anchor_grid(shape[1], shape[2], shape[0], len(mask), anchors)
        centers = self._get_centers(shape[1], shape[2], shape[0], len(mask))
    
        box_xy = (tf.math.sigmoid(data[..., 0:2]) + centers)/tf.cast(shape[1], dtype = tf.float32)
        box_wh = tf.math.exp(data[..., 2:4])*anchors
        box = K.concatenate([box_xy, box_wh], axis = -1)
        box = yolo_to_tf(box)

        objectness = tf.expand_dims(tf.math.sigmoid(data[..., 4]), axis = -1)
        classes = tf.math.sigmoid(data[..., 5:]) 
        scaled = classes * objectness #tf.repeat(objectness, tf.shape(classes)[-1], axis = -1)
        mask = tf.cast(objectness > self._thresh, tf.float32)

        box = box * mask #tf.repeat(mask, tf.shape(box)[-1], axis = -1)
        scaled = scaled * mask #tf.repeat(mask, tf.shape(scaled)[-1], axis = -1)

        box = tf.reshape(box, [tf.shape(box)[0], -1, 4])
        scaled = tf.where(scaled > self._thresh, scaled, 0.0)
        scaled = tf.reshape(scaled, [tf.shape(scaled)[0], -1, tf.shape(scaled)[-1]])

        temp = box > 0
        mask = tf.reduce_any(temp, axis= -1)
        mask = tf.reduce_any(mask, axis= 0)
        box = tf.boolean_mask(box, mask, axis = 1)[:, :200, :]
        scaled = tf.boolean_mask(scaled, mask, axis = 1)[:, :200, :]
        return (box, scaled)

    def call(self, inputs):
        boxes, classes = self._filter_output(inputs[self._keys[0]], self._masks[self._keys[0]])
        i = 1
        while i < self._len_keys:
            b, c = self._filter_output(inputs[self._keys[i]], self._masks[self._keys[i]])
            boxes = K.concatenate([boxes, b], axis = 1)
            classes = K.concatenate([classes, c], axis = 1)
            i += 1

        #nms = tf.image.combined_non_max_suppression(tf.expand_dims(boxes, axis=2), classes, 100, 100, 0.0)
        #return (nms.nmsed_boxes,  nms.nmsed_classes,  nms.nmsed_scores)
        return (boxes, classes)
    
    def build(self, input_shape):
        super().build(input_shape)
        return

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

@tf.function
def yolo_to_tf(box):
    box = tf.convert_to_tensor(box)
    minpoint = box[..., 0:2] - box[..., 2:4]/2
    maxpoint = box[..., 0:2] + box[..., 2:4]/2
    #return K.concatenate([maxpoint, minpoint], axis = -1)
    return tf.reverse(K.concatenate([maxpoint, minpoint], axis = -1), axis = [-1])

    

if __name__ == "__main__":
    from yolo.modeling.yolo_v3 import Yolov3, DarkNet53
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
    
    size = 90
    bsize = 1
    with tf.device("/CPU:0"): 
        value = load_testset(0, bsize, size//bsize)
    #model = DarkNet53(classes=1000, load_backbone_weights=True, config_file="yolov3.cfg", weights_file="yolov3_416.weights")
    model = Yolov3(classes = 80, boxes = 9, type = "regular")
    model.load_weights_from_dn(dn2tf_backbone = True, dn2tf_head = True, config_file=None, weights_file=None)
    model.summary()

    inputs = ks.layers.Input(shape=[None, None, 3])
    outputs = model(inputs) 
    # outputs = YoloLayer(masks = {1024:[6, 7, 8], 512:[3,4,5] ,256:[0,1,2]}, 
    #                     anchors =[(10,13),  (16,30),  (33,23),  (30,61),  (62,45),  (59,119),  (116,90),  (156,198),  (373,326)], 
    #                     thresh = 0.5)(outputs)
    # outputs = YoloLayer(masks = {1024:[3,4,5],256:[0,1,2]}, 
    #                     anchors =[(10,14),  (23,27),  (37,58),  (81,82),  (135,169),  (344,319)], 
    #                     thresh = 0.5)(outputs)
    run = ks.Model(inputs = [inputs], outputs = [outputs])
    run.build(input_shape = (None, None, 3))
    run.summary()

    # DarkNet Backbone
    # 0.04363393783569336,,            frame:89
    # fps:  24.129909627832046
    # end:  3.688368558883667
    # average per frame:  0.04144234335824345

    # Yolov3 Head + Darknet Backbone
    # 0.04388999938964844,,            frame:89
    # fps:  21.4014547870808
    # end:  4.15859580039978
    # average per frame:  0.04672579551010989

    # Yolov3 Head + Darknet Backbone + decoder
    # 0.04830646514892578,,            frame:89
    # fps:  20.118925162673673
    # end:  4.4236955642700195
    # average per frame:  0.04970444454235977

    import time   
    t = 1e-16
    i = 0
    with tf.device("/GPU:0"): 
        for image in value: 
            start = time.time() 
            outputs = run.predict(image)
            #outputs = run(image)
            #boxes = outputs[0][0][0]
            #dis_image(image[0], boxes)
            end = time.time() - start
            print(f"{end},\t\t frame:{i}", end = "\r")
            if i != 0:
                t += end
            i += 1
    print("\nfps: ", (size - 1)/t)
    print("end: ", t)
    print("average per frame: ", t/(i - 1 + 1e-16))







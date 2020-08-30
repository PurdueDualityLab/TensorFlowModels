import cv2
import datetime
import colorsys

import multiprocessing as mp
from multiprocessing import Process, Queue, Manager, Lock

import tensorflow as tf
import tensorflow.keras as ks
import tensorflow.keras.backend as K

from yolo.modeling.yolo_v3 import Yolov3, DarkNet53
import yolo.modeling.building_blocks as nn_blocks
    
def draw_box(image, boxes, classes, conf, colors, label_names):
    for i in range(boxes.shape[0]):
        if boxes[i][3] == 0:
            break
        box = boxes[i]
        cv2.rectangle(image, (box[0], box[2]), (box[1], box[3]), colors[classes[i]], 1)
        cv2.putText(image, "%s, %0.3f"%(label_names[classes[i]], conf[i]), (box[0], box[2]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[classes[i]], 1)
    return

def build_model(classes = 80, boxes = 9, use_mixed = True, w = 416, h = 416):
    if use_mixed:
        from tensorflow.keras.mixed_precision import experimental as mixed_precision
        # using mixed type policy give better performance than strictly float32
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_policy(policy)
        print('Compute dtype: %s' % policy.compute_dtype)
        print('Variable dtype: %s' % policy.variable_dtype)
        dtype = policy.compute_dtype
    else:
        dtype = tf.float32

    model = Yolov3(classes = classes, boxes = boxes, type = "regular", input_shape=(1, w, h, 3))
    model.load_weights_from_dn(dn2tf_backbone = True, dn2tf_head = True, config_file=None, weights_file="yolov3-regular.weights")
    model.summary()

    inputs = ks.layers.Input(shape=[w, h, 3])
    outputs = model(inputs) 
    b1, c1 = nn_blocks.YoloFilterCell(anchors = [(116,90),  (156,198),  (373,326)], thresh = 0.5, dtype=dtype)(outputs[1024])
    b2, c2 = nn_blocks.YoloFilterCell(anchors = [(30,61),  (62,45),  (59,119)], thresh = 0.5, dtype=dtype)(outputs[512])
    b3, c3 = nn_blocks.YoloFilterCell(anchors = [(10,13),  (16,30),  (33,23)], thresh = 0.5, dtype=dtype)(outputs[256])
    b = K.concatenate([b1, b2, b3], axis = 1)
    c = K.concatenate([c1, c2, c3], axis = 1)
    nms = tf.image.combined_non_max_suppression(tf.expand_dims(b, axis=2), c, 100, 100, 0.5, 0.5)
    run = ks.Model(inputs = [inputs], outputs = [nms.nmsed_boxes,  nms.nmsed_classes, nms.nmsed_scores])
    run.build(input_shape = (1, w, h, 3))
    run.summary()
    return run

class BufferVideo(object):
    def __init__(self, file_name, set_fps = 60):
        return

    def read(self, q, cap):
        return

    def fulsh_que(self, que):
        return
        
    def run(self):
        return

    def display(self, q):
        return

if __name__ == "__main__":
    cap = BufferVideo("test.mp4")
    cap.run()

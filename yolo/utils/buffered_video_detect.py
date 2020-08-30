import cv2
import datetime
import colorsys
import numpy as np
import time

# import multiprocessing as mp
# from multiprocessing import Process, Queue, Manager, Lock
import threading as t 
from queue import Queue

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

def prep_gpu():
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
    return

def build_model(classes = 80, boxes = 9, use_mixed = True, w = 416, h = 416, batch_size = None):
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

    model = Yolov3(classes = classes, boxes = boxes, type = "regular", input_shape=(batch_size, w, h, 3))
    model.load_weights_from_dn(dn2tf_backbone = True, dn2tf_head = True, config_file=None, weights_file="yolov3-regular.weights")
    model.summary()

    inputs = ks.layers.Input(shape=[w, h, 3])
    outputs = model(inputs) 

    # pack into a layer
    b1, c1 = nn_blocks.YoloFilterCell(anchors = [(116,90),  (156,198),  (373,326)], thresh = 0.5, dtype=dtype)(outputs[1024])
    b2, c2 = nn_blocks.YoloFilterCell(anchors = [(30,61),  (62,45),  (59,119)], thresh = 0.5, dtype=dtype)(outputs[512])
    b3, c3 = nn_blocks.YoloFilterCell(anchors = [(10,13),  (16,30),  (33,23)], thresh = 0.5, dtype=dtype)(outputs[256])
    b = K.concatenate([b1, b2, b3], axis = 1)
    c = K.concatenate([c1, c2, c3], axis = 1)
    nms = tf.image.combined_non_max_suppression(tf.expand_dims(b, axis=2), c, 100, 100, 0.5, 0.5)

    run = ks.Model(inputs = [inputs], outputs = [nms.nmsed_boxes,  nms.nmsed_classes, nms.nmsed_scores])
    run.build(input_shape = (batch_size, w, h, 3))
    run.summary()
    run.make_predict_function()
    return run

def int_scale_boxes(boxes, classes, width, height):
    boxes = K.stack([tf.cast(boxes[..., 1] * width, dtype = tf.int32),tf.cast(boxes[..., 3] * width, dtype = tf.int32), tf.cast(boxes[..., 0] * height, dtype = tf.int32), tf.cast(boxes[..., 2] * height, dtype = tf.int32)], axis = -1)
    classes = tf.cast(classes, dtype = tf.int32)
    return boxes, classes

def gen_colors(max_classes):
    hue = np.linspace(start = 0, stop = 1, num = max_classes)
    np.random.shuffle(hue)
    colors = []
    for val in hue:
        colors.append(colorsys.hsv_to_rgb(val, 0.75, 1.0))
    return colors

def get_coco_names(path = "/home/vishnu/Desktop/CAM2/TensorFlowModelGardeners/yolo/dataloaders/dataset_specs/coco.names"):
    f = open(path, "r")
    data = f.readlines()
    for i in range(len(data)):
        data[i] = data[i][:-1]
    return data

class BufferVideo(object):
    def __init__(self, file_name, set_fps = 60):
        self._file = file_name
        self._fps = 60
        #self._batch_size = 5 # 40 fps
        self._batch_size = 10 # 40 fps

        self._cap = cv2.VideoCapture(file_name)
        self._width = int(self._cap.get(3))
        self._height = int(self._cap.get(4))
        self._cap.set(5, 30)
        self._colors = gen_colors(80) #tf.cast(tf.convert_to_tensor(gen_colors(80)), dtype = tf.float32)
        self._labels = get_coco_names()

        self._load_que = Queue(self._batch_size * 5)
        self._display_que = Queue(10)
        self._running = True
        return

    def read(self, lock = None):
        timeout = 0 
        while (self._cap.isOpened() and self._running):
            while self._load_que.full() and self._running:
                timeout += 1
                time.sleep(0.05)
                if not self._running:
                    return
                if timeout >= self._fps: 
                    print("[EXIT] an error has occured, frames are not being pulled from que")
                    return 
            if not self._running:
                return
            timeout = 0
            success, image = self._cap.read()
            with tf.device("/CPU:0"):
                e = datetime.datetime.now()
                image = tf.cast(image, dtype = tf.float32)
                image = image/255
                self._load_que.put(image)
                f = datetime.datetime.now()
            time.sleep(0.02)
        return

    def display(self):
        start = time.time()
        l = 0
        tick = 0
        timeout = 0 
        print("here")
        while (self._cap.isOpened() and self._running):
            while self._display_que.empty() and self._running:
                timeout += 1
                time.sleep(0.01)
                if not self._running:
                    return
            if not self._running:
                return
            timeout = 0
            image, boxes, classes, conf = self._display_que.get()
            for i in range(image.shape[0]):
                draw_box(image[i], boxes[i], classes[i], conf[i], self._colors, self._labels)
                cv2.imshow("frame", image[i])
                time.sleep(0.01)
                l += 1
                if time.time() - start - tick >= 1:
                    tick += 1
                    print(f"current fps: {l}", end = "\r")
                    l = 0
                if cv2.waitKey(1) & 0xFF == ord('q') or not self._running:
                    break
        return
        
    def run(self):
        prep_gpu()
        with tf.device("/GPU:0"):
            model = build_model()
        
        if self._cap.isOpened():
            success, image = self._cap.read()
            with tf.device("/CPU:0"):
                e = datetime.datetime.now()
                image = tf.cast(image, dtype = tf.float32)
                image = image/255
                self._load_que.put(image)
                f = datetime.datetime.now()
        else:
            return

        load_thread = t.Thread(target=self.read, args=())
        load_thread.start()
        display_thread = t.Thread(target=self.display, args=())
        display_thread.start()

        try:
            while (self._cap.isOpened()):
                proc = []
                for i in range(self._batch_size):
                    if self._load_que.empty():
                        break
                    value = self._load_que.get()
                    proc.append(value)
                
                a = datetime.datetime.now()
                with tf.device("/GPU:0"):
                    image = tf.convert_to_tensor(proc)
                    pimage = tf.image.resize(image, (416, 416))
                    pred = model.predict(pimage)
                    boxes, classes = int_scale_boxes(pred[0], pred[1], self._width, self._height)
                b = datetime.datetime.now()
                timeout = 0

                while self._display_que.full() and not self._running:
                    time.sleep(0.01)
                self._display_que.put((image.numpy(), boxes.numpy(), classes.numpy(), pred[2]))
                    
            load_thread.join()
            display_thread.join()
            self._cap.release()
            cv2.destroyAllWindows()
        except KeyboardInterrupt:
            self._running = False
            load_thread.join()
            display_thread.join()
            self._cap.release()
            cv2.destroyAllWindows()
        return

if __name__ == "__main__":
    cap = BufferVideo("nyc.mp4")
    #cap = BufferVideo(0)
    cap.run()

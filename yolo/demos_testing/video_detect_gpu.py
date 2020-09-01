import cv2
import datetime
import colorsys
import numpy as np
import time

import threading as t 
from queue import Queue

import tensorflow as tf
import tensorflow.keras as ks
import tensorflow.keras.backend as K

from yolo.utils.testing_utils import prep_gpu, build_model, draw_box, int_scale_boxes, gen_colors, get_coco_names

class BufferVideo(object):
    def __init__(self, file_name, set_fps = 60):
        self._file = file_name
        self._fps = 60

        # fastest that maintains one to one 
        self._batch_size = 2 # 30 fps little to no in consistancy
        
        # fast but as close to one 2 one as possible
        if file_name == 0:
            self._batch_size = 5 # 40 fps more conistent frame to frame
        else:
            # faster but more potential for delay from input to output
            self._batch_size = 9 # 45 fps faster but less frame to frame consistent, it will remain consistant, but there is opertunity for more frames to be loaded than

        self._cap = cv2.VideoCapture(file_name)
        self._width = int(self._cap.get(3))
        self._height = int(self._cap.get(4))

        self._colors = gen_colors(80) #tf.cast(tf.convert_to_tensor(gen_colors(80)), dtype = tf.float32)
        self._labels = get_coco_names()

        self._load_que = Queue(self._batch_size * 5)
        self._display_que = Queue(10)
        self._running = True
        self._wait_time = 0.01

        self._read_fps = 1
        self._display_fps = 1
        self._latency = None#float('inf')
        self._batch_proc = 1
        self._frames = 0
        return

    def read(self, lock = None):
        start = time.time()
        l = 0
        tick = 0
        timeout = 0 
        while (self._cap.isOpened() and self._running):
            while self._load_que.full() and self._running:
                timeout += 1
                time.sleep(self._wait_time * 5)
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

            l += 1
            if time.time() - start - tick >= 1:
                tick += 1
                #print(f"\nread fps: {l}", end = "\n")
                self._read_fps = l
                l = 0
            time.sleep(self._wait_time)
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
                time.sleep(self._wait_time)
                if not self._running:
                    return
            if not self._running:
                return
            timeout = 0
            image, boxes, classes, conf = self._display_que.get()
            #print(len(self._display_que.queue))
            for i in range(image.shape[0]):
                draw_box(image[i], boxes[i], classes[i], conf[i], self._colors, self._labels)
                cv2.imshow("frame", image[i])
                time.sleep(self._wait_time)
                l += 1
                if time.time() - start - tick >= 1:
                    tick += 1
                    #print(f"\ndisplay fps: {l}", end = "\n\n")
                    self._display_fps = l
                    l = 0
                if cv2.waitKey(1) & 0xFF == ord('q') or not self._running:
                    break
        return
        
    def run(self):
        prep_gpu()
        with tf.device("/GPU:0"):
            model = build_model(name = "regular")
        
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
                    # we can watch it catch up to real time
                    #print(len(self._load_que.queue), end= " ")
                
                if len(proc) == 0:
                    time.sleep(self._wait_time)
                    continue
                    #print()

                a = datetime.datetime.now()
                with tf.device("/GPU:0"):
                    image = tf.convert_to_tensor(proc)
                    pimage = tf.image.resize(image, (416, 416))
                    pred = model.predict(pimage)
                    boxes, classes = int_scale_boxes(pred[0], pred[1], self._width, self._height)
                b = datetime.datetime.now()

                if self._frames >= 1000:
                    self._frames = 0
                    self._latency = None

                if self._latency != None:
                    self._latency += (b - a)
                else:
                    self._latency = (b - a)
                
                self._frames += image.shape[0]

                self._batch_proc = image.shape[0]
                timeout = 0

                while self._display_que.full() and not self._running:
                    time.sleep(self._wait_time)

                # print(len(self._display_que.queue))
                self._display_que.put((image.numpy(), boxes.numpy(), classes.numpy(), pred[2]))
                self.print_opt()
                    
            load_thread.join()
            display_thread.join()
            self._cap.release()
            cv2.destroyAllWindows()
        except KeyboardInterrupt:
            print("\n\n\n")
            self._running = False
            load_thread.join()
            display_thread.join()
            self._cap.release()
            cv2.destroyAllWindows()
        return
    
    def print_opt(self):
        print(f"                                \rlatency:, \033[1;32;40m{self._latency/self._frames * 1000} \033[0m ms", end = "\n")
        print("                                 \rread fps: \033[1;34;40m%d\033[0m " % (self._read_fps), end = "\n")
        print("                                 \rdisplay fps: \033[1;34;40m%d\033[0m" % (self._display_fps), end = "\n")
        print("                                 \rbatch processed: \033[1;37;40m%d\033[0m" % (self._batch_proc), end = "\n")
        print("\033[F\033[F\033[F\033[F\033[F", end="\n")
        return



if __name__ == "__main__":
    cap = BufferVideo("test.mp4")
    #cap = BufferVideo(0)
    cap.run()
    #rt_test()
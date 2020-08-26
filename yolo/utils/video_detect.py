import cv2
import time 
import tensorflow as tf
import tensorflow.keras as ks
import numpy as np
from yolo.modeling.yolo_v3 import Yolov3, DarkNet53
import yolo.modeling.building_blocks as nn_blocks
import datetime

'''Video Buffer using cv2'''


def video_processor(vidpath):
    cap = cv2.VideoCapture(vidpath)
    assert cap.isOpened()
    width = 0
    height = 0
    frame_count = 0
    img_array = []


    width = int(cap.get(3))
    height = int(cap.get(4))
    print('width, height, fps:', width, height, int(cap.get(5)))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cap.set(5, 30)

    i = 0
    t = 0
    start = time.time()
    tick = 0
    with tf.device("/GPU:0"): 
        model = build_model()
        model.make_predict_function()

    #output_writer = cv2.VideoWriter('yolo_output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), frame_count, (480, 640))  # change output file name if needed
    # may be batch by 2?
    pred = None
    while i <= frame_count:
        success, image = cap.read()

        with tf.device("/GPU:0"): 
            if t % 2 == 0:
                image = tf.cast(image, dtype = tf.float32)
                image = image/255
                image = tf.expand_dims(image, axis = 0)

                pimage = tf.image.resize(image, (416, 416))
                a = datetime.datetime.now()
                pred = model.predict(pimage)
                b = datetime.datetime.now()
                image = tf.image.draw_bounding_boxes(image, pred[0][0], [[0.0, 0.0, 1.0]])
                image = image[0]    
            else:
                image = tf.cast(image, dtype = tf.float32)
                image = image/255
                if pred != None:
                    image = tf.expand_dims(image, axis = 0)
                    image = tf.image.draw_bounding_boxes(image, pred[0][0], [[0.0, 0.0, 1.0]])
                    image = image[0]
            image = tf.image.resize(image, (height, width))

        cv2.imshow('frame', image.numpy())
        i += 1   
        t += 1  

        if time.time() - start - tick >= 1:
            tick += 1
            print(i, end = "\n")
            print(b - a)
            i = 0
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # for i in range(len(img_array)):
    #     output_writer.write(img_array[i])
    cap.release()
    cv2.destroyAllWindows()
    return

def webcam():
    cap = cv2.VideoCapture(0)
    assert cap.isOpened()
    cap.set(3, 416)
    cap.set(4, 416)
    i = 0
    start = time.time()
    tick = 0
    with tf.device("/GPU:0"): 
        model = build_model()
        model.make_predict_function()
    while(True):
        cap.set(cv2.CAP_PROP_FPS, 30)
        # Capture frame-by-frame
        ret, image = cap.read()

        # with tf.device("/GPU:0"): 
        #     image = tf.cast(image, dtype = tf.float32)
        #     image = tf.image.resize(image, (416, 416))
        #     image = image/255
        #     image = tf.expand_dims(image, axis = 0)
        #     pred = model.predict(image)
        #     image = tf.image.draw_bounding_boxes(image, pred[0][0], [[0.0, 0.0, 1.0]])
        #     image = image[0]    

        cv2.imshow('frame', image)#.numpy())
        i += 1     

        # print(time.time() - start)   

        if time.time() - start - tick >= 1:
            tick += 1
            print(i)
            i = 0

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

def build_model():
    #build backbone without loops
    # model = DarkNet53(classes=1000, load_backbone_weights=True, config_file="yolov3.cfg", weights_file="yolov3_416.weights")
    model = Yolov3(classes = 80, boxes = 9, type = "regular", input_shape=(None, None, None, 3))
    model.load_weights_from_dn(dn2tf_backbone = True, dn2tf_head = True, config_file=None, weights_file="yolov3_416.weights")
    model.summary()

    inputs = ks.layers.Input(shape=[None, None, 3])
    outputs = model(inputs) 
    outputs = nn_blocks.YoloLayer(masks = {1024:[6, 7, 8], 512:[3,4,5] ,256:[0,1,2]}, 
                                 anchors =[(10,13),  (16,30),  (33,23),  (30,61),  (62,45),  (59,119),  (116,90),  (156,198),  (373,326)], 
                                 thresh = 0.5)(outputs) # -> 1 frame cost
    # outputs = nn_blocks.YoloLayer(masks = {1024:[3,4,5],256:[0,1,2]}, 
    #                     anchors =[(10,14),  (23,27),  (37,58),  (81,82),  (135,169),  (344,319)], 
    #                     thresh = 0.5)(outputs)
    run = ks.Model(inputs = [inputs], outputs = [outputs])
    run.build(input_shape = (None, None, 3))
    run.summary()
    return run

def main():
    # vid_name = "yolo_vid.mp4"  # change input name if needed
    # video_processor(vid_name)
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

    video_processor("nyc.mp4")
    return 0


if __name__ == "__main__":
    main()

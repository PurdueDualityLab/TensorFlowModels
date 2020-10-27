#### ARGUMENT PARSER ####
from absl import app, flags
from absl.flags import argparse_flags
import sys
import os

parser = argparse_flags.ArgumentParser()

parser.add_argument(
    'model',
    metavar='model',
    default='regular',
    choices=('regular', 'spp', 'tiny'),
    type=str,
    help=
    'Name of the model. Defaults to regular. The options are ("regular", "spp", "tiny")',
    nargs='?')

parser.add_argument(
    'version',
    metavar='version',
    default='v3',
    choices=('v3', 'v4'),
    type=str,
    help='version of the model. Defaults to v3. The options are ("v3", "v4")',
    nargs='?')

parser.add_argument(
    'vidpath',
    default="",
    type=str,
    help='Path of the video stream to process. Defaults to the webcam.',
    nargs='?')

parser.add_argument(
    '--webcam',
    default=0,
    type=int,
    help=
    'ID number of the webcam to process as a video stream. This is only used if vidpath is not specified. Defaults to 0.',
    nargs='?')

if __name__ == '__main__':
    if '-h' in sys.argv or '--help' in sys.argv:
        parser.parse_args(sys.argv[1:])
        exit()

#### MAIN CODE ####

import cv2
import time

import numpy as np
import datetime
import colorsys

import tensorflow as tf
import tensorflow.keras as ks
import tensorflow.keras.backend as K

from yolo.utils.testing_utils import support_windows, prep_gpu, build_model, draw_box, int_scale_boxes, gen_colors, get_coco_names, get_draw_fn
'''Video Buffer using cv2'''


def preprocess_fn(image, dtype=tf.float32):
    image = tf.convert_to_tensor(image)
    image = tf.cast(image, dtype=dtype)
    image = image / 255
    return image


def rt_preprocess_fn(image, dtype=tf.float32):
    image = tf.convert_to_tensor(image)
    image = tf.cast(image, dtype=dtype)
    image = image / 255
    image = tf.image.resize(image, (416, 416))
    image = tf.expand_dims(image, axis=0)
    return image


class frame_iter():
    def __init__(self, file, preprocess_fn):
        self.file = file
        self.cap = cv2.VideoCapture(file)
        assert self.cap.isOpened()
        self.pre_process_fn = preprocess_fn
        self.width = int(self.cap.get(3))
        self.height = int(self.cap.get(4))
        self.frame = None
        self.time = None

    def __delete__(self):
        self.cap.release()

    def get_frame(self):
        while self.cap.isOpened():
            success, image = self.cap.read()
            self.frame = image
            e = datetime.datetime.now()
            image = self.pre_process_fn(image)
            f = datetime.datetime.now()
            self.time = (f - e)
            yield image

    def get_frame_rt(self):
        while self.cap.isOpened():
            success, image = self.cap.read()
            e = datetime.datetime.now()
            image = self.pre_process_fn(image)
            image = tf.expand_dims(image, axis=0)
            print(image.shape)
            yield image
            break

    def get_og_frame(self):
        return self.frame

    def rescale_frame(self, image):
        return tf.image.resize(image, (self.height, self.width))


def new_video_proc_rt(og_model_dir, rt_model_save_path, video_path):
    import yolo.demos.tensor_rt as trt
    video = frame_iter(video_path, rt_preprocess_fn)
    prep_gpu()
    trt.get_rt_model(og_model_dir, rt_model_save_path, video.get_frame_rt)
    # model = trt.get_func_from_saved_model(rt_model_save_path)
    # for frame in video.get_frame():
    #     pred = model(frame)
    #     frame = video.get_og_frame()
    #     cv2.imshow('frame', frame)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    return


def video_processor(model, version, vidpath, device="/CPU:0"):
    img_array = []

    i = 0
    t = 0
    start = time.time()
    tick = 0
    e, f, a, b, c, d = 0, 0, 0, 0, 0, 0
    if isinstance(model, str):
        with tf.device(device):
            model = build_model(name=model, model_version=version)
            model.make_predict_function()

    if hasattr(model, "predict"):
        predfunc = model.predict
        print("using pred function")
    else:
        predfunc = model
        print("using call function")

    colors = gen_colors(80)
    label_names = get_coco_names(
        path="yolo/dataloaders/dataset_specs/coco.names")
    print(label_names)

    # output_writer = cv2.VideoWriter('yolo_output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), frame_count, (480, 640))  # change output file name if needed
    pred = None
    cap = cv2.VideoCapture(vidpath)
    assert cap.isOpened()

    width = int(cap.get(3))
    height = int(cap.get(4))
    print('width, height, fps:', width, height, int(cap.get(5)))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while cap.isOpened():
        success, image = cap.read()

        #with tf.device(device):
        e = datetime.datetime.now()
        image = tf.cast(image, dtype=tf.float32)
        image = image / 255
        f = datetime.datetime.now()

        if t % 1 == 0:
            a = datetime.datetime.now()
            #with tf.device(device):
            pimage = tf.expand_dims(image, axis=0)
            pimage = tf.image.resize(pimage, (416, 416))
            pred = predfunc(pimage)
            b = datetime.datetime.now()

        image = image.numpy()
        if pred != None:
            c = datetime.datetime.now()
            boxes, classes = int_scale_boxes(pred["bbox"], pred["classes"],
                                             width, height)
            draw = get_draw_fn(colors, label_names, 'YOLO')
            draw_box(image, boxes[0].numpy(), classes[0].numpy(),
                     pred["confidence"][0], draw)
            d = datetime.datetime.now()

        cv2.imshow('frame', image)
        i += 1
        t += 1

        if time.time() - start - tick >= 1:
            tick += 1
            print_opt((((f - e) + (b - a) + (d - c))), i)
            i = 0
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return


def print_opt(latency, fps):
    print(
        f"                                \rlatency:, \033[1;32;40m{latency * 1000} \033[0m ms",
        end="\n")
    print("                                 \rfps: \033[1;34;40m%d\033[0m " %
          (fps),
          end="\n")
    print("\033[F\033[F\033[F", end="\n")
    return


def main(argv, args=None):
    if args is None:
        args = parser.parse_args(argv[1:])

    model = args.model
    if args.vidpath:
        if os.path.exists(args.vidpath):
            vidpath = args.vidpath
        else:
            print("Input video path doesn't exist.")
            exit()
    else:
        vidpath = args.webcam

    version = args.version

    # NOTE: on mac use the default terminal or the program will fail
    support_windows()
    video_processor(model, version, vidpath)
    return 0


def input_fn(cap, num_iterations):
    cap = cv2.VideoCapture(vidpath)
    assert cap.isOpened()
    while cap.isOpened():
        success, image = cap.read()
        yield image
    cap.release()


def main2():
    import contextlib
    import yolo.utils.tensor_rt as trt
    prep_gpu()

    def func(inputs):
        boxes = inputs["bbox"]
        classifs = tf.one_hot(inputs["classes"],
                              axis=-1,
                              dtype=tf.float32,
                              depth=80)
        nms = tf.image.combined_non_max_suppression(
            tf.expand_dims(boxes, axis=2), classifs, 200, 200, 0.5, 0.5)
        return {
            "bbox": nms.nmsed_boxes,
            "classes": nms.nmsed_classes,
            "confidence": nms.nmsed_scores,
        }

    name = "testing_weights/yolov4/full_models/v4_32"
    new_name = f"{name}_tensorrt"
    model = trt.TensorRT(saved_model=new_name,
                         save_new_path=new_name,
                         max_workspace_size_bytes=4000000000)
    model.compile()
    model.summary()
    model.set_postprocessor_fn(func)

    support_windows()
    video_processor(model, version=None, vidpath="testing_files/test2.mp4")

    return 0


if __name__ == "__main__":
    app.run(main)
    #main2()

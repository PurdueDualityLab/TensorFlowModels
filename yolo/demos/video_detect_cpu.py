#### ARGUMENT PARSER ####
from absl import app, flags
from absl.flags import argparse_flags
import sys

parser = argparse_flags.ArgumentParser()

parser.add_argument(
    'model',
    metavar='model',
    default='regular',
    choices=('regular', 'spp', 'tiny'),
    type=str,
    help='Name of the model. Defaults to regular. The options are ("regular", "spp", "tiny")',
    nargs='?'
)

parser.add_argument(
    'vidpath',
    default=None,
    type=str,
    help='Path of the video stream to process. Defaults to the webcam.',
    nargs='?'
)

parser.add_argument(
    '--webcam',
    default=0,
    type=int,
    help='ID number of the webcam to process as a video stream. This is only used if vidpath is not specified. Defaults to 0.',
    nargs='?'
)

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

from yolo.utils.testing_utils import support_windows, prep_gpu, build_model, draw_box, int_scale_boxes, gen_colors, get_coco_names


'''Video Buffer using cv2'''
def video_processor(model, vidpath, device = "/CPU:0"):
    width = 0
    height = 0
    frame_count = 0
    img_array = []


    width = int(cap.get(3))
    height = int(cap.get(4))
    print('width, height, fps:', width, height, int(cap.get(5)))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    i = 0
    t = 0
    start = time.time()
    tick = 0
    e,f,a,b,c,d = 0,0,0,0,0,0
    with tf.device(device):
        model = build_model(name = model, use_mixed=False)
        model.make_predict_function()
    colors = gen_colors(80)
    label_names = get_coco_names(path = "yolo/dataloaders/dataset_specs/coco.names")
    print(label_names)

    # output_writer = cv2.VideoWriter('yolo_output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), frame_count, (480, 640))  # change output file name if needed
    pred = None
    cap = cv2.VideoCapture(vidpath)
    assert cap.isOpened()

    while cap.isOpened():
        success, image = cap.read()

        with tf.device(device):
            e = datetime.datetime.now()
            image = tf.cast(image, dtype = tf.float32)
            image = image/255
            f = datetime.datetime.now()

        if t % 1 == 0:
            a = datetime.datetime.now()
            with tf.device(device):
                pimage = tf.expand_dims(image, axis = 0)
                pimage = tf.image.resize(pimage, (416, 416))
                pred = model.predict(pimage)
            b = datetime.datetime.now()

        image = image.numpy()
        if pred != None:
            c = datetime.datetime.now()
            boxes, classes = int_scale_boxes(pred["bbox"], pred["classes"], width, height)
            draw_box(image, boxes[0].numpy(), classes[0].numpy(), pred["confidence"][0], colors, label_names)
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
    print(f"                                \rlatency:, \033[1;32;40m{latency * 1000} \033[0m ms", end = "\n")
    print("                                 \rfps: \033[1;34;40m%d\033[0m " % (fps), end = "\n")
    print("\033[F\033[F\033[F", end="\n")
    return

def main(argv, args=None):
    if args is None:
        args = parser.parse_args(argv[1:])

    model = args.model
    vidpath = args.vidpath or args.webcam

    # NOTE: on mac use the default terminal or the program will fail
    support_windows()
    video_processor(model, vidpath)
    return 0


if __name__ == "__main__":
    app.run(main)

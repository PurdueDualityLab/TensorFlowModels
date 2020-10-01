import cv2
import datetime
import colorsys
import numpy as np
import time

import tensorflow as tf
import tensorflow.keras as ks
import tensorflow.keras.backend as K

try:
    from tensorflow.config import list_physical_devices, list_logical_devices
except ImportError:
    from tensorflow.config.experimental import list_physical_devices, list_logical_devices

import traceback


def support_windows():
    import platform
    if platform.system().lower() == 'windows':
        from ctypes import windll, c_int, byref
        stdout_handle = windll.kernel32.GetStdHandle(c_int(-11))
        mode = c_int(0)
        windll.kernel32.GetConsoleMode(c_int(stdout_handle), byref(mode))
        mode = c_int(mode.value | 4)
        windll.kernel32.SetConsoleMode(c_int(stdout_handle), mode)
    return


def draw_box(image, boxes, classes, conf, colors, label_names):
    for i in range(boxes.shape[0]):
        if boxes[i][3] == 0:
            break
        box = boxes[i]

        if type(conf) == type(None):
            cv2.rectangle(image, (box[0], box[2]), (box[1], box[3]), colors, 1)
        else:
            cv2.rectangle(image, (box[0], box[2]), (box[1], box[3]),
                          colors[classes[i]], 1)

        if type(conf) != type(None):
            cv2.putText(image,
                        "%s, %0.3f" % (label_names[classes[i]], conf[i]),
                        (box[0], box[2] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        colors[classes[i]], 1)
    return i


def build_model(name="regular",
                model_version="v3",
                classes=80,
                w=None,
                h=None,
                batch_size=None,
                saved=False,
                load_head=True,
                policy="float32",
                set_head=True):

    if model_version == "v3":
        from yolo import Yolov3
        model = Yolov3(classes=classes,
                       model=name,
                       input_shape=(batch_size, w, h, 3),
                       policy=policy)
    else:
        from yolo import Yolov4
        model = Yolov4(classes=classes,
                       model="regular",
                       input_shape=(batch_size, w, h, 3),
                       policy=policy)
    model.load_weights_from_dn(dn2tf_backbone=True,
                                dn2tf_head=load_head)
    return model


def filter_partial(end=255, dtype=tf.float32):
    import yolo.modeling.building_blocks as nn_blocks
    o1 = ks.layers.Input(shape=[None, None, 3, end // 3])
    o2 = ks.layers.Input(shape=[None, None, 3, end // 3])
    o3 = ks.layers.Input(shape=[None, None, 3, end // 3])

    outputs = {1024: o1, 512: o2, 256: o3}
    b1, c1 = nn_blocks.YoloGT(anchors=[(116, 90), (156, 198), (373, 326)],
                              thresh=0.5,
                              reshape=False,
                              dtype=dtype)(outputs[1024])
    b2, c2 = nn_blocks.YoloGT(anchors=[(30, 61), (62, 45), (59, 119)],
                              thresh=0.5,
                              reshape=False,
                              dtype=dtype)(outputs[512])
    b3, c3 = nn_blocks.YoloGT(anchors=[(10, 13), (16, 30), (33, 23)],
                              thresh=0.5,
                              reshape=False,
                              dtype=dtype)(outputs[256])
    b = K.concatenate([b1, b2, b3], axis=1)
    c = K.concatenate([c1, c2, c3], axis=1)

    run = ks.Model(inputs=[outputs], outputs=[b, c])
    return run


def build_model_partial(name="regular",
                        classes=80,
                        boxes=9,
                        ltype="giou",
                        use_mixed=True,
                        w=None,
                        h=None,
                        dataset_name="coco",
                        split='validation',
                        batch_size=1,
                        load_head=True,
                        fixed_size=False):
    from yolo.modeling.yolo_v3 import Yolov3
    model = Yolov3(classes=classes,
                   boxes=boxes,
                   type=name,
                   input_shape=(batch_size, w, h, 3))
    model.load_weights_from_dn(dn2tf_backbone=True, dn2tf_head=load_head)
    loss_fns = model.generate_loss(scale=w_scale)
    return model, loss_fns, model._boxes, model._masks


def prep_gpu(distribution=None):
    print(f"\n!--PREPPING GPU--! with stratagy ")
    #traceback.print_stack()
    if distribution == None:
        gpus = list_physical_devices('GPU')
        if gpus:
            try:
                # Currently, memory growth needs to be the same across GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    logical_gpus = list_logical_devices('GPU')
                    print(len(gpus), "Physical GPUs,", len(logical_gpus),
                          "Logical GPUs")
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)
                raise
        print()
    return


def int_scale_boxes(boxes, classes, width, height):
    boxes = K.stack([
        tf.cast(boxes[..., 1] * width, dtype=tf.int32),
        tf.cast(boxes[..., 3] * width, dtype=tf.int32),
        tf.cast(boxes[..., 0] * height, dtype=tf.int32),
        tf.cast(boxes[..., 2] * height, dtype=tf.int32)
    ],
                    axis=-1)
    classes = tf.cast(classes, dtype=tf.int32)
    return boxes, classes


def gen_colors(max_classes):
    hue = np.linspace(start=0, stop=1, num=max_classes)
    np.random.shuffle(hue)
    colors = []
    for val in hue:
        colors.append(colorsys.hsv_to_rgb(val, 0.75, 1.0))
    return colors


def get_coco_names(path="yolo/dataloaders/dataset_specs/coco.names"):
    with open(path, "r") as f:
        data = f.readlines()
    for i in range(len(data)):
        data[i] = data[i].strip()
    return data


def rt_test():
    cap = cv2.VideoCapture(0)
    cap.set(5, 30)
    start = time.time()
    l = 0
    tick = 0
    while (cap.isOpened()):
        rt, frame = cap.read()
        cv2.imshow("frame2", frame)
        l += 1
        if time.time() - start - tick >= 1:
            tick += 1
            print(f"current fps: {l}", end="\r")
            l = 0
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    return

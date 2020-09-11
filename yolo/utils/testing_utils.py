import cv2
import datetime
import colorsys
import numpy as np
import time

import tensorflow as tf
import tensorflow.keras as ks
import tensorflow.keras.backend as K

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
        cv2.rectangle(image, (box[0], box[2]), (box[1], box[3]), colors[classes[i]], 1)
        cv2.putText(image, "%s, %0.3f"%(label_names[classes[i]], conf[i]), (box[0], box[2]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[classes[i]], 1)
    return i

def build_model(name = "regular", classes = 80, boxes = 9, use_mixed = True, w = 416, h = 416, batch_size = None, saved = False):
    from yolo.modeling.yolo_v3 import Yolov3
    import yolo.modeling.building_blocks as nn_blocks

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

    if name != "tiny":
        masks = {"1024": [6,7,8], "512":[3,4,5], "256":[0,1,2]}
        anchors = [(10,13),  (16,30),  (33,23), (30,61),  (62,45),  (59,119), (116,90),  (156,198),  (373,326)]
        thresh = 0.5
        class_thresh = 0.5
        scale = 1
    else:
        masks = {"1024": [3,4,5], "256": [0,1,2]}
        anchors = [(10,14),  (23,27),  (37,58), (81,82),  (135,169),  (344,319)]
        thresh = 0.45
        class_thresh = 0.45
        scale = 1
    max_boxes = 200

    model = Yolov3(classes = classes, boxes = boxes, type = name, input_shape=(batch_size, w, h, 3))
    #tf.keras.utils.plot_model(model._head, to_file='model.png', show_shapes=True, show_layer_names=True,rankdir='TB', expand_nested=False, dpi=96)
    if not saved: 
        model.load_weights_from_dn(dn2tf_backbone = True, dn2tf_head = True, weights_file=f"yolov3-{name}.weights")


    w_scale  = 416 if w == None else w

    inputs = ks.layers.Input(shape=[w, h, 3])
    outputs = model(inputs)
    outputs = nn_blocks.YoloLayer(masks = masks, anchors= anchors, thresh = thresh, cls_thresh = class_thresh, max_boxes = max_boxes, dtype = dtype, scale_boxes=w_scale, scale_mult=scale)(outputs)

    run = ks.Model(inputs = [inputs], outputs = outputs)
    run.build(input_shape = (batch_size, w, h, 3))
    run.summary()
    run.make_predict_function()
    return run

def filter_partial(end = 255, dtype = tf.float32):
    import yolo.modeling.building_blocks as nn_blocks
    o1 = ks.layers.Input(shape=[None, None, 3, end//3])
    o2 = ks.layers.Input(shape=[None, None, 3, end//3])
    o3 = ks.layers.Input(shape=[None, None, 3, end//3])

    outputs = {1024: o1, 512: o2, 256: o3}
    b1, c1 = nn_blocks.YoloGT(anchors = [(116,90),  (156,198),  (373,326)], thresh = 0.5, reshape = False, dtype=dtype)(outputs[1024])
    b2, c2 = nn_blocks.YoloGT(anchors = [(30,61),  (62,45),  (59,119)], thresh = 0.5,  reshape = False, dtype=dtype)(outputs[512])
    b3, c3 = nn_blocks.YoloGT(anchors = [(10,13),  (16,30),  (33,23)], thresh = 0.5,  reshape = False, dtype=dtype)(outputs[256])
    b = K.concatenate([b1, b2, b3], axis = 1)
    c = K.concatenate([c1, c2, c3], axis = 1)

    run = ks.Model(inputs = [outputs], outputs = [b, c])
    return run

def build_model_partial(name = "regular", classes = 80, boxes = 9, ltype = "giou", use_mixed = True, w = None, h = None, dataset_name = "coco", split = 'validation', batch_size = 1, load_head = True, fixed_size = False):
    from yolo.modeling.yolo_v3 import Yolov3
    import yolo.modeling.building_blocks as nn_blocks
    from yolo.dataloaders.preprocessing_functions import preprocessing
    import tensorflow_datasets as tfds

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

    if name != "tiny":
        masks = {"1024": [6,7,8], "512":[3,4,5], "256":[0,1,2]}
        anchors = [(10,13),  (16,30),  (33,23), (30,61),  (62,45),  (59,119), (116,90),  (156,198),  (373,326)]
        thresh = 0.5
        class_thresh = 0.45
        scale = 1
    else:
        masks = {"1024": [3,4,5], "256": [0,1,2]}
        anchors = [(10,14),  (23,27),  (37,58), (81,82),  (135,169),  (344,319)]
        thresh = 0.45
        class_thresh = 0.45
        scale = 1
    max_boxes = 200

    model = Yolov3(classes = classes, boxes = boxes, type = name, input_shape=(batch_size, w, h, 3))
    model.load_weights_from_dn(dn2tf_backbone = True, dn2tf_head = load_head)#, weights_file=f"yolov3-{name}.weights")

    w_scale  = 416 if w == None else w
    loss_fns = load_loss(masks = masks, anchors = anchors, scale = w_scale, ltype=ltype)

    return model, loss_fns, anchors, masks

def prep_gpu(distribution = None):
    print(f"\n!--PREPPING GPU--! with stratagy {distribution}")
    traceback.print_stack()
    if distribution == None:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        print(gpus)
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
                raise
        print()
    return

def prep_gpu_limited(gb = 8):
    print(f"\n!--PREPPING GPU--! with limit: {gb}")
    traceback.print_stack()
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print(gpus)
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_virtual_device_configuration(gpu, [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*gb)])
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
            raise
    print()
    return

def load_loss(masks, anchors, scale, ltype = "mse", dtype = tf.float32):
    from yolo.modeling.functions.yolo_loss import Yolo_Loss
    loss_dict = {}
    for key in masks.keys():
        loss_dict[key] = Yolo_Loss(mask = masks[key],
                            anchors = anchors,
                            scale_anchors = scale, 
                            ignore_thresh = 0.7,
                            truth_thresh = 1, 
                            loss_type=ltype, 
                            dtype=dtype)
    print(loss_dict)
    return loss_dict

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

def get_coco_names(path = "yolo/dataloaders/dataset_specs/coco.names"):
    f = open(path, "r")
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
            print(f"current fps: {l}", end = "\r")
            l = 0
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    return

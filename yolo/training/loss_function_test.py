import cv2

import tensorflow as tf
import tensorflow.keras as ks
import tensorflow.keras.backend as K

import tensorflow_datasets as tfds
<<<<<<< HEAD
from yolo.utils.iou_utils import compute_iou
from .recall_metric import YoloMAP_recall
=======
from yolo.modeling.functions.iou import box_iou
from yolo.modeling.functions.recall_metric import YoloMAP_recall,YoloMAP_recall75, YoloMAP
>>>>>>> master

import matplotlib.pyplot as plt
import numpy as np
from absl import app
import time 

from yolo.utils.testing_utils import prep_gpu, build_model, build_model_partial, filter_partial, draw_box, int_scale_boxes, gen_colors, get_coco_names
prep_gpu()

from yolo.dataloaders.YoloParser import YoloParser

def lr_schedule(epoch, lr):
    if epoch == 60 or epoch == 90:
        lr = lr/10
    return lr

def gt_test():
    strat = tf.distribute.MirroredStrategy()
    with strat.scope():
        train, test = get_dataset(batch_size=1)
        pred_model = build_model(model_version="v4", set_head=True)#policy = "float32
        model = build_model(model_version="v4", set_head=False)#policy = "float32
        loss_fn = model.generate_loss(loss_type="ciou", scale= 416)
        #map_50 = YoloMAP_recall(name = "recall")
        partial_model = filter_partial()

    colors = gen_colors(80)
    coco_names = get_coco_names()
    i = 0 
    for image, label in train:
        box, classif = partial_model(label)
        pred = pred_model(image)
        item = model(image)

        #image = tf.image.draw_bounding_boxes(image, box, [[0.0, 1.0, 0.0]])
        #image = tf.image.draw_bounding_boxes(image, boxes, [[1.0, 0.0, 0.0]])
        image = image[0].numpy()
        boxes, classes = int_scale_boxes(pred["bbox"], pred["classes"], image.shape[0], image.shape[1])
        box, classif = int_scale_boxes(box, classif, image.shape[0], image.shape[1])
        draw_box(image, boxes[0].numpy(), classes[0].numpy(), None, [0,1,0], coco_names)
        draw_box(image, box[0].numpy(), classif[0].numpy(), None, [1,0,0], coco_names)

        plt.imshow(image)
        plt.show()

        for key in item.keys():
            print(key, loss_fn[key](label[key], item[key]))
        if i == 10:
            break 
        i += 1
    return

def get_dataset(batch_size = 10):
    import tensorflow_datasets as tfds
    from yolo.dataloaders.YoloParser import YoloParser
    train, info = tfds.load('coco', split = 'train', shuffle_files = False, with_info= True)
    test, info = tfds.load('coco', split = 'validation', shuffle_files = False, with_info= True) 

    train_size = tf.data.experimental.cardinality(train)
    test_size = tf.data.experimental.cardinality(test)
    print(train_size, test_size)

    parser = YoloParser(image_w = 416, image_h = 416, use_tie_breaker=True, fixed_size= False, jitter_im= 0.1, jitter_boxes= 0.005, anchors=[(10,13),  (16,30),  (33,23), (30,61),  (62,45),  (59,119), (116,90),  (156,198),  (373,326)])
    preprocess_train = parser.unbatched_process_fn(is_training = True)
    postprocess_train = parser.batched_process_fn(is_training = True)

    preprocess_test = parser.unbatched_process_fn(is_training = False)
    postprocess_test = parser.batched_process_fn(is_training = False)
    
    format_gt = parser.build_gt(is_training = True)
    
    train = train.map(preprocess_train).padded_batch(batch_size)
    train = train.map(postprocess_train)
    test = test.map(preprocess_test).padded_batch(batch_size)
    test = test.map(postprocess_test)
    # dataset = train.concatenate(test)

    # dataset = dataset.map(format_gt)
    # train = dataset.take(train_size//batch_size)
    # test = dataset.skip(train_size//batch_size)
    #train = train.map(format_gt)
    #test = test.map(format_gt)
    train_size = tf.data.experimental.cardinality(train)
    test_size = tf.data.experimental.cardinality(test)
    print(train_size, test_size)
    return train, test

def loss_test_(model_name = "regular"):
    #very large probelm, pre processing fails when you start batching
    prep_gpu_limited(gb = 8)
    from yolo.dataloaders.preprocessing_functions import preprocessing
    strat = tf.distribute.MirroredStrategy()
    with strat.scope():
        model, loss_fn, anchors, masks = build_model_partial(name=model_name, ltype = "giou", use_mixed= False, split="train", load_head = False, fixed_size= True)

        
        setname = "coco"
        dataset, Info = tfds.load(setname, split="train", with_info=True, shuffle_files=True, download=True)
        val, InfoVal = tfds.load(setname, split="validation", with_info=True, shuffle_files=True, download=True)
        dataset.concatenate(val)

        size = int(Info.splits["train"].num_examples)
        valsize = int(Info.splits["validation"].num_examples)
        
        dataset = preprocessing(dataset, 100, "detection", size + valsize, batch_size, 80, anchors= anchors, masks= masks, fixed=False, jitter = True)

        train = dataset.take(size//batch_size)
        test = dataset.skip(size//batch_size)

        map_50 = YoloMAP_recall(name = "recall")
    
    optimizer = ks.optimizers.SGD(lr=1e-3, momentum=0.9)
    callbacks = [ks.callbacks.LearningRateScheduler(lr_schedule)]#, tf.keras.callbacks.TensorBoard(log_dir="./logs", update_freq = 200)]
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=[map_50])
    try:
        model.summary()
        print(size//batch_size, epochs)
        model.fit(train, validation_data=test, shuffle=True, callbacks=callbacks, epochs = epochs)
        model.save_weights("weights/train_test_nojitter_helps_1")
    except KeyboardInterrupt:
        model.save_weights("weights/train_test_nojitter_helps_exit_early_1")
    return

def loss_test_fast(model_name = "regular", batch_size = 5, epochs = 3):
    #very large probelm, pre processing fails when you start batching
    prep_gpu()
    from yolo.dataloaders.preprocessing_functions import preprocessing
    strat = tf.distribute.MirroredStrategy()
    with strat.scope():
        model, loss_fn, anchors, masks = build_model_partial(name=model_name, ltype = "giou", use_mixed= False, split="train", load_head = False, fixed_size= True)

        setname = "coco"
        dataset, Info = tfds.load(setname, split="train", with_info=True, shuffle_files=True, download=True)
        val, InfoVal = tfds.load(setname, split="validation", with_info=True, shuffle_files=True, download=True)
        dataset.concatenate(val)

        size = int(Info.splits["train"].num_examples)
        valsize = int(Info.splits["validation"].num_examples)
        
        dataset = preprocessing(dataset, 100, "detection", size + valsize, batch_size, 80, anchors= anchors, masks= masks, fixed=False, jitter = True)

        train = dataset.take(size//batch_size)
        test = dataset.skip(size//batch_size)

        map_50 = YoloMAP_recall(name = "recall")
        Detection_50 = YoloMAP(name = "Det")
    
    optimizer = ks.optimizers.SGD(lr=1e-3, momentum=0.99)
    callbacks = [ks.callbacks.LearningRateScheduler(lr_schedule2)]#, tf.keras.callbacks.TensorBoard(log_dir="./logs", update_freq = 10)]
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=[map_50])#, Detection_50])
    try:
        model.summary()
        print(size//batch_size, epochs)
        model.fit(train, validation_data=test, shuffle=True, callbacks=callbacks, epochs = epochs)
        model.save_weights("weights/train_test_desk_fast_1")
    except KeyboardInterrupt:
        model.save_weights("weights/train_test_1")
        
    return

def loss_test():
    strat = tf.distribute.MirroredStrategy()
    with strat.scope():
        train, test = get_dataset(batch_size=2)
        # trianing fails at mixed precisions
        model = build_model(model_version="v4", set_head=False, load_head = True, policy = "float32")
        model.set_prediction_filter()
        loss_fn = model.generate_loss(loss_type="ciou")
        map_50 = YoloMAP_recall(name = "recall")
    
    optimizer = ks.optimizers.SGD(lr=1e-3) 
    callbacks = [ks.callbacks.LearningRateScheduler(lr_schedule)]#, tf.keras.callbacks.TensorBoard(log_dir="./logs")]
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=[map_50])
    model.fit(train, validation_data=test, shuffle=False, callbacks=callbacks)
    return 


def main(argv, args = None):
    loss_test()
    #loss_test_eager()
    #gt_test()
    return

if __name__ == "__main__":
    app.run(main)
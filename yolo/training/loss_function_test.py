import cv2

import tensorflow as tf
import tensorflow.keras as ks
import tensorflow.keras.backend as K

import tensorflow_datasets as tfds
from yolo.modeling.functions.iou import box_iou
from yolo.modeling.functions.recall_metric import YoloMAP_recall,YoloMAP_recall75, YoloMAP

import matplotlib.pyplot as plt
import numpy as np
from absl import app
import time 

from yolo.utils.testing_utils import prep_gpu, prep_gpu_limited, build_model, build_model_partial, filter_partial, draw_box, int_scale_boxes, gen_colors, get_coco_names, load_loss

def lr_schedule(epoch, lr):
    if epoch == 45 or epoch == 60:
        lr = lr/10
    return lr

def loss_test(model_name = "regular"):
    #very large probelm, pre processing fails when you start batching
    model, loss_fn, anchors, masks = build_model_partial(name=model_name, ltype = "giou", use_mixed= False, split="train", batch_size= 1, load_head = False, fixed_size= True)
    #model._head.trainable = True

    optimizer = ks.optimizers.SGD(lr=1e-4)
    # optimizer = ks.optimizers.Adam()
    map_50 = YoloMAP_recall(name = "recall")
    # map_75 = YoloMAP_recall75(name = "recall75")
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=[map_50])

    model.save_weights("weights/weights_test")
    model.load_weights("weights/weights_test")
    try:
        history = model.fit(dataset, epochs=20)
    except KeyboardInterrupt:
        model.save_weights("weights/custom_train1_adam_test")
        print()
        # plt.plot(history)
        # plt.show()
    
    return

def loss_test_eager(model_name = "regular", batch_size = 32, epochs = 80):
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
        
        dataset = preprocessing(dataset, 100, "detection", size + valsize, batch_size, 80, anchors= anchors, masks= masks, fixed=True)

        train = dataset.take(size//batch_size)
        test = dataset.skip(size//batch_size)

        map_50 = YoloMAP_recall(name = "recall")
        Detection_50 = YoloMAP(name = "Det")
    
    optimizer = ks.optimizers.SGD(lr=1e-3)
    callbacks = [ks.callbacks.LearningRateScheduler(lr_schedule), tf.keras.callbacks.TensorBoard(log_dir="./logs", update_freq = 200)]
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=[map_50, Detection_50])
    try:
        model.summary()
        print(size//batch_size, epochs)
        model.fit(train, validation_data=test, shuffle=True, callbacks=callbacks, epochs = epochs)
        model.save_weights("weights/train_test_helps_1")
    except:
        model.save_weights("weights/train_test_helps_exit_early_1")
    return

def gt_test():
    model, loss_fn, dataset, anchors, masks = build_model_partial(name="regular", use_mixed=False, split="validation", batch_size= 10)
    partial_model = filter_partial()
    pred_model = build_model()

    i = 0 

    for image, label in dataset:
        box, classif = partial_model(label)
        boxes, classes, conf = pred_model(image)
        pred = model(image)

        image = tf.image.draw_bounding_boxes(image, box, [[0.0, 1.0, 0.0]])
        image = tf.image.draw_bounding_boxes(image, boxes, [[1.0, 0.0, 0.0]])
        image = image[0].numpy()

        loss = 0
        sum_lab = 0
        for key in pred.keys():
            loss += loss_fn[key](label[key], pred[key])
        tf.print("loss: ", loss)
        print()
        if loss > 100:
            plt.imshow(image)
            plt.show()
        if i == 1000:
            break 
        i += 1

    return

def main(argv, args = None):
    loss_test_eager()
    #gt_test()
    return


if __name__ == "__main__":
    app.run(main)
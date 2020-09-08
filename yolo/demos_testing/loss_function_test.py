import cv2

import tensorflow as tf
import tensorflow.keras as ks
import tensorflow.keras.backend as K

import tensorflow_datasets as tfds
from yolo.modeling.functions.iou import box_iou

import matplotlib.pyplot as plt
import numpy as np
from absl import app

from yolo.utils.testing_utils import prep_gpu, build_model, build_model_partial, filter_partial, draw_box, int_scale_boxes, gen_colors, get_coco_names, load_loss
prep_gpu()

from yolo.dataloaders.preprocessing_functions import preprocessing

def loss_test(model_name = "regular"):
    #very large probelm, pre processing fails when you start batching
    model, loss_fn, dataset, anchors, masks = build_model_partial(name=model_name, ltype = "giou", use_mixed= False, split="train", batch_size= 1, load_head = False, fixed_size= True)
    model._head.trainable = True

    optimizer = ks.optimizers.SGD(lr=1e-4)
    model.compile(optimizer=optimizer, loss=loss_fn)

    model.save_weights("weights/weights_test")
    model.load_weights("weights/weights_test")
    try:
        history = model.fit(dataset, epochs=20)
    except KeyboardInterrupt:
        model.save_weights("weights/custom_train2_adam_test")
        print(history)
        plt.plot(history)
        plt.show()
    
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
    loss_test()
    #gt_test()
    return


if __name__ == "__main__":
    app.run(main)
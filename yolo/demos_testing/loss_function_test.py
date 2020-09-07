import cv2

import tensorflow as tf
import tensorflow.keras as ks
import tensorflow.keras.backend as K

import tensorflow_datasets as tfds
from yolo.modeling.functions.iou import box_iou

import matplotlib.pyplot as plt
import numpy as np
from absl import app

from yolo.utils.testing_utils import prep_gpu, build_model_partial, filter_partial, draw_box, int_scale_boxes, gen_colors, get_coco_names, load_loss
prep_gpu()

from yolo.dataloaders.preprocessing_functions import preprocessing

def loss_test(model_name = "regular"):
    model, loss_fn, dataset = build_model_partial(name=model_name, use_mixed= False, split="train")

    for image, label in dataset:
        pred = model(image)
        loss = 0
        sum_lab = 0
        for key in pred.keys():
            loss += loss_fn[key](label[key], pred[key])
            #tf.print(K.sum(pred[key]))
            sum_lab += K.sum(label[key])
        tf.print("labe1 sum: ", sum_lab)
        tf.print("loss: ", loss)
        print()
    return

def main(argv, args = None):
    loss_test()
    return


if __name__ == "__main__":
    app.run(main)
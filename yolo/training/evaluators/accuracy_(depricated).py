import cv2

import tensorflow as tf
import tensorflow.keras as ks
import tensorflow.keras.backend as K

import tensorflow_datasets as tfds
from yolo.modeling.functions.iou import box_iou

import matplotlib.pyplot as plt
import numpy as np

from yolo.utils.testing_utils import prep_gpu, build_model, filter_partial, draw_box, int_scale_boxes, gen_colors, get_coco_names, load_loss
prep_gpu()
#from yolo.modeling.functions.voc_test import load_dataset
from yolo.dataloaders.preprocessing_functions import preprocessing


def accuracy(model="regular", metrics=None):
    with tf.device("/GPU:0"):
        model = build_model(name=model, w=None, h=None)
        partial_filter = filter_partial()

    dataset, Info = tfds.load('coco',
                              split='validation',
                              with_info=True,
                              shuffle_files=True,
                              download=False)
    Size = int(Info.splits['test'].num_examples)
    dataset = preprocessing(dataset, 100, "detection", Size, 1, 80, False)
    #dataset = load_dataset(skip = 0, batch_size = 1, multiplier = 10000)
    optimizer = ks.optimizers.SGD()
    loss = load_loss()
    model.compile(optimizer=optimizer, loss=loss)
    #model.evaluate(dataset)
    counter = 0
    total_recall = 0
    total_accuracy = 0
    total_iou = 0

    for image, label in dataset:
        with tf.device("/GPU:0"):
            y_pred = model.predict(image)
            y_true = partial_filter(label)

        recall, accuracy, iou = test_iou_match(y_true, y_pred)

        total_recall += recall
        total_accuracy += accuracy
        total_iou += iou

        counter += 1
        print(
            "recall: %0.3f, accuracy: %0.3f, bbox iou: %0.3f, number of samples: %0.3f"
            % (total_recall / counter, total_accuracy / counter,
               total_iou / counter, counter),
            end="\r")
    return


def test_iou_match(gt, pred):
    boxt, catt = gt
    catt = catt.numpy()
    boxs, cats, conf = pred

    batch_size = boxs.shape[0]
    pred_num = boxs.shape[1]
    gt_num = boxt.shape[1]

    temp = np.zeros([batch_size, pred_num, gt_num])
    best_matches = []
    best_iou = []
    ac = []
    recall = []

    #make faster, currently o(n^2):
    for i in range(batch_size):
        for j in range(pred_num):
            box = boxs[i, j]
            if K.sum(box) == 0:
                break
            for k in range(gt_num):
                gt_box = boxt[i, k]
                iou = box_iou(box, gt_box)
                if iou > temp[i, j, k]:
                    temp[i, j, k] = iou
        try:
            indx = np.argmax(temp[0], axis=-1)[:j]
            best_matches.append(indx)
        except:
            indx = None
            best_matches.append([])

        if type(indx) != type(None):
            class_f = np.argmax(catt[i][indx], axis=-1)
            class_pred = []
            for p in range(j):
                class_pred.append(cats[i][p])

            if len(class_f) > 0:
                accuracy = 0
                seen_set = []
                for p in range(len(class_f)):
                    if class_f[p] == class_pred[p]:
                        accuracy += 1
                    if indx[p] not in seen_set:
                        seen_set.append(indx[p])

                accuracy /= len(class_f)
                ac.append(accuracy)
                rec = len(seen_set) / catt[i].shape[0]
                if not np.isnan(rec):
                    recall.append(rec)
                else:
                    recall.append(0)
        else:
            class_f = []
            class_pred = []
            ac.append(0)

        try:
            mean_iou = np.mean(
                np.trim_zeros(np.max(temp[i], axis=-1), trim='b'))
            if not np.isnan(mean_iou):
                best_iou.append(
                    np.mean(np.trim_zeros(np.max(temp[i], axis=-1), trim='b')))
            else:
                best_iou.append(0)
        except:
            best_iou.append(0)

    # print("classes", class_f)
    # print("classes_pred", class_pred)
    recall = np.mean(recall)
    biou = np.mean(best_iou)
    accuracy = np.mean(ac)

    if np.isnan(recall):
        recall = 0

    if np.isnan(biou):
        biou = 0

    if np.isnan(accuracy):
        accuracy = 0
    # a dict with the index corresponding to the best iou in pred
    # clip the dict to same size as the smallest one
    # then compute mse of the boxes
    # then compute the accuracy, return the avg accuracy for this image only
    return recall, accuracy, biou


if __name__ == "__main__":
    accuracy()

#all functions here are fir testing nothing to be included in final version. I am not garunteeing that any of this still works, if you need it make a brach and chnage it
import cv2

import tensorflow as tf
import tensorflow.keras as ks
import tensorflow.keras.backend as K

import tensorflow_datasets as tfds
from yolo.utils.iou_utils import compute_iou
from .recall_metric import YoloMAP_recall

import matplotlib.pyplot as plt
import numpy as np
from absl import app
import time

from yolo.utils.testing_utils import prep_gpu, build_model, draw_box, int_scale_boxes, gen_colors, get_coco_names
prep_gpu()

from yolo.dataloaders.YoloParser import YoloParser
from yolo.utils.box_utils import _xcycwh_to_yxyx


def lr_schedule(epoch, lr):
    if epoch == 60 or epoch == 90:
        lr = lr / 10
    return lr


def gt_test():
    import tensorflow_datasets as tfds
    strat = tf.distribute.MirroredStrategy()
    with strat.scope():
        train, info = tfds.load('voc',
                                split='train',
                                shuffle_files=True,
                                with_info=True)
        test, info = tfds.load('voc',
                               split='validation',
                               shuffle_files=False,
                               with_info=True)
        model = build_model(
            name="regular", model_version="v4", policy="mixed_float16"
        )  #, weights_file= "testing_weights/yolov3-regular.weights")
        #model.load_weights_from_dn
        model.get_summary()

        loss_fn = model.generate_loss(loss_type="ciou")
        train, test = model.process_datasets(train,
                                             test,
                                             jitter_boxes=None,
                                             jitter_im=0.1,
                                             batch_size=1,
                                             _eval_is_training=False)

    colors = gen_colors(80)
    coco_names = get_coco_names()
    i = 0
    for image, label in train:
        print(label.keys())
        pred = model.predict(image)

        image = tf.image.draw_bounding_boxes(image, pred["bbox"],
                                             [[1.0, 0.0, 0.0]])
        image = tf.image.draw_bounding_boxes(image,
                                             _xcycwh_to_yxyx(label["bbox"]),
                                             [[0.0, 1.0, 0.0]])
        image = image[0].numpy()

        plt.imshow(image)
        plt.show()

        loss, metric_dict = model.apply_loss_fn(label, pred["raw_output"])
        print(f"loss: {loss}")
        if i == 5:
            break
        i += 1
    return


def main(argv, args=None):
    #loss_test()
    #loss_test_eager()
    gt_test()
    return


if __name__ == "__main__":
    app.run(main)

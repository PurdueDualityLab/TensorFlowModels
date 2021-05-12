import dataclasses
import time

import matplotlib.pyplot as plt
import tensorflow as tf

from official.core import config_definitions as cfg
from official.core import input_reader
from official.modeling import hyperparams
from yolo.configs import darknet_classification as dcfg
from yolo.configs import yolo as yolocfg
from yolo.dataloaders import classification_input
from yolo.dataloaders import yolo_input as YOLO_Detection_Input
from yolo.dataloaders.decoders import tfds_coco_decoder
from yolo.ops import box_ops
from yolo.tasks import image_classification as imc
from yolo.tasks import yolo
from yolo.utils.demos import coco, utils


def test_yolo_input_task():
  with tf.device('/CPU:0'):
    config = yolocfg.YoloTask(
        model=yolocfg.Yolo(
            base='v4',
            min_level=3,
            norm_activation=yolocfg.common.NormActivation(activation='mish'),
            #norm_activation = yolocfg.common.NormActivation(activation="leaky"),
            #_boxes = ['(10, 14)', '(23, 27)', '(37, 58)', '(81, 82)', '(135, 169)', '(344, 319)'],
            #_boxes = ["(10, 13)", "(16, 30)", "(33, 23)","(30, 61)", "(62, 45)", "(59, 119)","(116, 90)", "(156, 198)", "(373, 326)"],
            # _boxes =  ['(10, 14)', '(23, 27)', '(37, 58)', '(81, 82)'], #, '(135, 169)'])
            _boxes=[
                '(12, 16)', '(19, 36)', '(40, 28)', '(36, 75)', '(76, 55)',
                '(72, 146)', '(142, 110)', '(192, 243)', '(459, 401)'
            ],
            filter=yolocfg.YoloLossLayer(use_nms=False)))
    task = yolo.YoloTask(config)

    # loading both causes issues, but oen at a time is not issue, why?
    config.train_data.dtype = 'float32'
    config.validation_data.dtype = 'float32'
    train_data = task.build_inputs(config.train_data)
    test_data = task.build_inputs(config.validation_data)
  return train_data, test_data


def test_classification_input():
  with tf.device('/CPU:0'):
    config = dcfg.ImageClassificationTask()

    task = imc.ImageClassificationTask(config)
    # config.train_data.dtype = "float32"
    # config.validation_data.dtype = "float32"
    train_data = task.build_inputs(config.train_data)
    test_data = task.build_inputs(config.validation_data)
  return train_data, test_data


def test_classification_pipeline():
  dataset, dsp = test_classification_input()
  for l, (i, j) in enumerate(dataset):
    plt.imshow(i[0].numpy())
    plt.show()
    if l > 30:
      break
  return




def test_pipeline():
  dataset, dsp = test_yolo_input_task()
  print(dataset)
  shind = 0
  dip = 0
  drawer = utils.DrawBoxes(labels=coco.get_coco_names(), thickness=1)
  ltime = time.time()
  for l, (i, j) in enumerate(dataset):
    ftime = time.time()
    print(ftime - ltime)

    print(tf.shape(i))
    # boxes = box_ops.xcycwh_to_yxyx(j['bbox'])
    # j["bbox"] = boxes
    i2 = drawer(i, j)  #
    i = tf.image.draw_bounding_boxes(i, j['bbox'], [[1.0, 0.0, 1.0]])

    gt = j['grid_form']

    obj3 = gt['3'][..., 4]
    obj4 = gt['4'][..., 4]
    obj5 = gt['5'][..., 4]

    cls3 = gt['3'][..., dip, 5]
    cls4 = gt['4'][..., dip, 5]
    cls5 = gt['5'][..., dip, 5]

    fig, axe = plt.subplots(1, 4)

    axe[0].imshow(i2[shind])
    axe[1].imshow(obj3[shind].numpy())
    axe[2].imshow(obj4[shind].numpy())
    axe[3].imshow(obj5[shind].numpy())

    fig.set_size_inches(18.5, 6.5, forward=True)
    plt.tight_layout()
    plt.show()

    ltime = time.time()

    if l >= 10:
      break


def time_pipeline():
  dataset, dsp = test_yolo_input_task()
  print(dataset)
  times = []
  ltime = time.time()
  for l, (i, j) in enumerate(dataset):
    ftime = time.time()
    # print(tf.reduce_min(i))
    # print(l , ftime - ltime, end = ", ")
    times.append(ftime - ltime)
    ltime = time.time()
    if l >= 100:
      break

  plt.plot(times)
  plt.show()

  print(f"total time {sum(times)}")


if __name__ == '__main__':
  test_pipeline()
  # test_classification_pipeline()
  # from yolo.ops import preprocessing_ops as po
  # dataset, dsp = test_yolo_input_task()

  # dataset = dataset.unbatch()
  # dataset = dataset.batch(4)
  # drawer = utils.DrawBoxes(labels=coco.get_coco_names(), thickness=1)

  # for l, (image, sample) in enumerate(dataset):

  #   image, boxes, classes, num_instances = po.randomized_cutmix_split(image, sample['bbox'], sample['classes'])

  #   # print(num_instances, tf.shape(boxes))
  #   sample = {
  #     'bbox': boxes,
  #     'classes': classes
  #   }

  #   image = drawer(image, sample)
  #   fig, axe = plt.subplots(1, 4)

  #   axe[0].imshow(image[0])
  #   axe[1].imshow(image[1])
  #   axe[2].imshow(image[2])
  #   axe[3].imshow(image[3])

  #   fig.set_size_inches(18.5, 6.5, forward=True)
  #   plt.tight_layout()
  #   plt.show()
  #   if l > 5:
  #     break

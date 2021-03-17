from yolo.tasks import image_classification as imc
from yolo.utils.demos import utils, coco
from yolo.tasks import yolo
from yolo.configs import yolo as yolocfg
from yolo.configs import darknet_classification as dcfg
from official.core import input_reader
from yolo.dataloaders import yolo_input as YOLO_Detection_Input
from yolo.dataloaders import classification_input
from yolo.dataloaders.decoders import tfds_coco_decoder
from yolo.ops import box_ops
import matplotlib.pyplot as plt
import dataclasses
from official.modeling import hyperparams
from official.core import config_definitions as cfg
import tensorflow as tf


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

  drawer = utils.DrawBoxes(labels=coco.get_coco_names(), thickness=1)
  for l, (i, j) in enumerate(dataset):

    # boxes = box_ops.xcycwh_to_yxyx(j['bbox'])
    # j["bbox"] = boxes
    i = drawer(i,
               j)  # tf.image.draw_bounding_boxes(i, boxes, [[1.0, 0.0, 1.0]])

    gt = j['grid_form']

    obj3 = gt['3'][..., 4]
    obj4 = gt['4'][..., 4]
    obj5 = gt['5'][..., 4]

    fig, axe = plt.subplots(1, 4)

    axe[0].imshow(i[0])
    axe[1].imshow(obj3[0].numpy())
    axe[2].imshow(obj4[0].numpy())
    axe[3].imshow(obj5[0].numpy())

    fig.set_size_inches(18.5, 6.5, forward=True)
    plt.tight_layout()
    plt.show()

    if l >= 30:
      break


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

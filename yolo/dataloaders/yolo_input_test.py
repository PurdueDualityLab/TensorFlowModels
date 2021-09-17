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

from yolo.modeling.layers import detection_generator
from official.vision.beta.ops import box_ops as bops
from official.vision.beta.tasks import retinanet
from official.vision.beta.configs import retinanet as retcfg

from official.core import train_utils
from official.modeling import performance
from official.core import task_factory
from yolo.run import load_model
from yolo.utils.run_utils import prep_gpu

prep_gpu()


def test_yolo_input_task():
  # with tf.device('/CPU:0'):
  experiment = "yolo_custom"
  # config_path = ["yolo/configs/experiments/yolov4/tpu/512-wd.yaml"]
  config_path = ["yolo/configs/experiments/yolov4-csp/tpu/640.yaml"]
  # config_path = ["yolo/configs/experiments/yolov4-csp-anchor-free/tpu/640-cstm2.yaml"]

  config = train_utils.ParseConfigOptions(
      experiment=experiment, config_file=config_path)
  params = train_utils.parse_configuration(config)
  config = params.task

  # anchor gen testing
  # config.model.boxes = None

  task = task_factory.get_task(params.task)

  config.train_data.global_batch_size = 1
  config.validation_data.global_batch_size = 1

  config.train_data.dtype = 'float32'
  config.validation_data.dtype = 'float32'

  config.validation_data.shuffle_buffer_size = 1
  config.train_data.shuffle_buffer_size = 1

  # if config.train_data.input_path == '':
  config.coco91to80 = False
  config.train_data.tfds_name = 'coco'
  config.validation_data.tfds_name = 'coco'
  config.train_data.tfds_split = 'train'
  config.validation_data.tfds_split = 'validation'
  config.train_data.tfds_data_dir = '/Users/vishnubanna/tensorflow_datasets'
  config.validation_data.tfds_data_dir = '/Users/vishnubanna/tensorflow_datasets'
  config.train_data.input_path = ''
  config.validation_data.input_path = ''
  # else:
  #   config.train_data.input_path = '/media/vbanna/DATA_SHARE/CV/datasets/COCO_raw/records/train*'
  #   config.validation_data.input_path = '/media/vbanna/DATA_SHARE/CV/datasets/COCO_raw/records/val*'

  with tf.device('/CPU:0'):
    train_data = task.build_inputs(config.train_data)
    test_data = task.build_inputs(config.validation_data)
  return train_data, test_data, config


def test_retinanet_input_task():
  with tf.device('/CPU:0'):
    config = retcfg.RetinaNetTask()
    task = retinanet.RetinaNetTask(config)

    # loading both causes issues, but oen at a time is not issue, why?
    config.train_data.dtype = 'float32'
    config.validation_data.dtype = 'float32'
    config.train_data.tfds_name = 'coco'
    config.validation_data.tfds_name = 'coco'
    config.train_data.tfds_split = 'train'
    config.validation_data.tfds_split = 'validation'
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


import time


def test_yolo_pipeline(is_training=True, num=30):
  dataset, dsp, config = test_yolo_input_task()
  print(dataset, dsp)
  # shind = 3
  dip = 0
  if config.coco91to80:
    drawer = utils.DrawBoxes(
        labels=coco.get_coco_names(
            path="yolo/dataloaders/dataset_specs/coco.names"),
        thickness=2,
        classes=91)
  else:
    drawer = utils.DrawBoxes(
        labels=coco.get_coco_names(
            path="yolo/dataloaders/dataset_specs/coco-91.names"),
        thickness=2,
        classes=91)
  ltime = time.time()

  data = dataset if is_training else dsp
  data = data.take(num)
  for l, (i, j) in enumerate(data):
    ftime = time.time()
    i_ = tf.image.draw_bounding_boxes(i, j['bbox'], [[1.0, 0.0, 1.0]])

    gt = j['true_conf']
    inds = j['inds']

    obj3 = tf.clip_by_value(gt['3'][..., 0], 0.0, 1.0)
    obj4 = tf.clip_by_value(gt['4'][..., 0], 0.0, 1.0)
    obj5 = tf.clip_by_value(gt['5'][..., 0], 0.0, 1.0)

    for shind in range(1):
      fig, axe = plt.subplots(1, 4)

      image = i[shind]
      boxes = j["bbox"][shind]
      classes = j["classes"][shind]
      confidence = j["classes"][shind]

      draw_dict = {
          'bbox': boxes,
          'classes': classes,
          'confidence': confidence,
      }
      # print(tf.cast(bops.denormalize_boxes(boxes, image.shape[:2]), tf.int32))
      image = drawer(image, draw_dict)

      (true_box, ind_mask, true_class, best_iou_match, num_reps) = tf.split(
          j['upds']['5'], [4, 1, 1, 1, 1], axis=-1)

      # true_xy = true_box[shind][..., 0:2] * 20
      # ind_xy = tf.cast(j['inds']['5'][shind][..., 0:2], true_xy.dtype)
      # x, y = tf.split(ind_xy, 2, axis=-1)
      # ind_xy = tf.concat([y, x], axis=-1)
      # tf.print(true_xy - ind_xy, summarize=-1)
      axe[0].imshow(image)
      axe[1].imshow(obj3[shind].numpy())
      axe[2].imshow(obj4[shind].numpy())
      axe[3].imshow(obj5[shind].numpy())

      fig.set_size_inches(18.5, 6.5, forward=True)
      plt.tight_layout()
      plt.show()

    ltime = time.time()

    if l >= 30:
      break


def time_pipeline(num=100):
  dataset, dsp, config = test_yolo_input_task()
  # dataset = dataset.take(100000)
  # print(dataset, dataset.cardinality())
  times = []
  ltime = time.time()
  for l, (i, j) in enumerate(dataset):
    ftime = time.time()

    times.append(ftime - ltime)
    ltime = time.time()
    print(times[-1], l)
    if l >= num:
      break

  plt.plot(times)
  plt.show()

  print(f"total time {sum(times)}")


def reduce(obj):
  return tf.math.ceil(
      tf.clip_by_value(
          tf.reduce_max(
              tf.reduce_sum(
                  tf.reshape(
                      obj, [obj.shape[0], obj.shape[1], obj.shape[2], 3, 3, 4]),
                  axis=-1),
              axis=-1), 0.0, 1.0))


def test_ret_pipeline():
  dataset, dsp = test_retinanet_input_task()
  print(dataset, dsp)

  shind = 0
  dip = 0
  drawer = utils.DrawBoxes(labels=coco.get_coco_names(), thickness=1)
  dfilter = detection_generator.YoloFilter()
  ltime = time.time()

  data = dsp
  data = data.take(10)

  lim1 = 0
  lim2 = 3
  for l, (i, j) in enumerate(data):
    ftime = time.time()

    i2 = drawer(i, j)  #
    i = tf.image.draw_bounding_boxes(i, j['bbox'], [[1.0, 0.0, 1.0]])

    gt = j['box_targets']

    obj3 = gt['3']
    obj4 = gt['4']
    obj5 = gt['5']
    obj6 = gt['6']
    obj7 = gt['7']

    obj3 = reduce(obj3)
    obj4 = reduce(obj4)
    obj5 = reduce(obj5)
    obj6 = reduce(obj6)
    obj7 = reduce(obj7)

    fig, axe = plt.subplots(1, 7)

    axe[0].imshow(i2[shind])
    axe[1].imshow(i[shind])
    axe[2].imshow(obj3[shind].numpy())
    axe[3].imshow(obj4[shind].numpy())
    axe[4].imshow(obj5[shind].numpy())
    axe[5].imshow(obj6[shind].numpy())
    axe[6].imshow(obj7[shind].numpy())

    fig.set_size_inches(10.5, 8.5, forward=True)
    plt.tight_layout()
    plt.show()

    ltime = time.time()

    if l >= 10:
      break


if __name__ == '__main__':
  # time_pipeline(num=10)
  test_yolo_pipeline(is_training=True, num=20)
  # test_yolo_pipeline(is_training=False, num=11)

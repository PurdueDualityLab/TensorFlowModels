import cv2
import colorsys
import numpy as np

import tensorflow as tf
import tensorflow.keras.backend as K


def draw_box(image, boxes, classes, conf, draw_fn):
  i = 0
  for i in range(boxes.shape[0]):
    if draw_fn(image, boxes[i], classes[i], conf[i]):
      i += 1
    else:
      return i
  return i


def get_draw_fn(colors, label_names, display_name):

  def draw_box_name(image, box, classes, conf):
    if box[3] == 0:
      return False
    cv2.rectangle(image, (box[0], box[2]), (box[1], box[3]), colors[classes], 1)
    cv2.putText(image, "%s, %0.3f" % (label_names[classes], conf),
                (box[0], box[2] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                colors[classes], 1)
    return True

  def draw_box(image, box, classes, conf):
    if box[3] == 0:
      return False
    cv2.rectangle(image, (box[0], box[2]), (box[1], box[3]), colors[classes], 1)
    return True

  if display_name:
    return draw_box_name
  else:
    return draw_box


def int_scale_boxes(boxes, classes, width, height):
  boxes = K.stack([
      tf.cast(boxes[..., 1] * width, dtype=tf.int32),
      tf.cast(boxes[..., 3] * width, dtype=tf.int32),
      tf.cast(boxes[..., 0] * height, dtype=tf.int32),
      tf.cast(boxes[..., 2] * height, dtype=tf.int32)
  ],
                  axis=-1)
  classes = tf.cast(classes, dtype=tf.int32)
  return boxes, classes


def gen_colors(max_classes):
  hue = np.linspace(start=0, stop=1, num=max_classes)
  np.random.shuffle(hue)
  colors = []
  for val in hue:
    colors.append(colorsys.hsv_to_rgb(val, 0.75, 1.0))
  return colors


def get_coco_names(path="yolo/dataloaders/dataset_specs/coco.names"):
  with open(path, "r") as f:
    data = f.readlines()
  for i in range(len(data)):
    data[i] = data[i].strip()
  return data

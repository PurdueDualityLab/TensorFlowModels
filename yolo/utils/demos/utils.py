from inspect import stack
import tensorflow as tf
import tensorflow.keras.backend as K
import socket
import struct
from typing import Callable
import numpy as np
import colorsys
import cv2
from concurrent.futures import ThreadPoolExecutor as pooler
import multiprocessing as mp
import os
# from concurrent.futures import ProcessPoolExecutor as pooler


def get_device(policy):
  if not isinstance(policy, str):
    return policy
  elif policy == "mirrored":
    strat = tf.distribute.MirroredStrategy()
    return strat.scope()
  elif policy == "oneDevice":
    strat = tf.distribute.OneDeviceStrategy()
    return strat.scope()
  elif policy == "tpu":
    strat = tf.distribute.TPUStrategy()
    return strat.scope()
  elif "GPU" in policy or "gpu" in policy:
    return tf.device(policy)
  return tf.device("/CPU:0")


def get_wait_time(wait_time, batch_size):
  if wait_time is None:
    return 0.001 * batch_size
  return wait_time


def set_policy(policy_name):
  if policy_name is None:
    return tf.float32
  from tensorflow.keras.mixed_precision import experimental as mixed_precision
  policy = mixed_precision.Policy(policy_name)
  mixed_precision.set_policy(policy)
  dtype = policy.compute_dtype
  return dtype


def get_run_fn(model, signature="serving_default") -> Callable:
  if ("saved_model" in str(type(model))):
    try:
      return model.signatures[signature]
    except BaseException:
      print("WARNING: signature not found in model")
      return model
  elif (hasattr(model, "predict")):
    return model.predict
  else:
    return model


def udp_socket(address, port, server=False):
  sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
  if server:
    sock.bind((address, port))
  return sock


def tcp_socket(address, port, server=False):
  sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  if server:
    sock.bind((address, port))
    sock.listen(1)
  return sock


def clear_buffer(s):
  while True:
    seg, addr = s.recvfrom(2**16)
    print(seg[0])
    if struct.unpack("B", seg[0:1])[0] == 1:
      print("finish emptying buffer")
      break


def gen_colors(max_classes):
  hue = np.linspace(start=0, stop=1, num=max_classes)
  np.random.shuffle(hue)
  colors = []
  for val in hue:
    colors.append(colorsys.hsv_to_rgb(val, 0.75, 1.0))
  return colors


def draw_box(image, boxes, classes, conf, draw_fn):
  i = 0
  for i in range(boxes.shape[0]):
    if draw_fn(image, boxes[i], classes[i], conf[i]):
      i += 1
    else:
      return i
  return i


def get_draw_fn(self, colors, label_names, display_name):

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

  if display_name or label_names is None:
    return draw_box_name
  else:
    return draw_box


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

  if display_name or label_names is None:
    return draw_box_name
  else:
    return draw_box


def int_scale_boxes(boxes, classes, width, height):
  boxes = K.stack([
      tf.cast(boxes[..., 0] * height, dtype=tf.int32),
      tf.cast(boxes[..., 1] * width, dtype=tf.int32),
      tf.cast(boxes[..., 2] * height, dtype=tf.int32),
      tf.cast(boxes[..., 3] * width, dtype=tf.int32)
  ],
                  axis=-1)
  classes = tf.cast(classes, dtype=tf.int32)
  return boxes, classes


class DrawBoxes(object):

  def __init__(self, classes=80, labels=None, display_names=True, thickness=2):

    self._classes = classes
    self._colors = gen_colors(classes)
    self._labels = labels
    self._display_names = display_names
    self._thickness = thickness
    self._draw_fn = self._get_draw_fn(self._colors, self._labels,
                                      self._display_names)
    return

  def _get_draw_fn(self, colors, label_names, display_name):

    def draw_box_name(image, box, classes, conf):
      if box[1] - box[0] == 0 or box[3] - box[2] == 0:
        return False
      
      x0 = int(box[1])
      y0 = int(box[0])
      x1 = int(box[3])
      y1 = int(box[2])   
      classes = int(classes)

      color = colors[classes]
      txt_bk_color = (np.array(colors[classes]) * 0.7).tolist()

      if conf is not None:
        text = '{}:{:.1f}%'.format(label_names[classes], conf * 100)
      else:
        text = '{}'.format(label_names[classes])
      txt_color = (0, 0, 0) if np.mean(colors[classes]) > 0.5 else (1, 1, 1)
      font = cv2.FONT_HERSHEY_SIMPLEX

      txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]

      cv2.rectangle(image, (x0, y0), (x1, y1), color, self._thickness)

      if conf is None:
        x = (box[1] + box[0]) // 2
        y = (box[3] + box[2]) // 2
        cv2.circle(
            image, (x, y),
            radius=0,
            color=colors[classes],
            thickness=self._thickness * 3)

      cv2.rectangle(
          image,
          (x0, y0 + 1),
          (x0 + txt_size[0] + 1, y0 + int(1.5*txt_size[1])),
          txt_bk_color,
          -1
      )
      cv2.putText(image, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)
      return True

    def draw_box(image, box, classes, conf):
      if box[1] - box[0] == 0 or box[3] - box[2] == 0:
        return False
      cv2.rectangle(image, (box[0], box[2]), (box[1], box[3]), colors[classes],
                    1)
      cv2.circle(
          image, (box[0], box[2]),
          radius=0,
          color=colors[classes],
          thickness=self._thickness)
      return True

    if display_name and label_names is not None:
      return draw_box_name
    else:
      return draw_box

  def _draw(self, image, boxes, classes, conf, num_detections = None):
    i = 0

    if num_detections is None:
      lim = boxes.shape[0]
    else:
      lim = int(num_detections)
    for i in range(lim):
      if self._draw_fn(image, boxes[i], classes[i], conf[i]):
        i += 1
    return image

  def _parent(self, image, boxes, classes, conf):

    def func(i):
      return self._draw(image[i], boxes[i], classes[i], conf[i])

    return func

  def __call__(self, image, results, stacked = True):
    """ expectoed format = {bbox: , classes: , "confidence": }"""

    boxes = results["bbox"]
    classes = results["classes"]

    try:
      conf = results["confidence"]
    except:
      conf = results["classes"]

    # if "num_detections" in results.keys():
    #   num_detections = results["num_detections"]
    # else:
    num_detections = None

    if not isinstance(image, list):
      ndims = len(image.shape)
      if ndims == 4:
        boxes, classes = int_scale_boxes(boxes, classes, image.shape[2],
                                         image.shape[1])
      elif ndims == 3:
        boxes, classes = int_scale_boxes(boxes, classes, image.shape[1],
                                         image.shape[0])
    else:
      img = image[0]
      boxes, classes = int_scale_boxes(boxes, classes, img.shape[1],
                                       img.shape[0])

    if hasattr(image, "numpy"):
      image = image.numpy()
    if hasattr(boxes, "numpy"):
      boxes = boxes.numpy()
    if hasattr(classes, "numpy"):
      classes = classes.numpy()
    if not isinstance(conf, type(None)) and hasattr(conf, "numpy"):
      conf = conf.numpy()

    if not isinstance(image, list):

      if ndims == 4:
        images = []
        func = self._parent(image, boxes, classes, conf)
        #images = tf.map_fn(func, elems = tf.range(image.shape[0]), parallel_iterations=10)
        for i in range(image.shape[0]):
          im = func(i)
          images.append(im)
        image = np.stack(images, axis=0)
      elif ndims == 3:
        if len(boxes.shape) == 2:
          self._draw(image, boxes, classes, conf)
        else:
          self._draw(image, boxes[0], classes[0], conf[0])
    else:
      images = []
      for i, im in enumerate(image):
        if hasattr(im, "numpy"):
          im = im.numpy()
        self._draw(im, boxes[i], classes[i], conf[i], num_detections = num_detections[i])
        images.append(im)
      if stacked:
        image = np.stack(images, axis=0)
      else:
        image = images
    return image

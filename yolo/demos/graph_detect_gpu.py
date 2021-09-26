import cv2
import datetime
import colorsys
import numpy as np
import time

import threading as t
from queue import Queue

import tensorflow as tf
import tensorflow.keras as ks
import tensorflow.keras.backend as K
from tensorflow.python.util.nest import _yield_sorted_items

from yolo.utils.run_utils import support_windows
from yolo.utils.demos.coco import draw_box
from yolo.utils.demos.coco import get_draw_fn
from yolo.utils.demos.coco import gen_colors
from yolo.utils.demos.coco import get_coco_names
from yolo.utils.demos.coco import int_scale_boxes
from yolo.utils.demos import utils
# from utils.demos import utils
from yolo.utils.run_utils import prep_gpu
from yolo.configs import yolo as exp_cfg
from yolo.tasks.yolo import YoloTask
import yolo.demos.three_servers.video_server as video_t
from official.vision.beta.ops import box_ops
from yolo.ops import preprocessing_ops


def letterbox(image, desired_size, letter_box = True):
  """Letter box an image for image serving."""

  with tf.name_scope('letter_box'):
    image_size = tf.cast(preprocessing_ops.get_image_shape(image), tf.float32)

    scaled_size = tf.cast(desired_size, image_size.dtype)
    if letter_box:
      scale = tf.minimum(
          scaled_size[0] / image_size[0], scaled_size[1] / image_size[1])
      scaled_size = tf.round(image_size * scale)
    else:
      scale = 1.0

    # Computes 2D image_scale.
    image_scale = scaled_size / image_size
    image_offset = tf.cast((desired_size - scaled_size) * 0.5, tf.int32)
    offset = (scaled_size - desired_size) * 0.5
    scaled_image = tf.image.resize(image, 
                        tf.cast(scaled_size, tf.int32), 
                        method = 'nearest')

    output_image = tf.image.pad_to_bounding_box(
        scaled_image, image_offset[0], image_offset[1], 
                      desired_size[0], desired_size[1])

    
    image_info = tf.stack([
        image_size,
        tf.cast(desired_size, dtype=tf.float32),
        image_scale,
        tf.cast(offset, tf.float32)])
    return output_image, image_info

DESIRED_SIZE = [640, 640]
# DESIRED_SIZE = [512, 512]
# DESIRED_SIZE = [416, 416]
LETTERBOX = True
HALF_PRECISION = True
BATCH_SIZE=1

def preprocess_fn(image):
  if HALF_PRECISION:
    image = tf.cast(image, tf.float16)
  else:
    image = tf.cast(image, tf.float32)

  image/=255
  pimage, info = letterbox(image, DESIRED_SIZE, LETTERBOX)

  if HALF_PRECISION:
    info = tf.cast(info, tf.float16)
  else:
    info = tf.cast(info, tf.float32)  

  image = tf.cast(image, tf.float32)
  return image, pimage, info

def video(cap):
  def run():      
    while (cap.isOpened()):
      success, image = cap.read()
      if success:
        yield tf.convert_to_tensor(image)
  dataset = tf.data.Dataset.from_generator(run, output_signature=(
    tf.TensorSpec(shape = [None, None, 3], dtype = tf.uint8)
  ))
  # return dataset.map(preprocess_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(BATCH_SIZE)
  return dataset.map(preprocess_fn).batch(BATCH_SIZE)

def undo_info(boxes, num_detections, info):
  mask = tf.sequence_mask(num_detections, maxlen=tf.shape(boxes)[1])
  boxes = tf.cast(tf.expand_dims(mask, axis = -1), boxes.dtype) * boxes

  inshape = tf.expand_dims(info[:, 1, :], axis = 1)
  ogshape = tf.expand_dims(info[:, 0, :], axis = 1)
  scale = tf.expand_dims(info[:, 2, :], axis = 1)
  offset = tf.expand_dims(info[:, 3, :], axis = 1)

  boxes = box_ops.denormalize_boxes(boxes, inshape)
  boxes += tf.tile(offset, [1, 1, 2])
  boxes /= tf.tile(scale, [1, 1, 2])
  boxes = box_ops.normalize_boxes(boxes, ogshape)

  boxes = box_ops.clip_boxes(boxes, ogshape)    
  return boxes

def int_scale_boxes(boxes, classes, image):
  shape = tf.cast(preprocessing_ops.get_image_shape(image), boxes.dtype)
  shape = tf.expand_dims(shape, axis = 0)
  boxes = tf.cast(box_ops.denormalize_boxes(boxes, shape), dtype=tf.int32)
  classes = tf.cast(classes, dtype=tf.int32)
  return boxes, classes

def run_fn(model):
  @tf.function
  def run(image, pimage, info):
    predictions = model(pimage)
    predictions["bbox"] = undo_info(
      predictions["bbox"], predictions["num_detections"], info)
    # predictions["bbox"], predictions["classes"] = int_scale_boxes(
    #   predictions["bbox"], predictions["classes"], image)
    return image, predictions
  return run


if __name__ == '__main__':
  import yolo.utils.export.tensor_rt as trt
  from yolo import run
  import os
  video_file = 0 #"../../Videos/nyc2.mp4"


  # name = "/home/vbanna/Research/TensorFlowModels/cache/saved_models/yolo_v4_csp"
  # new_name = f"{name}_tensorrt"
  # model = trt.TensorRT(saved_model=name, save_new_path=new_name, max_workspace_size_bytes=4000000000, max_batch_size=5)
  # model.convertModel()
  # model.compile()
  # model.summary()
  # # model.set_postprocessor_fn(func)



  # 
  config = [os.path.abspath('yolo/configs/experiments/yolov4-csp/inference/640.yaml')]
  model_dir = os.path.abspath("../checkpoints/640-baseline-e13")
  # config = [os.path.abspath('yolo/configs/experiments/yolov4-nano/inference/416-3l.yaml')]
  # model_dir = os.path.abspath("../checkpoints/416-3l-baseline-e1")
  # config = [os.path.abspath('yolo/configs/experiments/yolov4/inference/512.yaml')]
  # model_dir = os.path.abspath("../checkpoints/512-wd-baseline-e1")


  task, model, params = run.load_model(experiment='yolo_custom', config_path=config, model_dir=model_dir)
  model.fuse()

  # model = tf.keras.models.save_model(model, "cache/saved_models/yolo_v4_csp", include_optimizer = False)
  model(tf.ones([1] + DESIRED_SIZE + [3]))
  # model.save("cache/saved_models/yolo_v4_csp", include_optimizer = False)

  model_fn = run_fn(model)
  labels = get_coco_names(path='yolo/dataloaders/dataset_specs/coco.names')
  draw_fn = utils.DrawBoxes(
        classes=80, labels=labels,
        display_names=True,
        thickness=2)
  
  cap = cv2.VideoCapture(video_file)
  width, height = cap.get(3), cap.get(4)
  display = video_t.DisplayThread() #res = (width, height))
  video_stream = video(cap)

  display.start()
  latency = 0.0
  alpha = 0.9
  for frames in video_stream:
    start = time.time()
    images, preds = model_fn(*frames)  
    end = time.time()
    images = draw_fn(images, preds, stacked = False)
    display.put_all(images)
    
    # images, pimages, info = frames
    # ims = []
    # images = images.numpy()
    # for i, im in enumerate(images):
    #   ims.append(im)
    # display.put_all(ims)
    latency = alpha*latency + (1 - alpha)*((end - start)/BATCH_SIZE)
    print(f"fps: {display.fps}, latency: {latency}", end = "\r")

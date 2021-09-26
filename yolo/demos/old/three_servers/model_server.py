import yolo.demos.three_servers.video_server as video_t
import struct
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

from yolo.utils.run_utils import support_windows
from yolo.utils.run_utils import prep_gpu
from yolo.utils.demos import utils
from yolo.utils.demos import coco
import traceback


class ModelServer(object):

  def __init__(self,
               model=None,
               preprocess_fn=None,
               postprocess_fn=None,
               process_dims=416,
               run_strat="/GPU:0",
               max_batch=5,
               wait_time=0.000001):
    # support for ANSI cahracters in windows
    support_windows()
    self._model = model
    self._timeout = 120000000

    self._device = utils.get_device(run_strat)

    # steps to take before loading into model
    self._preprocess_fn = preprocess_fn if preprocess_fn is not None else self._pre
    self._process_fn = utils.get_run_fn(model)
    # what you want me to send back
    self._postprocess_fn = postprocess_fn if postprocess_fn is not None else self._post

    self._pdims = process_dims
    self._max_batch = max_batch
    self._dynamic_wt = (wait_time == "dynamic" or wait_time is None)
    if not self._dynamic_wt:
      self._wait_time = utils.get_wait_time(wait_time, max_batch)
    else:
      self._wait_time = 0.001

    self._load_buffer = Queue(maxsize=max_batch)
    self._batched_que = Queue(maxsize=max_batch)
    self._processed_que = Queue(maxsize=max_batch)
    self._return_buffer = Queue(maxsize=max_batch)

    self._running = False
    self._thread = None
    self._clear_thread = None
    self._lbuffer_size = 0
    self._rbuffer_size = 0
    self._lsum = 0
    self._latency = 0
    self._prev_latency = 0
    self._frames = 0
    return

  def _pre(self, frame, pdim):
    return frame

  def _post(self, frame, result):
    return result

  def put(self, raw_frame):
    if self._load_buffer.full():
      return False
    frame = self._preprocess_fn(raw_frame, self._pdims)
    self._load_buffer.put((frame, raw_frame))
    return True

  def _get_batch_input(self, que, max_batch):
    frames = []
    raw = []
    i = 0
    while (i < max_batch and not que.empty()):
      i += 1
      frame, rawframe = que.get()
      frames.append(frame)
      raw.append(rawframe)
    return frames, raw

  def process_frames(self):
    frame_count = 0
    try:
      self._running = True
      while (self._running):
        if self._load_buffer.empty():
          time.sleep(self._wait_time)
          continue
        if self._processed_que.full():
          time.sleep(self._wait_time)
          continue

        start_t = time.time()
        frames = []
        raw = []
        i = 0
        while (i < self._max_batch and not self._load_buffer.empty()):
          i += 1
          frame, rawframe = self._load_buffer.get()
          frames.append(frame)
          raw.append(rawframe)
        rframes = len(raw)
        with tf.device("/GPU:0"):
          frame = tf.convert_to_tensor(frames)
          result = self._process_fn(frame)
        self._processed_que.put((raw, result))
        time.sleep(self._wait_time)
        end_t = time.time()

        if self._frames >= 1000:
          self._frames = 0
          self._lsum = 0
        self._frames += rframes
        self._lsum += (end_t - start_t)
        #self._prev_latency = self._latency * 0.1 + 0.9 * self._prev_latency
        self._latency = self._lsum / self._frames
        if self._dynamic_wt:
          if self._prev_latency >= self._latency:
            self._prev_latency = self._latency * 0.1 + 0.9 * self._prev_latency
            self._wait_time = self._wait_time - 0.02 * self._wait_time
          else:
            self._prev_latency = self._latency * 0.1 + 0.9 * self._prev_latency
            self._wait_time = self._wait_time + 0.01 * self._wait_time

    except KeyboardInterrupt:
      self._running = False
    except Exception as e:
      print(e)
      traceback.print_exc()
      self._running = False
    return

  def postprocess_buffer(self):
    frame_count = 0
    try:
      self._running = True
      while (self._running):
        if self._processed_que.empty():
          time.sleep(self._wait_time)
          continue
        start_t = time.time()
        frames, results = self._processed_que.get()
        ret = self._postprocess_fn(frames, results)
        if not isinstance(ret, dict):
          for frame in ret:
            self._return_buffer.put(frame)
        else:
          self._return_buffer.put((frames, ret))

        end_t = time.time()
        time.sleep(self._wait_time)
    except KeyboardInterrupt:
      self._running = False
    except Exception as e:
      print(e)
      traceback.print_exc()
      self._running = False

  def get(self):
    if self._return_buffer.empty():
      return None
    return self._return_buffer.get()

  def getall(self):
    frames = []
    while not self._return_buffer.empty():
      frames.append(self._return_buffer.get())
    return frames

  def read(self):
    return self._running, self.get()

  def start(self):
    self._thread = t.Thread(target=self.process_frames, args=())
    self._thread.start()
    self._clear_thread = t.Thread(target=self.postprocess_buffer, args=())
    self._clear_thread.start()
    return self._thread

  def close(self):
    self._running = False
    if self._thread is not None:
      self._thread.join()
    if self._clear_thread is not None:
      self._clear_thread.join()
    return

  @property
  def latency(self):
    return self._latency

  @property
  def wait_time(self):
    return self._wait_time

  @wait_time.setter
  def wait_time(self, value):
    self._wait_time = value

  @property
  def running(self):
    return self._running or not self._processed_que.empty()

  @running.setter
  def running(self, value):
    self._running = value

  def full(self):
    return self._return_buffer.full()

  def empty(self):
    return self._load_buffer.empty()

  def __call__(self, frame):
    red = self._preprocess_fn(frame, self._pdims)
    if len(red.shape) == 3:
      red = tf.expand_dims(red, axis=0)
    results = self._process_fn(red)
    return self._postprocess_fn(frame, results)


def preprocess_fn(raw_frame, pdim):
  with tf.device("/GPU:0"):
    image = tf.image.resize(raw_frame, (pdim, pdim))
  return image


def func(inputs):
  boxes = inputs["bbox"]
  classifs = inputs["confidence"]
  nms = tf.image.combined_non_max_suppression(
      tf.expand_dims(boxes, axis=2), classifs, 200, 200, 0.5, 0.5)
  return {
      "bbox": nms.nmsed_boxes,
      "classes": nms.nmsed_classes,
      "confidence": nms.nmsed_scores,
  }


def run(model, video, disp_h, wait_time, max_batch, que_size):
  max_batch = 5 if max_batch is None else max_batch
  pfn = preprocess_fn
  pofn = utils.DrawBoxes(
      classes=80, labels=coco.get_coco_names(), display_names=True, thickness=2)

  server = ModelServer(
      model=model,
      preprocess_fn=pfn,
      postprocess_fn=pofn,
      wait_time=wait_time,
      max_batch=max_batch)
  video = video_t.VideoServer(
      video, wait_time=0.00000001, que=que_size, disp_h=disp_h)
  display = video_t.DisplayThread(
      server, alpha=0.9, wait_time=0.000001, fix_wt=False)
  server.start()
  video.start()
  display.start()

  # issue at soem point there is a
  # bottlenecked by the readeing thread.
  try:
    while (video.running and display.running):
      frame = video.get()
      if not isinstance(frame, type(None)):
        while not server.put(frame):
          time.sleep(server.wait_time)
      time.sleep(server.wait_time)
  except Exception as e:
    print(e)
    traceback.print_exc()

  server.close()
  display.close()


if __name__ == "__main__":
  from yolo.utils.run_utils import prep_gpu
  from yolo.configs import yolo as exp_cfg
  from yolo.tasks.yolo import YoloTask
  import tensorflow_datasets as tfds
  import yolo.utils.export.tensor_rt as trt
  import matplotlib.pyplot as plt

  prep_gpu()

  from tensorflow.keras.mixed_precision import experimental as mixed_precision
  mixed_precision.set_policy("mixed_float16")

  config = exp_cfg.YoloTask(model=exp_cfg.Yolo(base="v4tiny", min_level=4))
  task = YoloTask(config)
  model = task.build_model()
  task.initialize(model)
  model.summary()
  model.predict(tf.ones((1, 416, 416, 3), dtype=tf.float16))

  # #name = "saved_models/v4/tflite-regualr-no-nms"
  # #name = "saved_models/v4/tflite-tiny-no-nms"
  # name = "saved_models/v4/tiny"
  # new_name = f"{name}_tensorrt"
  # model = trt.TensorRT(
  #     saved_model=new_name,
  #     save_new_path=new_name,
  #     max_workspace_size_bytes=4000000000,
  #     max_batch_size=5)  # , precision_mode="INT8", use_calibration=True)
  # model.compile()
  # model.summary()
  # model.set_postprocessor_fn(func)

  run(model, 0, 416, "dynamic", 5, 10000)

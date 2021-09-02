# Lint as: python3
# pylint: skip-file
from absl import app
from absl import flags
import gin
import sys

from official.core import train_utils
# pylint: disable=unused-import
from yolo.common import registry_imports
# pylint: enable=unused-import
from official.common import flags as tfm_flags

from typing import Tuple, List
from official.core import train_utils
from official.modeling import performance
from official.core import task_factory
import os
import tensorflow as tf

from yolo.run import load_model
from yolo.utils.demos import utils, coco
import cv2
import numpy as np
from skimage import io

from yolo.utils.run_utils import prep_gpu
import matplotlib.pyplot as plt
import numpy as np
try:
  prep_gpu()
except BaseException:
  print("GPUs ready")
"""
python3.8 -m yolo.run_image
"""

if __name__ == "__main__":
  task, model, params = load_model(
      experiment="yolo_custom",
      config_path=[
          "yolo/configs/experiments/yolov4-csp/inference/640.yaml"
          # "yolo/configs/experiments/yolov4-csp/debug/640-3x-64.yaml"
      ],
      # config_path=["yolo/configs/experiments/yolov4/inference/512.yaml"],
      # config_path=["yolo/configs/experiments/yolov4-tiny/inference/640.yaml"],
      model_dir="")
  # model_dir='/home/vbanna/Research/checkpoints/yolov4-csp/tpu/512')
  # model_dir='/home/vbanna/Research/checkpoints/250k-512-lr-special-t2')
  optimizer = task.create_optimizer(params.trainer.optimizer_config,
                                    params.runtime)

  train_data = task.build_inputs(params.task.train_data)
  test_data = task.build_inputs(params.task.validation_data)

  # task.initialize(model)
  # print(obj_mask)
  # fig, axe = plt.subplots(1, 2)
  # axe[0].imshow(tf.cast(iou_max[0], tf.float32).numpy())
  # axe[1].imshow(tf.cast(true_conf[0], tf.float32).numpy())
  # if obj_mask.shape[1] == 16:
  #    plt.show()

  lim = 10
  for p, image in enumerate(test_data):
    if p >= lim:
      break
    i, k = image

    # plt.imshow(i[0])
    id = k['source_id'][0]
    # plt.imshow(tf.cast(i[0], tf.float32).numpy())
    a = task.validation_step(image, model)
    # a = task.train_step(image, model, optimizer)

    # pred = model(image)
    # image = tf.image.draw_bounding_boxes(tf.cast(image, tf.flaot32),
    #                                      tf.cast(pred['bbox'], tf.flaot32),
    #                                      [[0, 0, 1]])

    # plt.imshow(image[0])

    print(id.numpy(), a['loss'].numpy())

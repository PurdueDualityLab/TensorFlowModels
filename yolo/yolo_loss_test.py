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
      config_path=["yolo/configs/experiments/yolov4/inference/512-jitter.yaml"],
      model_dir="")
                      
  train_data = task.build_inputs(params.task.train_data)
  test_data = task.build_inputs(params.task.validation_data)
  optimizer = task.create_optimizer(params.trainer.optimizer_config,
                                      params.runtime)

  for image in test_data:
    a = task.train_step(image, model, optimizer)
    print(a)
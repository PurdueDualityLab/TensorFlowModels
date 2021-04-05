import os
import sys
from typing import List, Tuple

import cv2
import gin
import numpy as np
import tensorflow as tf
from absl import app, flags

# pylint: enable=unused-import
from official.common import flags as tfm_flags
from official.core import task_factory, train_utils
from official.modeling import performance
from skimage import io
# pylint: disable=unused-import
from yolo.common import registry_imports
from yolo.run import load_model
from yolo.utils.demos import utils
from yolo.utils.run_utils import prep_gpu

try:
  prep_gpu()
except BaseException:
  print("GPUs ready")
"""
python3.8 -m yolo.run_image
"""


def url_to_image(url):
  image = io.imread(url)
  return image


def resize_input_image(image,
                       shape,
                       normalize=False,
                       expand_dims=True,
                       dtype=np.float32):
  if len(shape) == 4:
    width, height = shape[1], shape[2]
  else:
    width, height = shape[0], shape[1]

  image = cv2.resize(image, (width, height))
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  if normalize and (dtype is not np.uint8 and dtype is not np.int8):
    image = image / 255

  if expand_dims:
    image = np.expand_dims(image.astype(dtype), axis=0)
  return image


if __name__ == "__main__":
  task, model, params = load_model(
      experiment="yolo_custom",
      config_path=["yolo/configs/experiments/yolov3-tiny-eval-autodrone.yaml"],
      model_dir="")
  draw_fn = utils.DrawBoxes(
      classes=params.task.model.num_classes,
      labels=None,
      display_names=False,
      thickness=2)

  image = url_to_image("/home/vbanna/Downloads/images/images/crop.png")
  save_name = "save.png"

  image_ = resize_input_image(image, [416, 416, 3], normalize=True)

  pred = model(image_, training=False)
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  image = cv2.resize(image, (1280, 720))
  image = draw_fn(image / 255, pred)

  cv2.imshow("testframe", image)
  k = cv2.waitKey(0)
  if k == 27:  # wait for ESC key to exit
    cv2.destroyAllWindows()
  elif k == ord("s"):  # wait for 's' key to save and exit
    cv2.imwrite(save_name, image)
    cv2.destroyAllWindows()

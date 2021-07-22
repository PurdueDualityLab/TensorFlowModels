# Lint as: python3
# pylint: skip-file
from absl import app
from absl import flags
import math
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

  image = cv2.resize(image, (height, width))
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  if normalize and (dtype is not np.uint8 and dtype is not np.int8):
    image = image / 255

  if expand_dims:
    image = np.expand_dims(image.astype(dtype), axis=0)
  return image


if __name__ == "__main__":
  task, model, params = load_model(
      experiment="yolo_custom",
      # config_path=["yolo/configs/experiments/yolov4-csp/inference/512-baseline.yaml"],
      config_path=["yolo/configs/experiments/yolov4-csp/inference/512-dbg.yaml"],
      model_dir='')
  draw_fn = utils.DrawBoxes(
      classes=params.task.model.num_classes,
      labels=coco.get_coco_names(
          path="/home/vbanna/Research/TensorFlowModels/yolo/dataloaders/dataset_specs/coco-91.names"
      ),
      thickness=1)


  image = url_to_image("/media/vbanna/DATA_SHARE/CV/datasets/COCO_raw/testing_records/images/139.jpg")
  save_name = "save.png"

  size = params.task.model.input_size
  fit = lambda x: int(math.ceil((x / 32) + 0.5) * 32)
  size = list(image.shape)
  size[0], size[1] = fit(size[0]), fit(size[1])
  image_ = resize_input_image(image, size, normalize=True)

  pred = model(image_, training=False)
  image = cv2.cvtColor(image_[0], cv2.COLOR_BGR2RGB)
  # image = cv2.resize(image, size[:2])
  image = draw_fn(image_[0], pred)

  cv2.imshow("testframe", image)
  k = cv2.waitKey(0)
  if k == 27:  # wait for ESC key to exit
    cv2.destroyAllWindows()
  elif k == ord("s"):  # wait for 's' key to save and exit
    cv2.imwrite(save_name, image)
    cv2.destroyAllWindows()

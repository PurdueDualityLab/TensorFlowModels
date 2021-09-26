# Lint as: python3
# pylint: skip-file

from official import modeling
from yolo.utils.run_utils import prep_gpu
# try:
#
# except BaseException:
#   print("GPUs ready")

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

# from yolo.demos.old import video_detect_gpu as vgu
# from yolo.demos.old import video_detect_cpu as vcu
from yolo.demos import detect
import tensorflow as tf
from skimage import io
import cv2

'''
python3.8 -m yolo.run --experiment=yolo_custom --config_file=yolo/configs/experiments/yolov4-csp/inference/640.yaml --model_dir ../checkpoints/640-baseline-e13/ --file ../../Videos/korea.mp4 --save_file ../../Videos/korea-detect.avi --batch_size 1 --buffer_size 100 
'''

FLAGS = flags.FLAGS


def define_flags():
  """Defines flags."""

  flags.DEFINE_string(
      'experiment', default=None, help='The experiment type registered.')

  flags.DEFINE_string(
      'model_dir',
      default=None,
      help='The directory where the model and training/evaluation summaries'
      'are stored.')

  flags.DEFINE_multi_string(
      'config_file',
      default=None,
      help='YAML/JSON files which specifies overrides. The override order '
      'follows the order of args. Note that each file '
      'can be used as an override template to override the default parameters '
      'specified in Python. If the same parameter is specified in both '
      '`--config_file` and `--params_override`, `config_file` will be used '
      'first, followed by params_override.')

  flags.DEFINE_string(
      'params_override',
      default=None,
      help='a YAML/JSON string or a YAML file which specifies additional '
      'overrides over the default parameters and those specified in '
      '`--config_file`. Note that this is supposed to be used only to override '
      'the model parameters, but not the parameters like TPU specific flags. '
      'One canonical use case of `--config_file` and `--params_override` is '
      'users first define a template config file using `--config_file`, then '
      'use `--params_override` to adjust the minimal set of tuning parameters, '
      'for example setting up different `train_batch_size`. The final override '
      'order of parameters: default_model_params --> params from config_file '
      '--> params in params_override. See also the help message of '
      '`--config_file`.')

  flags.DEFINE_string(
      'tpu',
      default=None,
      help='The Cloud TPU to use for training. This should be either the name '
      'used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 '
      'url.')

  flags.DEFINE_string(
      'tf_data_service', default=None, help='The tf.data service address')

  
  flags.DEFINE_integer('batch_size', default=1, help='preprocess on the gpu')
  flags.DEFINE_integer('buffer_size', default=1, help='preprocess on the gpu')
  flags.DEFINE_integer('resolution', default=720, help='preprocess on the gpu')
  flags.DEFINE_string('save_file', default=None, help='The tf.data service address')
  flags.DEFINE_string('file', default="/dev/video0", help='path to video to run on')
  flags.DEFINE_string('label_file', default="yolo/dataloaders/dataset_specs/coco.names", help='path to video to run on')
  flags.DEFINE_bool('no_display', default=False, help='The experiment type registered.')


def load_model(experiment='yolo_custom', config_path=[], model_dir=''):
  CFG = train_utils.ParseConfigOptions(
      experiment=experiment, config_file=config_path)
  params = train_utils.parse_configuration(CFG)

  if params.runtime.mixed_precision_dtype:
    performance.set_mixed_precision_policy(params.runtime.mixed_precision_dtype,
                                           params.runtime.loss_scale)

  task = task_factory.get_task(params.task, logging_dir=model_dir)
  model = task.build_model()

  if model_dir is not None and model_dir != '':
    optimizer = task.create_optimizer(params.trainer.optimizer_config,
                                      params.runtime)
    # optimizer = tf.keras.mixed_precision.LossScaleOptimizer(tf.keras.optimizers.SGD(), dynamic = True)
    ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
    status = ckpt.restore(tf.train.latest_checkpoint(model_dir))

    # status.expect_partial().assert_existing_objects_matched()
    print(dir(status), status)
  else:
    task.initialize(model)

  return task, model, params


def load_flags(CFG):
  params = train_utils.parse_configuration(CFG)
  model_dir = CFG.model_dir

  if params.runtime.mixed_precision_dtype:
    performance.set_mixed_precision_policy(params.runtime.mixed_precision_dtype,
                                           params.runtime.loss_scale)

  task = task_factory.get_task(params.task, logging_dir=model_dir)
  model = task.build_model()

  if model_dir is not None and model_dir != '':
    optimizer = task.create_optimizer(params.trainer.optimizer_config,
                                      params.runtime)
    # optimizer = tf.keras.mixed_precision.LossScaleOptimizer(tf.keras.optimizers.SGD(), dynamic = True)
    ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
    status = ckpt.restore(tf.train.latest_checkpoint(model_dir))

    # try:
    #   status.expect_partial().assert_existing_objects_matched()
    # except:
    #   print("this checkpoint could not assert all components consumed, componnets may not match")
    print(dir(status), status)
  else:
    task.initialize(model)

  return task, model, params


def url_to_image(url):
  image = io.imread(url)
  return image


def main(_):
  prep_gpu()
  _, model, params = load_flags(FLAGS)
  module = detect.DetectionModule(params, model, FLAGS.label_file)

  valid = [".jpeg", ".jpg", ".png"]
  if any([str.endswith(FLAGS.file, key) for key in valid]): 
    image = url_to_image(FLAGS.file)
    _, image, _ = module.image(image)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    if FLAGS.save_file is not None:
      cv2.imwrite(FLAGS.save_file, image)

    if not FLAGS.no_display:
      cv2.imshow("frame", image)
      k = cv2.waitKey(0)
      if k:
        cv2.destroyAllWindows()
  else:
    module.video(
      file_name=FLAGS.file, 
      save_file=FLAGS.save_file,
      display = not FLAGS.no_display,
      batch_size = FLAGS.batch_size,
      buffer_size = FLAGS.buffer_size,
      output_resolution = FLAGS.resolution,
    )


if __name__ == '__main__':
  define_flags()
  app.run(main)



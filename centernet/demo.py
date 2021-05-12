import tensorflow as tf

from centernet.configs.centernet import CenterNetTask
from centernet.modeling.CenterNet import build_centernet
from centernet.utils.weight_utils.load_weights import (
    get_model_weights_as_dict, load_weights_backbone, load_weights_model)
from official.vision.beta.ops.preprocess_ops import normalize_image
from yolo.demos.video_detect_cpu import runner
from yolo.demos.video_detect_gpu import FastVideo
from yolo.utils.run_utils import prep_gpu

CENTERNET_CKPT_PATH = 'D:\\weights\centernet_hg104_512x512_coco17_tpu-8\checkpoint'
CLIP_PATH = r'D:\\Documents\Research\Software\nyc_demo_fast.mp4'

def preprocess_fn(image, 
                  channel_means=(104.01362025, 114.03422265, 119.9165958), 
                  channel_stds=(73.6027665 , 69.89082075, 70.9150767)):

  image = tf.cast(image, dtype=tf.float32)
  red, green, blue = tf.unstack(image, num=3, axis=3)
  image = tf.stack([blue, green, red], axis=3)
  image = normalize_image(image, offset=channel_means, scale=channel_stds)

  return image


if __name__ == '__main__':
  prep_gpu()
  input_specs = tf.keras.layers.InputSpec(shape=[1, 512, 512, 3])
  config = CenterNetTask()
  
  model, loss = build_centernet(input_specs=input_specs,
      task_config=config, l2_regularization=0)

  weights_dict, _ = get_model_weights_as_dict(CENTERNET_CKPT_PATH)
  load_weights_model(model, weights_dict, 'hourglass104_512', 'detection_2d')

  cap = FastVideo(
      CLIP_PATH, # set to 0 if using webcam
      model=model,
      preprocess_function=preprocess_fn,
      process_width=512,
      process_height=512,
      preprocess_with_gpu=False,
      classes=91,
      print_conf=True,
      max_batch=1,
      disp_h=512,
      scale_que=1,
      wait_time='dynamic')
  cap.run()
  runner(model, 0, 512, 512)

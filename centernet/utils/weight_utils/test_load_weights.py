from yolo.demos.video_detect_gpu import FastVideo
from yolo.demos.video_detect_cpu import runner

from centernet.modeling.CenterNet import build_centernet
from centernet.configs.centernet import CenterNetTask

from centernet.utils.weight_utils.load_weights import get_model_weights_as_dict, load_weights_model

import tensorflow as tf

CKPT_PATH = 'D:\\weights\centernet_hg104_512x512_coco17_tpu-8\checkpoint'

if __name__ == '__main__':
  input_specs = tf.keras.layers.InputSpec(shape=[1, 512, 512, 3])
  config = CenterNetTask()
  
  model, loss = build_centernet(input_specs=input_specs,
      task_config=config, l2_regularization=0)

  weights_dict, n_weights = get_model_weights_as_dict(CKPT_PATH)
  load_weights_model(model, weights_dict, 'hourglass104_512', 'detection_2d')

  cap = FastVideo(
      0,
      model=model,
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
  # runner(model, 0, 512, 512)


from yolo.utils.run_utils import prep_gpu
try:
  prep_gpu()
except:
  print("GPUs ready")
  
from absl import app
from absl import flags
import gin
import sys

from official.core import train_utils
# pylint: disable=unused-import
from yolo.common import registry_imports
# pylint: enable=unused-import
from official.common import flags as tfm_flags


from yolo.demos import video_detect_gpu as vgu
from yolo.demos.three_servers import model_server as msu


'''
python3.8 -m yolo.run --experiment=yolo_custom --out_resolution 416 --config_file=yolo/configs/experiments/yolov4-eval.yaml --video ../videos/nyc.mp4  --max_batch 5
'''

'''
python3.8 -m yolo.run --experiment=yolo_custom --out_resolution 416 --config_file=yolo/configs/experiments/yolov3-eval.yaml --video ../videos/nyc.mp4  --max_batch 9
'''

'''
python3.8 -m yolo.run --experiment=yolo_custom --out_resolution 416 --config_file=yolo/configs/experiments/yolov4-tiny-eval.yaml --video ../videos/nyc.mp4  --max_batch 9
'''

FLAGS = flags.FLAGS

def main(_):
  task, model, params = vgu.load_flags(FLAGS)

  cap = vgu.FastVideo(
      FLAGS.video,
      model=model,
      process_width=FLAGS.process_size,
      process_height=FLAGS.process_size,
      preprocess_with_gpu=FLAGS.preprocess_gpu,
      print_conf=FLAGS.print_conf,
      max_batch=FLAGS.max_batch,
      disp_h=FLAGS.out_resolution,
      scale_que=FLAGS.scale_que,
      wait_time=FLAGS.wait_time)
  cap.run()
 
if __name__ == '__main__':
  import datetime

  a = datetime.datetime.now()
  vgu.define_flags()
  app.run(main)
  b = datetime.datetime.now()


  print("\n\n\n\n\n\n\n {b - a}")
#### ARGUMENT PARSER ####
from yolo.utils.demos import utils
from yolo.utils.demos import coco
from yolo.demos.three_servers import video_server as vs
import tensorflow as tf
import cv2


def print_opt(latency, fps):
  print(
      f"                                \rlatency:, \033[1;32;40m{latency * 1000} \033[0m ms",
      end='\n')
  print(
      '                                 \rfps: \033[1;34;40m%d\033[0m ' % (fps),
      end='\n')
  print('\033[F\033[F\033[F', end='\n')
  return


def process_model(model, process_size):
  drawer = utils.DrawBoxes(classes=80, labels=coco.get_coco_names())

  def run(image):
    image_ = tf.convert_to_tensor(image)
    image_ = tf.image.resize(image_, (process_size, process_size))
    image_ = tf.expand_dims(image_, axis=0)
    pred = model.predict(image_)
    image = drawer(image, pred)
    return image

  return run


def runner(model, file_name, process_size, display_size):
  file_name = 0 if file_name is None else file_name
  try:
    file_name = int(file_name)
  except BaseException:
    print(file_name)
  model_run = process_model(model, process_size)
  video = vs.VideoPlayer(
      file=file_name, post_process=model_run, disp_h=display_size)
  display = vs.DisplayThread(video)
  video.start()
  display.start()
  # while video.running:
  #   image = video.get()
  #   cv2.imshow("frame", image)
  #   if cv2.waitKey(1) & 0xFF == ord("q"):
  #     break


if __name__ == '__main__':
  from yolo import run
  import os
  config = [os.path.abspath('yolo/configs/experiments/yolov4-tiny-eval.yaml')]
  model_dir = ''  # os.path.abspath("../checkpoints/yolo_dt8_norm_iou")
  task, model = run.load_model(
      experiment='yolo_custom', config_path=config, model_dir=model_dir)

  runner(model, 0, 416, 416)

  # display = vs.DisplayThread(video)
  # display.start()

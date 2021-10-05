from numpy.lib.arraysetops import isin
from yolo.serving.utils import video as pyvid
from yolo.serving.utils import drawer as pydraw
from yolo.serving.utils import model_fn as pymodel_run
from yolo.serving.utils import output as pyoutput
from yolo.serving.utils import tflite_model

import tensorflow as tf
from skimage import io

class DetectionModule:
  """Detection for Objects Only."""
  def __init__(self, 
               params,
               model, 
               labels_file):

    labels = pydraw.get_coco_names(path=labels_file)
    colors = pydraw.gen_colors_per_class(len(labels))

    self._undo_infos = True
    if isinstance(model, str) and str.endswith(model, ".tflite"):
      model = tflite_model.TfLiteModel(model_name=model)
      self.model_fn = pyvid.Statistics(model)
      self.scale_boxes = True
    else:
      self.model_fn = pymodel_run.get_wrapped_model(model, params, include_statistics=True, undo_infos=self._undo_infos)
      self.scale_boxes = False
    self.drawer = pydraw.DrawBoxes(labels = labels, colors = colors, thickness=1)

  def video(self, 
          file_name = 0, 
          save_file = None, 
          batch_size = 1, 
          buffer_size = 1, 
          output_resolution = None,
          display = True):

    file_name = 0 if file_name is None else file_name
    try:
      file_name = int(file_name)
    except BaseException:
      print(file_name)

    video = pyvid.VideoFile(
      file_name=file_name, 
      batch_size=batch_size, 
      buffer_size=buffer_size,
      output_resolution=output_resolution, 
      include_statistics=True)

    buffer_size = 1 if buffer_size == 0 else buffer_size
    
    if display:
      display = pyoutput.DisplayThread(frame_buffer_size=buffer_size) 
      display.start()
    
    if save_file is not None:
      saver = pyoutput.SaveVideoThread(save_file=save_file, 
                                       fps = video.file_fps, 
                                       width = video.width, 
                                       height = video.height, 
                                       frame_buffer_size=buffer_size)
      saver.start()

    
    video_stream = video.get_generator()
    print("resolution: {}, {}".format(video.width, video.height))

    # try:
    for frames in video_stream:
      images, preds = self.model_fn(frames)  
      images = self.drawer(images, preds, scale_boxes=self.scale_boxes, stacked = False, include_heatmap = not self._undo_infos)
      if display:
        display.put_all(images)
      
      if save_file is not None:
        saver.put_all(images)
      
      print("read FPS: {}, process FPS: {}, model latency: {:0.3f}ms".format(
        video.fps, self.model_fn.fps * batch_size, self.model_fn.latency * 1000/batch_size
      ), end="\r")

    if display:
      display.close()
    if save_file is not None:
      saver.close()
    # except:
    #   if display:
    #     display.close()
    #   if save_file is not None:
    #     saver.close()

  def image(self, image):
    image = tf.expand_dims(image, axis = 0)
    image, preds = self.model_fn(image)
    overlayed_image = self.drawer(image, 
                                  preds, 
                                  scale_boxes=False, 
                                  stacked = True)

    print("model latency: {:0.3f}ms".format(self.model_fn.latency * 1000))

    preds["bbox"] = preds["bbox"][0]
    preds["classes"] = preds["classes"][0]
    preds["confidence"] = preds["confidence"][0]
    preds["num_detections"] = preds["num_detections"][0]
    return image[0], overlayed_image[0], preds

  def webcam(self, file_name = 0):
    video = pyvid.VideoFile(
      file_name=file_name, 
      batch_size=1, 
      buffer_size=1,
      output_resolution=None, 
      include_statistics=True)

    display = pyoutput.RouteThread(
      width = video.width, 
      height = video.height, 
      frame_buffer_size=1) 
    display.start()

    video_stream = video.get_generator()

    print("resolution: {}, {}".format(video.width, video.height))
    for frames in video_stream:
      images, preds = self.model_fn(frames)  
      images = self.drawer(images, preds, scale_boxes=False, stacked = False)
      display.put_all(images)

      print("read FPS: {}, process FPS: {}, model latency: {:0.3f}ms".format(
        video.fps, self.model_fn.fps, self.model_fn.latency * 1000), end="\r")

def url_to_image(url):
  image = io.imread(url)
  return image

if __name__ == "__main__":
  import os
  from yolo import run
  import matplotlib.pyplot as plt

  # config = [os.path.abspath('yolo/configs/experiments/yolov4-csp/inference/640.yaml')]
  # model_dir = os.path.abspath("../checkpoints/640-baseline-e13")

  # # config = [os.path.abspath('yolo/configs/experiments/yolov4-nano/inference/416-3l.yaml')]
  # # model_dir = os.path.abspath("../checkpoints/416-3l-baseline-e1")

  # # config = [os.path.abspath('yolo/configs/experiments/yolov4/inference/512.yaml')]
  # # model_dir = os.path.abspath("../checkpoints/512-wd-baseline-e1")

  # task, model, params = run.load_model(experiment='yolo_custom', config_path=config, model_dir=model_dir)

  # detect = DetectionModule(
  #   None, 
  #   "/home/vbanna/Research/TensorFlowModels/cache/yolov4_csp", 
  #   'yolo/dataloaders/dataset_specs/coco.names')
  detect = DetectionModule(
    None, 
    "cache/yolov4_csp/assets/yolov4_csp.tflite", 
    'yolo/dataloaders/dataset_specs/coco.names')
  detect.video(
    file_name="../../Videos/soccer.mp4", 
    save_file="../../Videos/soccer-v4.avi",
    display=True,
    batch_size=1,
    buffer_size=1,
    output_resolution = None,
  )

  # detect.webcam(file_name="/dev/video0")

  # image = url_to_image("/home/vbanna/Research/TensorFlowModels/1_EYFejGUjvjPcc4PZTwoufw.jpeg")
  # image, omage, preds = detect.image(image)

  # plt.imshow(omage)
  # plt.show()

  
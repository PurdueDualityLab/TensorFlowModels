import tensorflow as tf
from yolo.utils.run_utils import prep_gpu
from yolo.configs import yolo as exp_cfg
from yolo.tasks.yolo import YoloTask
from skimage import io
import cv2
prep_gpu()

def conversion(model):
  @tf.function
  def run(image):
    with tf.device("cpu:0"):
      image = tf.cast(image, tf.float32)
      image = image/tf.fill(tf.shape(image), 255.0)
      pred = model.call(image, training = False)
      pred = {"bbox": pred["bbox"], "classes":pred["classes"], "confidence":pred["confidence"], "num_dets":pred["num_dets"]}
    return pred
  return run

def url_to_image(url):
	image = io.imread(url)
	return image

with tf.device("gpu:0"):
  model = None
  config = exp_cfg.YoloTask(
      model=exp_cfg.Yolo(base='v4', 
                        min_level=3, 
                        norm_activation = exp_cfg.common.NormActivation(activation="mish"), 
                        #_boxes = ['(10, 14)', '(23, 27)', '(37, 58)', '(81, 82)', '(135, 169)', '(344, 319)'],
                        #_boxes = ['(20, 28)', '(46, 54)', '(74, 116)', '(81, 82)', '(135, 169)', '(344, 319)'],
                        _boxes = ['(12, 16)', '(19, 36)', '(40, 28)', '(36, 75)','(76, 55)', '(72, 146)', '(142, 110)', '(192, 243)','(459, 401)'],
                        filter = exp_cfg.YoloLossLayer(use_nms=False)
                        ))  
  task = YoloTask(config)
  model = task.build_model()
  task.initialize(model)
  model(tf.ones((1, 416, 416, 3), dtype = tf.float32), training = False)

  image = url_to_image("https://raw.githubusercontent.com/zhreshold/mxnet-ssd/master/data/demo/dog.jpg")
  image = cv2.resize(image, (416, 416))
  image = tf.expand_dims(image, axis = 0)
  func = conversion(model)

  converter = tf.lite.TFLiteConverter.from_concrete_functions([func.get_concrete_function(image)])
  converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_LATENCY]
  #converter.target_spec.supported_types = [tf.float16]

  try:
    tflite_model = converter.convert()
  except:
    print("here")
    import sys
    sys.exit()
  
  # Save the model.
  with open('detect-large.tflite', 'wb') as f:
    f.write(tflite_model)




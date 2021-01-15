import tensorflow as tf
from yolo.utils.run_utils import prep_gpu
from yolo.configs import yolo as exp_cfg
from yolo.tasks.yolo import YoloTask
from skimage import io
import cv2
prep_gpu()


def conversion_top(model):

  @tf.function
  def run(image):
    with tf.device("cpu:0"):
      image = tf.cast(image, tf.float32)
      image = image / tf.fill(tf.shape(image), 255.0)
    pred = model.call(image, training=True)
    return pred["raw_output"]

  return run


def filter_3(model, reverse=False):
  model = model.filter

  @tf.function
  def run(in1, in2, in3):
    with tf.device("cpu:0"):
      inputs = {"5": in1, "4": in2, "3": in3}
      pred = model.call(inputs)
      pred = {
          "bbox": tf.cast(pred["bbox"], tf.float32),
          "classes": tf.cast(pred["classes"], tf.float32),
          "confidence": tf.cast(pred["confidence"], tf.float32),
          "num_dets": tf.cast(pred["num_dets"], tf.float32)
      }
    return pred

  @tf.function
  def run_reverse(in1, in2, in3):
    with tf.device("cpu:0"):
      inputs = {"3": in1, "4": in2, "5": in3}
      pred = model.call(inputs)
      pred = {
          "bbox": tf.cast(pred["bbox"], tf.float32),
          "classes": tf.cast(pred["classes"], tf.float32),
          "confidence": tf.cast(pred["confidence"], tf.float32),
          "num_dets": tf.cast(pred["num_dets"], tf.float32)
      }
    return pred

  if reverse:
    return run_reverse
  return run


def filter_2():
  model = model.filter

  @tf.function
  def run(in1, in2):
    with tf.device("cpu:0"):
      inputs = {"5": in1, "4": in2}
      pred = model.call(inputs)
      pred = {
          "bbox": tf.cast(pred["bbox"], tf.float32),
          "classes": tf.cast(pred["classes"], tf.float32),
          "confidence": tf.cast(pred["confidence"], tf.float32),
          "num_dets": tf.cast(pred["num_dets"], tf.float32)
      }
    return pred

  return run


def url_to_image(url):
  image = io.imread(url)
  return image


def get_model(model, input_size=[416, 416, 3], training=True):
  with tf.device("gpu:0"):
    if model == "v4tiny":
      config = exp_cfg.YoloTask(
          model=exp_cfg.Yolo(
              _input_size=input_size,
              base="v4tiny",
              min_level=4,
              norm_activation=exp_cfg.common.NormActivation(activation="leaky"),
              _boxes=[
                  "(10, 14)", "(23, 27)", "(37, 58)", "(81, 82)", "(135, 169)",
                  "(344, 319)"
              ],
              #_boxes = ['(20, 28)', '(46, 54)', '(74, 116)', '(81, 82)', '(135, 169)', '(344, 319)'],
              #_boxes = ["(10, 13)", "(16, 30)", "(33, 23)","(30, 61)", "(62, 45)", "(59, 119)","(116, 90)", "(156, 198)", "(373, 326)"],
              #_boxes = ['(12, 16)', '(19, 36)', '(40, 28)', '(36, 75)','(76, 55)', '(72, 146)', '(142, 110)', '(192, 243)','(459, 401)'],
              filter=exp_cfg.YoloLossLayer(use_nms=False)))
      name = f"detect-{input_size[0]}.tflite"
    else:
      config = exp_cfg.YoloTask(
          model=exp_cfg.Yolo(
              _input_size=input_size,
              base="v3",
              min_level=3,
              norm_activation=exp_cfg.common.NormActivation(activation="leaky"),
              _boxes=[
                  "(10, 13)", "(16, 30)", "(33, 23)", "(30, 61)", "(62, 45)",
                  "(59, 119)", "(116, 90)", "(156, 198)", "(373, 326)"
              ],
              filter=exp_cfg.YoloLossLayer(use_nms=False)))
      name = f"detect-large-{input_size[0]}.tflite"

    task = YoloTask(config)
    model = task.build_model()
    task.initialize(model)
    model(tf.ones((1, *input_size), dtype=tf.float32), training=False)
    return model, name


def convert_tiny():
  model, name = get_model(model="v4tiny", input_size=[416, 416, 3])

  image = url_to_image(
      "https://raw.githubusercontent.com/zhreshold/mxnet-ssd/master/data/demo/dog.jpg"
  )
  image = cv2.resize(image, (416, 416))
  image = tf.expand_dims(image, axis=0)

  runner = conversion_top(model)
  brunner = filter_2(model)

  converter = tf.lite.TFLiteConverter.from_concrete_functions(
      [runner.get_concrete_function(image)])
  converter.optimizations = [tf.lite.Optimize.DEFAULT]

  try:
    tflite_model = converter.convert()
  except BaseException:
    print("here")
    # st.print_exc()
    import sys
    sys.exit()

  with open(name, "wb") as f:
    f.write(tflite_model)

  pred = runner(image)
  print(pred.keys())

  inputs = []
  for key in pred.keys():
    inputs.append(pred[key].numpy())
  print(inputs)

  converter = tf.lite.TFLiteConverter.from_concrete_functions(
      [brunner.get_concrete_function(*inputs)])
  converter.optimizations = [tf.lite.Optimize.DEFAULT]

  try:
    tflite_model = converter.convert()
  except BaseException:
    print("here")
    # st.print_exc()
    import sys
    sys.exit()

  with open("detection_filter_2.tflite", "wb") as f:
    f.write(tflite_model)

  a = brunner(*inputs)
  print(a.keys())

  return


def convert_large(model="v3", input_size=[416, 416, 3]):
  model, name = get_model(model=model, input_size=input_size)

  image = url_to_image(
      "https://raw.githubusercontent.com/zhreshold/mxnet-ssd/master/data/demo/dog.jpg"
  )
  image = cv2.resize(image, tuple(input_size[:2]))
  image = tf.expand_dims(image, axis=0)

  runner = conversion_top(model)
  if model == "v3":
    brunner = filter_3(model, reverse=False)
  else:
    brunner = filter_3(model, reverse=True)

  converter = tf.lite.TFLiteConverter.from_concrete_functions(
      [runner.get_concrete_function(image)])
  converter.optimizations = [tf.lite.Optimize.DEFAULT]

  try:
    tflite_model = converter.convert()
  except BaseException:
    import sys
    print("here")
    sys.exit()

  with open(name, "wb") as f:
    f.write(tflite_model)

  pred = runner(image)
  print(pred.keys())

  inputs = []
  for key in pred.keys():
    inputs.append(pred[key].numpy())
  print(inputs)

  converter = tf.lite.TFLiteConverter.from_concrete_functions(
      [brunner.get_concrete_function(*inputs)])
  converter.optimizations = [tf.lite.Optimize.DEFAULT]

  try:
    tflite_model = converter.convert()
  except BaseException:
    print("here")
    # st.print_exc()
    import sys
    sys.exit()

  with open(f"detect-filter3-{input_size[0]}.tflite", "wb") as f:
    f.write(tflite_model)

  a = brunner(*inputs)
  print(a.keys())

  return


if __name__ == "__main__":
  convert_large()

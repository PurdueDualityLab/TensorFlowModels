import numpy as np
import tensorflow as tf
from skimage import io
import cv2
from yolo.serving.utils import drawer


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
  if normalize and (dtype is not np.uint8 and dtype is not np.int8):
    image = image / 255

  if expand_dims:
    image = np.expand_dims(image.astype(dtype), axis=0)
  return image

def print_mod(model_name="detect.tflite"):
  interpreter = tf.lite.Interpreter(model_path=model_name)
  interpreter.allocate_tensors()

  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()
  details = interpreter.get_tensor_details()

  for i in input_details:
    print(i)

  print()
  for i in output_details:
    print(i)

  print(dir(interpreter))
  print()
  for i in details:
    if "max" in i["name"] or "Max" in i["name"]:
      print(i)

class TfLiteModel:

  def __init__(self, model_name="cache/yolov4_csp/assets/yolov4_csp.tflite"):
    self._interpreter = tf.lite.Interpreter(model_path=model_name)
    self._interpreter.allocate_tensors()

    self._input_details = self._interpreter.get_input_details()
    self._output_details = self._interpreter.get_output_details()

    for i in self._input_details:
      print(i)

    print()
    for i in self._output_details:
      print(i)

    self._input_shape = self._input_details[0]["shape"]

  def __call__(self, image_, device = "cpu:0"):
    image = tf.image.resize(image_, self._input_shape[1:-1], preserve_aspect_ratio=True)
    image = tf.image.pad_to_bounding_box(image, 0, 0, self._input_shape[1], self._input_shape[2])
    if len(image.shape) != 4:
      image = tf.expand_dims(image, axis = 0)
    input_data = tf.cast(image, self._input_details[0]["dtype"])

    with tf.device(device):
      self._interpreter.set_tensor(self._input_details[0]["index"], input_data)
      self._interpreter.invoke()

    num_dets = self._interpreter.get_tensor(self._output_details[0]["index"])
    boxes = self._interpreter.get_tensor(self._output_details[1]["index"])
    confidences = self._interpreter.get_tensor(self._output_details[2]["index"])
    classes = self._interpreter.get_tensor(self._output_details[3]["index"])
    pred = {"bbox": boxes, "classes": classes, "confidence": confidences}
    return image/255, pred





if __name__ == "__main__":
  image = url_to_image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRqJDwNw4ZvRlzd49BNiV-hAkWGkZ2VXpk4sQ&usqp=CAU")
  model = TfLiteModel(model_name="cache/yolov4_csp/assets/yolov4_csp.tflite")

  model(image)
  # print_mod(model_name="detect.tflite")

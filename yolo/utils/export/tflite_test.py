import numpy as np
import tensorflow as tf
from skimage import io
import cv2
# from yolo.utils.run_utils import prep_gpu
# prep_gpu()
# from yolo.modeling.layers.detection_generator import YoloLayer as filters
from yolo.utils.demos import utils


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


def TfLiteModel(image, model_name="detect.tflite"):
  draw_fn = utils.DrawBoxes(
      classes=80, labels=None, display_names=False, thickness=2)

  interpreter = tf.lite.Interpreter(model_path=model_name)
  interpreter.allocate_tensors()

  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()

  for i in input_details:
    print(i)

  print()
  for i in output_details:
    print(i)

  input_shape = input_details[0]["shape"]
  input_data = resize_input_image(
      image, input_shape, normalize=True, dtype=input_details[0]["dtype"])
  interpreter.set_tensor(input_details[0]["index"], input_data)
  interpreter.invoke()

  boxes = interpreter.get_tensor(output_details[0]["index"])
  classes = interpreter.get_tensor(output_details[1]["index"])
  confidences = interpreter.get_tensor(output_details[2]["index"])
  num_dets = interpreter.get_tensor(output_details[3]["index"])
  pred = {"bbox": boxes, "classes": classes, "confidence": confidences}

  print(num_dets)
  pimage = draw_fn(image, pred)
  cv2.imshow("testframe", pimage)
  k = cv2.waitKey(0)
  if k == 27:  # wait for ESC key to exit
    cv2.destroyAllWindows()
  elif k == ord("s"):  # wait for 's' key to save and exit
    cv2.imwrite("messigray.png", pimage)
    cv2.destroyAllWindows()
  return


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


if __name__ == "__main__":
  image = url_to_image(
      "https://raw.githubusercontent.com/zhreshold/mxnet-ssd/master/data/demo/dog.jpg"
  )
  TfLiteModel(image, model_name="detect.tflite")
  # print_mod(model_name="detect.tflite")

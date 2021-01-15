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


def TfLiteModel(image,
                model_base="detect-416.tflite",
                model_filter="detect-tiny-filter-416.tflite"):
  draw_fn = utils.DrawBoxes(
      classes=80, labels=None, display_names=False, thickness=2)

  process = tf.lite.Interpreter(model_path=model_base)
  filters = tf.lite.Interpreter(model_path=model_filter)

  process.allocate_tensors()
  filters.allocate_tensors()

  input_details = process.get_input_details()
  inter_out_details = process.get_output_details()

  inter_in_details = filters.get_input_details()
  output_details = filters.get_output_details()

  # print(input_details)
  # print(inter_in_details)
  # print(inter_out_details)
  # print(output_details)

  input_shape = input_details[0]["shape"]
  input_data = resize_input_image(
      image, input_shape, normalize=True, dtype=input_details[0]["dtype"])

  process.set_tensor(input_details[0]["index"], input_data)
  process.invoke()
  inter1 = process.get_tensor(inter_out_details[0]["index"])
  inter2 = process.get_tensor(inter_out_details[1]["index"])

  filters.set_tensor(inter_in_details[0]["index"], inter2)
  filters.set_tensor(inter_in_details[1]["index"], inter1)
  filters.invoke()
  boxes = filters.get_tensor(output_details[0]["index"])
  classes = filters.get_tensor(output_details[1]["index"])
  confidences = filters.get_tensor(output_details[2]["index"])
  num_dets = filters.get_tensor(output_details[3]["index"])

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


def TfLiteModelLarge(image,
                     model_base="detect-large-416.tflite",
                     model_filter="detect-filter3-416.tflite"):
  draw_fn = utils.DrawBoxes(
      classes=80, labels=None, display_names=False, thickness=2)

  process = tf.lite.Interpreter(model_path=model_base)
  filters = tf.lite.Interpreter(model_path=model_filter)

  process.allocate_tensors()
  filters.allocate_tensors()

  input_details = process.get_input_details()
  inter_out_details = process.get_output_details()

  inter_in_details = filters.get_input_details()
  output_details = filters.get_output_details()

  # print(input_details)
  # print(inter_in_details)
  # print(inter_out_details)
  # print(output_details)

  input_shape = input_details[0]["shape"]
  input_data = resize_input_image(
      image, input_shape, normalize=True, dtype=input_details[0]["dtype"])

  process.set_tensor(input_details[0]["index"], input_data)
  process.invoke()
  inter1 = process.get_tensor(inter_out_details[0]["index"])
  inter2 = process.get_tensor(inter_out_details[1]["index"])
  inter3 = process.get_tensor(inter_out_details[2]["index"])

  filters.set_tensor(inter_in_details[0]["index"], inter3)
  filters.set_tensor(inter_in_details[1]["index"], inter2)
  filters.set_tensor(inter_in_details[2]["index"], inter1)
  filters.invoke()
  boxes = filters.get_tensor(output_details[0]["index"])
  classes = filters.get_tensor(output_details[1]["index"])
  confidences = filters.get_tensor(output_details[2]["index"])
  num_dets = filters.get_tensor(output_details[3]["index"])

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


if __name__ == "__main__":
  image = url_to_image(
      "https://raw.githubusercontent.com/zhreshold/mxnet-ssd/master/data/demo/dog.jpg"
  )
  # TfLiteModel(image)
  TfLiteModelLarge(image)
  # print_mod(model_name="detect.tflite")

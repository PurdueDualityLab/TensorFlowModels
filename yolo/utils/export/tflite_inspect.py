import tensorflow as tf
from skimage import io
# from yolo.utils.run_utils import prep_gpu
# prep_gpu()
# from yolo.modeling.layers.detection_generator import YoloLayer as filters


def url_to_image(url):
  image = io.imread(url)
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
    print(i["name"], i["shape"], i["shape_signature"])


if __name__ == "__main__":
  image = url_to_image(
      "https://raw.githubusercontent.com/zhreshold/mxnet-ssd/master/data/demo/dog.jpg"
  )
  print_mod(model_name="detect.tflite")

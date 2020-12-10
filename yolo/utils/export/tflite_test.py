import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from skimage import io
import cv2
# from yolo.utils.run_utils import prep_gpu
# prep_gpu()
# from yolo.modeling.layers.detection_generator import YoloLayer as filters
from yolo.utils.demos import utils

def url_to_image(url):
	image = io.imread(url)
	return image

image = url_to_image("https://raw.githubusercontent.com/zhreshold/mxnet-ssd/master/data/demo/dog.jpg")

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="model-nopad.tflite")
# a = filters({"4":[0, 1, 2], "5":[3, 4, 5]}, [[10,14], [23,27], [37,58], [81,82], [135,169], [344,319]], 80, path_scale={"4": 2**4, "5": 2**5})
draw_fn = utils.DrawBoxes(classes=80, labels=None, display_names=False, thickness=2) 

print(interpreter)
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print(input_details, output_details)

# Test the model on random input data.
input_shape = input_details[0]['shape']
input_data = cv2.resize(image, (input_shape[1], input_shape[2]))
input_data = input_data/255
input_data = np.expand_dims(input_data.astype(np.float32), axis=0)
print(input_data.shape)

interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()

# # The function `get_tensor()` returns a copy of the tensor data.
# # Use `tensor()` in order to get a pointer to the tensor.
output_data1 = interpreter.get_tensor(output_details[0]['index'])
output_data2 = interpreter.get_tensor(output_details[1]['index'])
output_data3 = interpreter.get_tensor(output_details[2]['index'])
print(output_data1.shape, output_data2.shape, output_data3.shape)
pred = {"bbox": output_data1, "classes": output_data2, "confidence": output_data3}

pimage = draw_fn(image/255, pred)

fig, (ax1, ax2) = plt.subplots(1,2)
fig.set_figwidth(100)
fig.set_figheight(100)
ax1.imshow(image)
ax2.imshow(pimage)
plt.show()

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

def resize_input_image(image, shape, normalize = False, expand_dims = True, dtype = np.float32):
	if len(shape) == 4:
		width, height = shape[1], shape[2]
	else:
		width, height = shape[0], shape[1]

	image = cv2.resize(image, (width, height))
	if normalize and (dtype is not np.uint8 and dtype is not np.int8): 
		image = image/255

	if expand_dims:
		image = np.expand_dims(image.astype(dtype), axis=0)
	return image

# Load the TFLite model and allocate tensors.
def yololite(image):
	draw_fn = utils.DrawBoxes(classes=80, labels=None, display_names=False, thickness=2) 
	interpreter = tf.lite.Interpreter(model_path="model-nopad.tflite")
	print(interpreter)
	interpreter.allocate_tensors()

	# Get input and output tensors.
	input_details = interpreter.get_input_details()
	output_details = interpreter.get_output_details()
	print(input_details, output_details)

	# Test the model on random input data.
	input_shape = input_details[0]['shape']
	input_data = resize_input_image(image, input_shape, normalize=True)
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

def TfLiteModel(image):
	draw_fn = utils.DrawBoxes(classes=80, labels=None, display_names=False, thickness=2) 
	
	interpreter = tf.lite.Interpreter(model_path="detect.tflite")
	interpreter.allocate_tensors()

	input_details = interpreter.get_input_details()
	output_details = interpreter.get_output_details()

	for i in input_details:
		print(i)
	
	print()
	for i in output_details:
		print(i)

	input_shape = input_details[0]['shape']
	input_data = resize_input_image(image, input_shape, normalize=True, dtype=input_details[0]['dtype'])
	interpreter.set_tensor(input_details[0]['index'], input_data)
	interpreter.invoke()

	boxes = interpreter.get_tensor(output_details[0]['index'])
	classes = interpreter.get_tensor(output_details[1]['index'])
	confidences = interpreter.get_tensor(output_details[2]['index'])
	pred = {"bbox": boxes, "classes": classes, "confidence": confidences}

	pimage = draw_fn(image, pred)
	cv2.imshow("testframe", pimage)
	k = cv2.waitKey(0)
	if k == 27:         # wait for ESC key to exit
		cv2.destroyAllWindows()
	elif k == ord('s'): # wait for 's' key to save and exit
		cv2.imwrite('messigray.png',pimage)
		cv2.destroyAllWindows()
	return 

if __name__ == "__main__":
	image = url_to_image("https://raw.githubusercontent.com/zhreshold/mxnet-ssd/master/data/demo/dog.jpg")
	TfLiteModel(image)
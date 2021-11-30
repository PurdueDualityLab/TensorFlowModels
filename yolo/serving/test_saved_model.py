import tensorflow as tf
from skimage import io
import numpy as np
import cv2
from yolo.serving.utils import drawer as pydraw
from yolo.serving.utils.model_fn import scale_boxes
import pickle

def url_to_tensor(url):
  image = io.imread(url)
  image = tf.expand_dims(image, axis = 0)
  return image

imported = tf.saved_model.load('export_dir')
f = imported.signatures['serving_default']
labels = pydraw.get_coco_names()
colors = pydraw.gen_colors_per_class(len(labels))
drawer = pydraw.DrawBoxes(labels = labels, colors = colors, thickness = 1)

image = url_to_tensor('https://github.com/pjreddie/darknet/blob/master/data/dog.jpg?raw=true')
out = f(image)

out['bbox'] = out['detection_boxes']
out['confidence'] = out['detection_scores']
out['classes'] = out['detection_classes']

del out['detection_boxes']
del out['detection_scores']
del out['detection_classes']
with open('out.pkl', 'wb') as file:
  pickle.dump(out, file, protocol=pickle.HIGHEST_PROTOCOL)


new_img = drawer(image, out, scale_boxes = True, stacked = True)
new_img = cv2.cvtColor(new_img[0], cv2.COLOR_BGR2RGB)
cv2.imshow('frame', new_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


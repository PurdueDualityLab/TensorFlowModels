import tensorflow as tf
from skimage import io
import numpy as np
def url_to_image(url):
  image = io.imread(url)
  return image

imported = tf.saved_model.load('export_dir')
print(list(imported.signatures.keys()))
f = imported.signatures['serving_default']
print(f.structured_outputs)
image = url_to_image('https://github.com/pjreddie/darknet/blob/master/data/dog.jpg?raw=true')
print(f(tf.convert_to_tensor(image)))

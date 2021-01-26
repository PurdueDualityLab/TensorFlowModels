import tensorflow_datasets as tfds 
import tensorflow as tf
import matplotlib.pyplot as plt

def get_id(image_mask):
  image_mask = tf.cast(image_mask, tf.int64)
  id = image_mask[..., 0] + image_mask[..., 1] * 256 + image_mask[..., 2] * 256 ** 2
  return id
  #return image_mask[..., 0]

def get_class(id_mask):
  return 


path = "/media/vbanna/DATA_SHARE/tfds"
dataset = "coco/2017_panoptic"
val = tfds.load(dataset, data_dir = path, split = "validation")
print(val)

lim = 10


for i, sample in enumerate(val):
  id_mask = get_id(sample["panoptic_image"])
  ids = sample["panoptic_objects"]['id']  
  classes = sample["panoptic_objects"]['label']  

  ids = tf.expand_dims(ids, axis = 0)
  ids = tf.expand_dims(ids, axis = 0)
  ids = tf.repeat(ids, tf.shape(id_mask)[0], axis = 0)
  ids = tf.repeat(ids, tf.shape(id_mask)[1], axis = 1)

  classes = tf.expand_dims(classes, axis = 0)
  classes = tf.expand_dims(classes, axis = 0)
  classes = tf.repeat(classes, tf.shape(id_mask)[0], axis = 0)
  classes = tf.repeat(classes, tf.shape(id_mask)[1], axis = 1)

  bool_mask = tf.cast(tf.math.equal(tf.expand_dims(id_mask, axis = -1), ids), tf.int64)

  class_mask = tf.reduce_sum(bool_mask * classes, axis = -1)

  fig, axe = plt.subplots(1, 3)
  axe[0].imshow(class_mask)
  axe[1].imshow(id_mask % 256)
  axe[2].imshow(sample["image"])
  plt.show()
  if i > (lim + 1):
    break
from inspect import stack
import tensorflow as tf 
from yolo.ops import preprocessing_ops
from official.vision.beta.ops import box_ops
from yolo.serving.utils import video as pyvid

def letterbox(image, desired_size, letter_box = True):
  """Letter box an image for image serving."""

  with tf.name_scope('letter_box'):
    image_size = tf.cast(preprocessing_ops.get_image_shape(image), tf.float32)

    scaled_size = tf.cast(desired_size, image_size.dtype)
    if letter_box:
      scale = tf.minimum(
          scaled_size[0] / image_size[0], scaled_size[1] / image_size[1])
      scaled_size = tf.round(image_size * scale)
    else:
      scale = 1.0

    # Computes 2D image_scale.
    image_scale = scaled_size / image_size
    image_offset = tf.cast((desired_size - scaled_size) * 0.5, tf.int32)
    offset = (scaled_size - desired_size) * 0.5
    scaled_image = tf.image.resize(image, 
                        tf.cast(scaled_size, tf.int32), 
                        method = 'nearest')

    output_image = tf.image.pad_to_bounding_box(
        scaled_image, image_offset[0], image_offset[1], 
                      desired_size[0], desired_size[1])

    
    image_info = tf.stack([
        image_size,
        tf.cast(desired_size, dtype=tf.float32),
        image_scale,
        tf.cast(offset, tf.float32)])
    return output_image, image_info

def undo_info(boxes, num_detections, info, expand = True):
  mask = tf.sequence_mask(num_detections, maxlen=tf.shape(boxes)[1])
  boxes = tf.cast(tf.expand_dims(mask, axis = -1), boxes.dtype) * boxes

  if expand:
    info = tf.cast(tf.expand_dims(info, axis = 0), boxes.dtype)
  inshape = tf.expand_dims(info[:, 1, :], axis = 1)
  ogshape = tf.expand_dims(info[:, 0, :], axis = 1)
  scale = tf.expand_dims(info[:, 2, :], axis = 1)
  offset = tf.expand_dims(info[:, 3, :], axis = 1)

  boxes = box_ops.denormalize_boxes(boxes, inshape)
  boxes += tf.tile(offset, [1, 1, 2])
  boxes /= tf.tile(scale, [1, 1, 2])
  boxes = box_ops.clip_boxes(boxes, ogshape)
  boxes = box_ops.normalize_boxes(boxes, ogshape)  
  return boxes

def scale_boxes(boxes, classes, image):
  height, width = preprocessing_ops.get_image_shape(image)

  height = tf.cast(height, boxes.dtype)
  width = tf.cast(width, boxes.dtype)
  boxes = tf.stack([
      tf.cast(boxes[..., 0] * height, dtype=tf.int32),
      tf.cast(boxes[..., 1] * width, dtype=tf.int32),
      tf.cast(boxes[..., 2] * height, dtype=tf.int32),
      tf.cast(boxes[..., 3] * width, dtype=tf.int32)],axis=-1)
  classes = tf.cast(classes, dtype=tf.int32)
  return boxes, classes

def get_wrapped_model(model, params, include_statistics = False, undo_infos = True):
  size = params.task.model.input_size[:2]
  letter_box = not params.task.model.darknet_based_model
  dtype = params.runtime.mixed_precision_dtype

  #fuse the model
  if hasattr(model, "fuse"):
    model.fuse()

  # build the model
  ones = tf.ones([1]+size+[3], dtype = dtype)  
  _ = model(ones)

  @tf.function
  def run(image):
    pimage, info = letterbox(image, size, letter_box)
    pimage = tf.cast(pimage, dtype)/255.0
    predictions = model(pimage, training=False)

    if undo_infos:
      predictions["bbox"] = undo_info(
        predictions["bbox"], predictions["num_detections"], info)

      (predictions["bbox"], 
      predictions["classes"]) = scale_boxes(predictions["bbox"], 
                                            predictions["classes"], 
                                            image)
    return image, predictions 

  if include_statistics:
    run = pyvid.Statistics(run)
  return run

def get_saved_model(model_path, include_statistics = False):
  model = tf.saved_model.load(model_path, tags = ["serve"])
  graph = model.signatures["serving_default"]
  @tf.function
  def run(image):
    images = tf.unstack(image)
    return image, graph(model, images)

  if include_statistics:
    run = pyvid.Statistics(run)
  return run
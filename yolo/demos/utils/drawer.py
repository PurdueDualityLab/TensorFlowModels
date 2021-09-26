import tensorflow as tf
import numpy as np
import colorsys
import cv2
from yolo.ops import preprocessing_ops

def gen_colors_per_class(max_classes):
  hue = np.linspace(start=0, stop=1, num=max_classes)
  np.random.shuffle(hue)
  colors = []
  for val in hue:
    colors.append(np.array(colorsys.hsv_to_rgb(val, 0.75, 1.0)) * 255)
  return colors

def get_coco_names(path="yolo/dataloaders/dataset_specs/coco.names"):
  with open(path, "r") as f:
    data = f.readlines()
  for i in range(len(data)):
    data[i] = data[i].strip()
  return data

class DrawBoxes(object):

  def __init__(self, labels=None, colors = None, thickness=2):
    self._labels = labels
    self._thickness = thickness
    self._colors = colors
    self._font = cv2.FONT_HERSHEY_SIMPLEX
    return

  def _scale_boxes(self, boxes, classes, image):
    height, width = preprocessing_ops.get_image_shape(image)
    boxes = tf.stack([
        tf.cast(boxes[..., 0] * height, dtype=tf.int32),
        tf.cast(boxes[..., 1] * width, dtype=tf.int32),
        tf.cast(boxes[..., 2] * height, dtype=tf.int32),
        tf.cast(boxes[..., 3] * width, dtype=tf.int32)],axis=-1)
    classes = tf.cast(classes, dtype=tf.int32)
    return boxes, classes

  def _get_text(self, classification, conf):
    if self._labels is None and conf is None: 
      return ''
    elif conf is None:
      return '{}'.format(self._labels[classification])
    elif self._labels is None: 
      return '{:.1f}%'.format(conf * 100)  
    return '{}:{:.1f}%'.format(self._labels[classification], conf * 100)

  def _draw(self, image, box, classes, conf):
    if box[1] - box[0] == 0 or box[3] - box[2] == 0:
      return False
    
    x0, y0 = int(box[1]), int(box[0])
    x1, y1 = int(box[3]), int(box[2])   
    classes = int(classes)

    color = self._colors[classes]
    text = self._get_text(classes, conf)
    txt_bk_color = (np.array(color) * 0.7).tolist()
    txt_color = (0, 0, 0) if np.mean(color/255) > 0.5 else (255, 255, 255)
    txt_size = cv2.getTextSize(text, self._font, 0.4, 1)[0]

    if isinstance(image.item, np.uint8) or isinstance(image.item, np.int32):
      txt_bk_color = (np.array(txt_bk_color)/255).tolist()
      txt_color = (np.array(txt_color)/255).tolist()
      color = (np.array(color)/255).tolist()

    if conf is None:
      x = (box[1] + box[0]) // 2
      y = (box[3] + box[2]) // 2
      cv2.circle(image, (x, y), radius=0, color=color, thickness=self._thickness * 3)
    cv2.rectangle(image, (x0, y0), (x1, y1), color, self._thickness)

    cv2.rectangle(image, (x0, y0 + 1), (x0 + txt_size[0] + 1, y0 + int(1.5*txt_size[1])), txt_bk_color, -1)
    cv2.putText(image, text, (x0, y0 + txt_size[1]), self._font, 0.4, txt_color, thickness=1)
    return True
  
  def __call__(self, images, results, scale_boxes = True, stacked = True):

    if hasattr(images, "numpy"):
      images = images.numpy()

    boxes = tf.convert_to_tensor(results["bbox"])
    classes = tf.convert_to_tensor(results["classes"])

    if scale_boxes:
      boxes, classes = self._scale_boxes(boxes, classes, images)

    if "confidence" in results: 
      conf = results["confidence"]
      if conf is not None:
        conf = tf.convert_to_tensor(results["confidence"])
      else:
        conf = [None] * classes.shape[0] if len(images.shape) != 3 else 1
    else:
      conf = tf.convert_to_tensor(results["classes"])/100

    boxes = boxes.numpy()
    classes = classes.numpy()
    if hasattr(conf, "numpy"):
      conf = conf.numpy()

    if len(images.shape) == 3: 
      if len(boxes.shape) == 2:
        self._draw(images, boxes, classes, conf)
      else:
        self._draw(images, boxes[0], classes[0], conf[0])
      return images
    else:
      image = []
      for i, im in enumerate(images):
        for j in range(boxes[i].shape[0]):
          if not self._draw(im, boxes[i][j], classes[i][j], conf[i][j]):
            break
        image.append(im)
      if stacked:
        image = np.stack(images, axis=0)
      return image 
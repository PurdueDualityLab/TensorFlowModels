import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from utils.demos import utils

"""{image: (None, None, 3), 
    image/filename: (), 
    image/id: (), 
    panoptic_image: (None, None, 3), 
    panoptic_image/filename: (), 
    panoptic_objects: 
      {area: (None,), 
       bbox: (None, 4), 
       id: (None,), 
       is_crowd: (None,), 
       label: (None,)}
    }, 
    types: 
      {image: tf.uint8,
       image/filename: tf.string, 
       image/id: tf.int64, 
       panoptic_image: tf.uint8, 
       panoptic_image/filename: tf.string, 
       panoptic_objects: 
         {area: tf.int64, 
          bbox: tf.float32, 
          id: tf.int64, 
          is_crowd: tf.bool, 
          label: tf.int64}
      }
    }
"""

class PanopticDraw(object):
  def __init__(self, things_classes = None, stuff_classes = None, display_names = False, thickness = 2):
    if things_classes is not None:
      self._num_classes = len(things_classes)
    else:
      self._num_classes = 80
    
    if stuff_classes is not None:
      self._num_classes += len(stuff_classes)
    else:
      self._num_classes += 80
    
    if things_classes is not None and stuff_classes is not None:
      self._labels = things_classes + stuff_classes
    else:
      self._labels = None

    self._drawer = utils.DrawBoxes(classes = self._num_classes, labels = self._labels, display_names = display_names, thickness = thickness)
    return 
    

  def __call__(self, sample):
    image = sample[0]
    mask = sample[1]["mask"]
    results = sample[1]

    image = self._drawer(image, results)
    fig, axe = plt.subplots(nrows = 1, ncols = 2)

    print(mask)

    axe[0].imshow(image)
    axe[1].imshow(mask[..., 2])
    plt.tight_layout()
    plt.show()
    return 

if __name__ == "__main__":
  path = "/media/vbanna/DATA_SHARE/tfds"
  dataset = "coco/2017_panoptic"
  train = tfds.load(dataset, data_dir = path, split = "train")

  draw = PanopticDraw()

  for i, sample in enumerate(train):
    sample["panoptic_objects"]["classes"] = sample["panoptic_objects"]["label"]
    sample["panoptic_objects"]["mask"] = sample["panoptic_image"]
    sample = (sample["image"], sample["panoptic_objects"])

    draw(sample)
    if i > 10:
      break
    





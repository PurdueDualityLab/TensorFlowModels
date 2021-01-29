import tensorflow_datasets as tfds 
import tensorflow as tf
import matplotlib.pyplot as plt
import utils.demos.utils as utils

from panoptic.dataloaders.decoders import tfds_panoptic_coco_decoder

path = "/media/vbanna/DATA_SHARE/tfds"
dataset = "coco/2017_panoptic"
val = tfds.load(dataset, data_dir = path, split = "validation")
drawer = utils.DrawBoxes(classes = 133, display_names=False)

decoder = tfds_panoptic_coco_decoder.MSCOCODecoder(include_mask=True)
val = val.map(decoder.decode)

lim = 10
for i, sample in enumerate(val):
  fig, axe = plt.subplots(1, 3)
  
  axe[0].imshow(sample["groundtruth_semantic_mask"])
  axe[1].imshow(sample["groundtruth_instance_id"] % 256)

  image = sample["image"]/255
  
  results = {"classes": sample["groundtruth_classes"], "bbox":sample['groundtruth_boxes']}
  image = drawer(image, results)
  axe[2].imshow(image)

  fig.set_size_inches(18.5, 10.5, forward=True)
  plt.tight_layout()
  plt.show()
  if i > (lim + 1):
    break
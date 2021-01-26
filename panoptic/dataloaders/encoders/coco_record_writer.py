"""MS Coco Dataset."""

import collections
import json
import os

from absl import logging
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
import pycocotools.mask as mask_utils
import json 
import matplotlib.pyplot as plt
import numpy as np 


def reformat_dictionary(things_file, stuff_file, panoptic_file):
  reformatted = dict()
  for file in things_file["images"]:
    reformatted[file["id"]] = {"image": file}
  
  for file in things_file["annotations"]:
    if file["id"] in reformatted.keys():
      if "annotations" not in reformatted[file["id"]].keys():
        reformatted[file["id"]]["things"] = [file]
      else:
        reformatted[file["id"]]["things"].append(file)
    
  for file in stuff_file["annotations"]:
    if file["id"] in reformatted.keys():
      if "annotations" not in reformatted[file["id"]].keys():
        reformatted[file["id"]]["stuff"] = [file]
      else:
        reformatted[file["id"]]["stuff"].append(file)

  for file in pantopic_file["annotations"]:
    if file["id"] in reformatted.keys():
      if "annotations" not in reformatted[file["id"]].keys():
        reformatted[file["id"]]["annotations"] = [file]
      else:
        reformatted[file["id"]]["annotations"].append(file)
  return 

def get_image_from_id(id):
  image = "%s%012d.jpg" % (image_path, id)
  return image

def get_polygon_mask(masks, image_shape):
  masks = mask_utils.frPyObjects(masks, image_shape[0], image_shape[1])
  masks = mask_utils.merge(masks)
  masks = mask_utils.decode(masks).astype(np.uint8)
  return masks


def get_rsi_mask(masks):
  bin_mask = []
  zero = True
  for value in masks["counts"]:
    if zero:
      bin_mask.extend([0]*value)
    else:
      bin_mask.extend([1]*value)
    zero = not zero
  
  bin_mask = np.array(bin_mask)
  masks_vals = np.reshape(bin_mask, masks["size"], order = 'F').astype(np.uint8)

  print(bin_mask.shape, masks["size"][0] * masks["size"][1])
  # mask 
  return masks_vals

def decode_things_annotation(annotation):
  image = get_image_from_id(annotation["image_id"])
  imdata = plt.imread(image)
  if annotation["iscrowd"] == 0:
    # polygon format
    annotation["segmentation"] = get_polygon_mask(annotation["segmentation"], imdata.shape)
  else:
    #RSI foramt
    annotation["segmentation"] = get_rsi_mask(annotation["segmentation"])
  annotation["image"] = imdata
  return annotation

def decode_stuff_annotation(annotation):
  print(annotation.keys())
  image = get_image_from_id(annotation["image_id"])
  masks = annotation["segmentation"]
  mask = mask_utils.decode(masks).astype(np.uint8)


  print(annotation["bbox"])
  plt.imshow(imdata)
  plt.show()
  plt.imshow(mask)
  plt.show()
  return annotation

def generate_unified_mask():
  return 

def serialized_sample():
  return 

def convert_to_record():
  return 

def write_tfrecord():
  return 
  
# traipaziodal reiman sums 

dataset_path = "/media/vbanna/DATA_SHARE/COCO_raw/"
image_path = dataset_path + "val2017/"
instance_path = dataset_path + "annotations/instances_val2017.json"
segment_path = dataset_path + "annotations/stuff_val2017.json"
panoptic_path = dataset_path + "annotations/panoptic_val2017.json"

segmentation = open(segment_path, 'r')
segmentation = json.load(segmentation)
stuff = segmentation["annotations"][2]

panoptic_ = open(panoptic_path, 'r')
panoptic_ = json.load(panoptic_)
panoptic = panoptic_["annotations"][2]

instances = open(instance_path, 'r')
instances = json.load(instances)
instance = instances["annotations"][2]

print(instance)
print(stuff)
print(panoptic)





# def extrapolate_line(x1, x2, y1, y2):

  

#   #slope = xdist/ydist if ydist != 0 else 0


#   return 


# def get_reiman_mask(masks, image_shape):
#   i = 0 
#   masks.append(masks[0])
#   masks.append(masks[1])

#   mask = np.zeros(shape = image_shape)

#   i = 3
#   while (i < len(masks)):
#     x1 = int(masks[i - 3])
#     y1 = int(masks[i - 2])
#     x2 = int(masks[i - 1])
#     y2 = int(masks[i])

#     xmin = min(x1, x2)
#     xmax = max(x1, x2)

#     xdist = x1 - x2 
#     ydist = y1 - y2

#     slope = ydist/xdist if xdist != 0 else 0
#     for j in range(xmin, xmax + 1):
#       pix = int(slope * (j - xmin))
#       mask[y1 + pix, j] = 1

#     # #temp
#     mask[y1, x1] = 1
    
#     i += 2
#     # print(x1, y1, x2, y2, "||", xmid, ymid)
#   return mask

# def get_reiman_mask(masks, image_shape):
#   i = 0 
#   masks.append(masks[0])
#   masks.append(masks[1])

#   mask = np.zeros(shape = image_shape)

#   i = 3
#   while (i < len(masks)):
#     x1 = int(masks[i - 3])
#     y1 = int(masks[i - 2])
#     x2 = int(masks[i - 1])
#     y2 = int(masks[i])

#     xmin = min(x1, x2)
#     xmax = max(x1, x2) + 1

#     ymin = min(y1, y2)
#     ymax = max(y1, y2) + 1


#     xdir = -1 if x1 > x2 else 1
#     xdist = (xmax - xmin) * xdir

#     ydir = -1 if y1 > y2 else 1
#     ydist = (ymax - ymin) * ydir

#     slope = ydist/xdist if xdist != 0 else 0
#     for j in range(xmin, xmax):
#       pix = int(slope * (j - xmin)) + y1

#       if slope <= 0:
#         mask[pix:ymax, j] = 1
#       else:
#         mask[pix:ymax, j] = 1
        
#     mask[ymin:ymax, xmin:xmax] = 1

#     # #temp
#     mask[y1, x1] = 1
    
#     i += 2
#     # print(x1, y1, x2, y2, "||", xmid, ymid)
#   return mask

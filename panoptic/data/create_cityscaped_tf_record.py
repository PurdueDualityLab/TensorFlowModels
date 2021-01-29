import tensorflow as tf 
import os
import cv2
import json
from PIL import Image, ImageDraw
import numpy as np

dataset_folder = "/media/vbanna/DATA_SHARE/CityScapes_raw"

path_images = dataset_folder + "/leftImg8bit"
path_gt_fine = dataset_folder + "/gtFine"
path_gt_coarse = dataset_folder + "/gtCoarse"

train = "/train"
test = "/test"
val = "/val"

def _get_ID(file):
  file = file.split("_")[:-1]
  ID = "_".join(file)
  return ID

def get_file_lists(image_path, file):
  image_path = image_path.replace(path_images, "")
  ID = _get_ID(image_path +"/"+ file)

  image_file = path_images + image_path + "/" + file
  fine_labels = path_gt_fine + ID + "_gtFine_labelIds.png"
  fine_id = path_gt_fine + ID + "_gtFine_instanceIds.png"
  fine_polygons = path_gt_fine + ID + "_gtFine_polygons.json"
  coarse_labels = path_gt_coarse + ID + "_gtCoarse_labelIds.png"
  coarse_id = path_gt_fine + ID + "_gtCoarse_instanceIds.png"
  coarse_polygons = path_gt_coarse + ID + "_gtCoarse_polygons.json"

  sample = {"image": image_file, 
            "fine_instances": fine_polygons, 
            "coarse_instances": coarse_polygons, 
            "fine_id": fine_id, 
            "coarse_id": coarse_id, 
            "fine_labels": fine_labels,
            "coarse_labels": coarse_labels}

  return sample

def _get_file_generator(image_folder, split):
  path = image_folder + split
  files = os.walk(path, topdown = False)

  samples = []
  for i, (folder, dirs, files) in enumerate(files):
    if i > 10:
      break 
    for file in files:
      samples.append(get_file_lists(folder, file))
  return samples

import pycocotools.mask as mask_utils
def draw_polygons():
  # mask_utils.
  return

def read_png_image(image_file):
  with tf.io.gfile.GFile(image_file, 'rb') as fid:
    encoded_png = fid.read()
  encoded_png = tf.io.decode_png(encoded_png)
  return encoded_png

def read_json_sample(image_json):
  with tf.io.gfile.GFile(image_json, 'r') as fid:
    json_file = fid.read()
  json_file = json.loads(json_file)
  return json_file

def load_instance(polygon, width, height):
  # ImageDraw.Draw()
  polygon = np.array(polygon)
  polygon_a = np.reshape(polygon, (-1)).tolist()

  mask = mask_utils.frPyObjects([polygon_a], height, width)
  mask = mask_utils.decode(mask)

  return mask

def get_instance_list(json_file):
  return 

# import matplotlib.pyplot as plt
# samples = _get_file_generator(path_images, train)

# image = read_png_image(samples[0]["image"])
# plt.imshow(image)
# plt.show()

# json_polygons = read_json_sample(samples[0]["fine_instances"])
# for polygon in json_polygons['objects']:
#   print(polygon["label"])
#   instance = load_instance(polygon["polygon"], json_polygons["imgWidth"], json_polygons["imgHeight"])
#   plt.imshow(image_)
#   plt.show()
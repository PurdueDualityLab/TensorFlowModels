import tensorflow as tf
from PIL import Image
from tqdm import tqdm
import io
import os
import json


def encode_image(image, fmt='JPEG'):
  with io.BytesIO() as output:
    image.save(output, format=fmt)
    return output.getvalue()


def serialize(img_bytes, source_id, height, width, xmins, xmaxs, ymins, ymaxs,
              labels, is_crowds, areas):
  serialized_example = tf.train.Example(
      features=tf.train.Features(
          feature={
              'image/encoded': (tf.train.Feature(
                  bytes_list=tf.train.BytesList(value=[img_bytes]))),
              'image/source_id': (tf.train.Feature(
                  bytes_list=tf.train.BytesList(value=[source_id]))),
              'image/height': (tf.train.Feature(
                  int64_list=tf.train.Int64List(value=[height]))),
              'image/width': (tf.train.Feature(
                  int64_list=tf.train.Int64List(value=[width]))),
              'image/object/bbox/xmin': (tf.train.Feature(
                  float_list=tf.train.FloatList(value=xmins))),
              'image/object/bbox/xmax': (tf.train.Feature(
                  float_list=tf.train.FloatList(value=xmaxs))),
              'image/object/bbox/ymin': (tf.train.Feature(
                  float_list=tf.train.FloatList(value=ymins))),
              'image/object/bbox/ymax': (tf.train.Feature(
                  float_list=tf.train.FloatList(value=ymaxs))),
              'image/object/class/label': (tf.train.Feature(
                  int64_list=tf.train.Int64List(value=labels))),
              'image/object/is_crowd': (tf.train.Feature(
                  int64_list=tf.train.Int64List(value=is_crowds))),
              'image/object/area': (tf.train.Feature(
                  float_list=tf.train.FloatList(value=areas))),
          })).SerializeToString()
  return serialized_example


def convert(img_folder, annotations_file, record):
  with open(annotations_file, 'r') as file:
    dic = json.load(file)

  with tf.io.TFRecordWriter(record) as writer:
    for img in tqdm(dic['images']):
      img_bytes = encode_image(
          Image.open(os.path.join(img_folder, f'{img["id"]:012d}.jpg')))
      source_id = str(img['id']).encode()
      height = img['height']
      width = img['width']
      xmins = []
      ymins = []
      xmaxs = []
      ymaxs = []
      is_crowds = []
      labels = []
      areas = []
      for annotation in dic['annotations']:
        if annotation['image_id'] == img['id']:
          x, y, w, h = annotation['bbox']
          xmins.append(x)
          xmaxs.append(x + w)
          ymins.append(y)
          ymaxs.append(y + h)
          is_crowds.append(annotation['iscrowd'])
          labels.append(annotation['category_id'])
          areas.append(annotation['area'])
      tfexample = serialize(img_bytes, source_id, height, width, xmins, xmaxs,
                            ymins, ymaxs, labels, is_crowds, areas)
      writer.write(tfexample)


if __name__ == '__main__':
  convert('coco_subset/train2017', 'coco_subset/train2017_labels.json',
          'train.tfrecord')
  convert('coco_subset/val2017', 'coco_subset/val2017_labels.json',
          'val.tfrecord')

import tensorflow_datasets as tfds 


class MSCOCODecoder(decoder.Decoder):
  """Tensorflow Example proto decoder."""

  def __init__(self,
               include_mask=False,
               regenerate_source_id=False):
    self._include_mask = include_mask
    self._regenerate_source_id = regenerate_source_id

  def _decode_image(self, parsed_tensors):
    """Decodes the image and set its static shape."""
    image = tf.io.decode_image(parsed_tensors['image/encoded'], channels=3)
    image.set_shape([None, None, 3])
    return image

  def decode(self, sample):
    """Decode the serialized example"""


    decoded_tensors = {
        'source_id': source_id,
        'image': image,
        'height': parsed_tensors['image/height'],
        'width': parsed_tensors['image/width'],
        'groundtruth_classes': classes,
        'groundtruth_is_crowd': is_crowds,
        'groundtruth_area': areas,
        'groundtruth_boxes': boxes,
    }
    return decoded_tensors

coco, info = tfds.load('coco', split = 'train', with_info= True)

for i in coco.take(1):
    print(i)

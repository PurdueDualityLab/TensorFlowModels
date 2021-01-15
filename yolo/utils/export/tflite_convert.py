import os
import tensorflow as tf
from yolo.utils.run_utils import prep_gpu
from yolo.configs import yolo as exp_cfg
from yolo.tasks.yolo import YoloTask
from skimage import io
import cv2
prep_gpu()


def create_metadata(model_file_name, label_map_file_name, num_labels):
  from tflite_support import flatbuffers
  from tflite_support import metadata as _metadata
  from tflite_support import metadata_schema_py_generated as _metadata_fb
  """ ... """
  """Creates the metadata for an image classifier."""

  # Creates model info.
  model_meta = _metadata_fb.ModelMetadataT()
  model_meta.name = 'MobileNetV1 image classifier'
  model_meta.description = ('Identify the most prominent object in the '
                            'image from a set of 1,001 categories such as '
                            'trees, animals, food, vehicles, person etc.')
  model_meta.version = 'v1'
  model_meta.author = 'TensorFlow'
  model_meta.license = ('Apache License. Version 2.0 '
                        'http://www.apache.org/licenses/LICENSE-2.0.')

  # Creates input info.
  input_meta = _metadata_fb.TensorMetadataT()
  input_meta.name = 'image'
  input_meta.description = (
      'Input image to be classified. The expected image is {0} x {1}, with '
      'three channels (red, blue, and green) per pixel. Each value in the '
      'tensor is a single byte between 0 and 255.'.format(416, 416))
  input_meta.content = _metadata_fb.ContentT()
  input_meta.content.contentProperties = _metadata_fb.ImagePropertiesT()
  input_meta.content.contentProperties.colorSpace = (
      _metadata_fb.ColorSpaceType.RGB)
  input_meta.content.contentPropertiesType = (
      _metadata_fb.ContentProperties.ImageProperties)
  input_normalization = _metadata_fb.ProcessUnitT()
  input_normalization.optionsType = (
      _metadata_fb.ProcessUnitOptions.NormalizationOptions)
  input_normalization.options = _metadata_fb.NormalizationOptionsT()
  input_normalization.options.mean = [127.5]
  input_normalization.options.std = [127.5]
  input_meta.processUnits = [input_normalization]
  input_stats = _metadata_fb.StatsT()
  input_stats.max = [255]
  input_stats.min = [0]
  input_meta.stats = input_stats

  # Creates output info.
  bbox_meta = _metadata_fb.TensorMetadataT()
  bbox_meta.name = 'bbox'
  bbox_meta.description = '.'
  bbox_meta.content = _metadata_fb.ContentT()
  bbox_meta.content.content_properties = _metadata_fb.FeaturePropertiesT()
  bbox_meta.content.contentPropertiesType = (
      _metadata_fb.ContentProperties.FeatureProperties)
  bbox_stats = _metadata_fb.StatsT()
  bbox_stats.max = [416.0]
  bbox_stats.min = [0.0]
  bbox_meta.stats = bbox_stats

  classes_meta = _metadata_fb.TensorMetadataT()
  classes_meta.name = 'classes'
  classes_meta.description = '.'
  classes_meta.content = _metadata_fb.ContentT()
  classes_meta.content.content_properties = _metadata_fb.FeaturePropertiesT()
  classes_meta.content.contentPropertiesType = (
      _metadata_fb.ContentProperties.FeatureProperties)
  classes_stats = _metadata_fb.StatsT()
  classes_stats.max = [num_labels]
  classes_stats.min = [0]
  classes_meta.stats = classes_stats
  label_file = _metadata_fb.AssociatedFileT()
  label_file.name = os.path.basename(label_map_file_name)
  label_file.description = 'Labels for objects that the model can recognize.'
  label_file.type = _metadata_fb.AssociatedFileType.TENSOR_AXIS_LABELS
  classes_meta.associatedFiles = [label_file]

  confidence_meta = _metadata_fb.TensorMetadataT()
  confidence_meta.name = 'confidence'
  confidence_meta.description = '.'
  confidence_meta.content = _metadata_fb.ContentT()
  confidence_meta.content.content_properties = _metadata_fb.FeaturePropertiesT()
  confidence_meta.content.contentPropertiesType = (
      _metadata_fb.ContentProperties.FeatureProperties)
  confidence_stats = _metadata_fb.StatsT()
  confidence_stats.max = [1.0]
  confidence_stats.min = [0.0]
  confidence_meta.stats = confidence_stats

  num_dets_meta = _metadata_fb.TensorMetadataT()
  num_dets_meta.name = 'num_dets'
  num_dets_meta.description = '.'
  num_dets_meta.content = _metadata_fb.ContentT()
  num_dets_meta.content.content_properties = _metadata_fb.FeaturePropertiesT()
  num_dets_meta.content.contentPropertiesType = (
      _metadata_fb.ContentProperties.FeatureProperties)
  num_dets_stats = _metadata_fb.StatsT()
  num_dets_stats.max = [200]
  num_dets_stats.min = [0]
  num_dets_meta.stats = num_dets_stats

  raw_output_meta = _metadata_fb.TensorMetadataT()
  raw_output_meta.name = 'raw_output'
  raw_output_meta.description = '.'
  raw_output_meta.content = _metadata_fb.ContentT()
  raw_output_meta.content.content_properties = _metadata_fb.FeaturePropertiesT()
  raw_output_meta.content.contentPropertiesType = (
      _metadata_fb.ContentProperties.FeatureProperties)

  subgraph = _metadata_fb.SubGraphMetadataT()
  subgraph.inputTensorMetadata = [input_meta]
  subgraph.outputTensorMetadata = [
      bbox_meta, classes_meta, confidence_meta, num_dets_stats
  ]  # raw_output_meta
  model_meta.subgraphMetadata = [subgraph]

  b = flatbuffers.Builder(0)
  b.Finish(
      model_meta.Pack(b), _metadata.MetadataPopulator.METADATA_FILE_IDENTIFIER)
  metadata_buf = b.Output()

  populator = _metadata.MetadataPopulator.with_model_file(model_file_name)
  populator.load_metadata_buffer(metadata_buf)
  populator.load_associated_files([label_map_file_name])
  populator.populate()


def conversion(model):

  @tf.function
  def run(image):
    with tf.device('cpu:0'):
      image = tf.cast(image, tf.float32)
      image = image / tf.fill(tf.shape(image), 255.0)
      pred = model.call(image, training=False)
      pred = {
          'bbox': tf.cast(pred['bbox'], tf.float32),
          'classes': tf.cast(pred['classes'], tf.float32),
          'confidence': tf.cast(pred['confidence'], tf.float32),
          'num_dets': tf.cast(pred['num_dets'], tf.float32)
      }
    return pred

  return run


def get_rep_data():
  import tensorflow_datasets as tfds
  dataset = tfds.load('coco', split='train', shuffle_files=True)
  dataset = dataset.take(100)

  def representative_dataset():
    for data in dataset:
      data = tf.cast(
          tf.expand_dims(tf.image.resize(data['image'], (416, 416)), axis=0),
          tf.uint8)
      yield [data]

  return representative_dataset


def url_to_image(url):
  image = io.imread(url)
  return image


def uniary_convert():
  with tf.device('gpu:0'):
    model = None
    input_size = [416, 416, 3]
    # config = exp_cfg.YoloTask(
    #     model=exp_cfg.Yolo(_input_size = [416, 416, 3],
    #                       base='v4tiny',
    #                       min_level=4,
    #                       norm_activation = exp_cfg.common.NormActivation(activation="leaky"),
    #                       _boxes = ['(10, 14)', '(23, 27)', '(37, 58)', '(81, 82)', '(135, 169)', '(344, 319)'],
    #                       #_boxes = ['(20, 28)', '(46, 54)', '(74, 116)', '(81, 82)', '(135, 169)', '(344, 319)'],
    #                       #_boxes = ["(10, 13)", "(16, 30)", "(33, 23)","(30, 61)", "(62, 45)", "(59, 119)","(116, 90)", "(156, 198)", "(373, 326)"],
    #                       #_boxes = ['(12, 16)', '(19, 36)', '(40, 28)', '(36, 75)','(76, 55)', '(72, 146)', '(142, 110)', '(192, 243)','(459, 401)'],
    #                       filter = exp_cfg.YoloLossLayer(use_nms=False)
    #                       ))
    config = exp_cfg.YoloTask(
        model=exp_cfg.Yolo(
            _input_size=input_size,
            base='v3',
            min_level=3,
            norm_activation=exp_cfg.common.NormActivation(activation='leaky'),
            _boxes=[
                '(10, 13)', '(16, 30)', '(33, 23)', '(30, 61)', '(62, 45)',
                '(59, 119)', '(116, 90)', '(156, 198)', '(373, 326)'
            ],
            filter=exp_cfg.YoloLossLayer(use_nms=False)))
    task = YoloTask(config)
    model = task.build_model()
    task.initialize(model)
    #model.build((1, 416, 416, 3))
    model(tf.ones((1, 416, 416, 3), dtype=tf.float32), training=False)

    image = url_to_image(
        'https://raw.githubusercontent.com/zhreshold/mxnet-ssd/master/data/demo/dog.jpg'
    )
    image = cv2.resize(image, (416, 416))
    image = tf.expand_dims(image, axis=0)
    func = conversion(model)
    model.summary()

    # Convert the model
    converter = tf.lite.TFLiteConverter.from_concrete_functions(
        [func.get_concrete_function(image)])
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    #converter.representative_dataset = get_rep_data()

    try:
      tflite_model = converter.convert()
    except BaseException:
      print('here')
      # st.print_exc()
      import sys
      sys.exit()

    with open('detect.tflite', 'wb') as f:
      f.write(tflite_model)


if __name__ == '__main__':
  uniary_convert()

  # with open("saved_models/v4/tiny_no_nms/label_map.txt", 'w') as label_map:
  #   label_map.write("???\n")
  #   with open("yolo/dataloaders/dataset_specs/coco.names") as coco:
  #     lines = coco.readlines()
  #     num_labels = len(lines)
  #     label_map.writelines(lines)
  # Save the model.

  #create_metadata('detect-large.tflite', 'saved_models/v4/tiny_no_nms/label_map.txt', num_labels)

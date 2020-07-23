#!/usr/bin/env python3
"Convert a DarkNet config file and weights into a TensorFlow model"

from absl import app, flags
from absl.flags import argparse_flags
import argparse

flags.DEFINE_boolean('weights_only', False, 'Save only the weights and not the entire model.')
flags.DEFINE_integer('input_image_size', 224, 'Size of the image to be used as an input.')

def makeParser(parser):
    parser.add_argument('cfg', default=None, help='name of the config file. Defaults to YOLOv3', nargs='?')
    parser.add_argument('weights', default=None, help='name of the weights file. Defaults to YOLOv3', nargs='?')
    parser.add_argument('output', help='name of the location to save the generated model')

parser = argparse_flags.ArgumentParser()
makeParser(parser)

def main(argv, args=None):
    from ...file_manager import download
    if args is None:
        args = parser.parse_args(argv[1:])

    cfg = args.cfg
    weights = args.weights
    output = args.output
    if cfg is None:
        cfg = download('yolov3', 'cfg')
    if weights is None:
        weights = download('yolov3', 'weights')

    # This is horrible design that makes it impossible to load any model except
    # YOLOv3, but I have no option right now.
    from yolo.modeling.yolo_v3 import DarkNet53
    import tensorflow as tf
    model = DarkNet53(classes = 1000, load_backbone_weights = True, config_file=cfg, weights_file=weights)
    input_image_size = flags.FLAGS.input_image_size
    x = tf.ones(shape=[1, input_image_size, input_image_size, 3], dtype = tf.float32)
    model.predict(x)
    if flags.FLAGS.weights_only:
        model.save_weights(output)
    else:
        model.save(output)

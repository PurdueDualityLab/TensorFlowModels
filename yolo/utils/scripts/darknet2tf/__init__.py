#!/usr/bin/env python3
"Convert a DarkNet config file and weights into a TensorFlow model"

from absl import app, flags
from absl.flags import argparse_flags
import argparse
import os

from ._darknet_model import DarkNetModel
from ._read_weights import read_weights, split_list

flags.DEFINE_boolean('weights_only', False,
                     'Save only the weights and not the entire model.')
flags.DEFINE_integer('input_image_size', 224,
                     'Size of the image to be used as an input.')


def _makeParser(parser):
    parser.add_argument(
        'cfg',
        default=None,
        help='name of the config file. Defaults to YOLOv3',
        type=argparse.FileType('r'),
        nargs='?')
    parser.add_argument(
        'weights',
        default=None,
        help='name of the weights file. Defaults to YOLOv3',
        type=argparse.FileType('rb'),
        nargs='?')
    parser.add_argument(
        'output', help='name of the location to save the generated model')


_parser = argparse_flags.ArgumentParser()
_makeParser(_parser)


def main(argv, args=None):
    from ...file_manager import download
    if args is None:
        args = _parser.parse_args(argv[1:])

    cfg = args.cfg
    weights = args.weights
    output = args.output
    if cfg is None:
        cfg = download('yolov3.cfg')
    if weights is None:
        weights = download('yolov3.weights')

    """
    # This is horrible design that makes it impossible to load any model except
    # YOLOv3, but I have no option right now.
    from yolo.modeling.yolo_v3 import Yolov3
    import tensorflow as tf
    model = Yolov3(classes=80)
    model.load_weights_from_dn(dn2tf_backbone = True, dn2tf_head = True, config_file=cfg, weights_file=weights)
    input_image_size = flags.FLAGS.input_image_size
    x = tf.ones(shape=[1, input_image_size,
                       input_image_size, 3], dtype=tf.float32)
    model.predict(x)
    if output != os.devnull:
        if flags.FLAGS.weights_only:
            model.save_weights(output)
        else:
            model.save(output)
    """

    import tensorflow as tf
    model = read_weights(cfg, weights).to_tf()
    print(model)

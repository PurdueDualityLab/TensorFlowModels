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

    model = read_weights(cfg, weights).to_tf()
    if output != os.devnull:
        if flags.FLAGS.weights_only:
            model.save_weights(output)
        else:
            model.save(output)

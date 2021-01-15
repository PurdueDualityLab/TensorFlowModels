# !/usr/bin/env python3
import argparse as _argparse
from absl.flags import argparse_flags as _argparse_flags
from absl import flags as _flags
from . import DarkNetConverter
from absl import app
import sys
from . import main, _parser

"Convert a DarkNet config file and weights into a TensorFlow model"

_flags.DEFINE_boolean("weights_only", False,
                      "Save only the weights and not the entire model.")


def _makeParser(parser):
  parser.add_argument(
      "cfg",
      default=None,
      help="name of the config file. Defaults to YOLOv3",
      type=_argparse.FileType("r"),
      nargs="?")
  parser.add_argument(
      "weights",
      default=None,
      help="name of the weights file. Defaults to YOLOv3",
      type=_argparse.FileType("rb"),
      nargs="?")
  parser.add_argument(
      "output", help="name of the location to save the generated model")


def main(argv, args=None):
  from yolo.utils.downloads.file_manager import download
  import os

  if args is None:
    args = _parser.parse_args(argv[1:])

  cfg = args.cfg
  weights = args.weights
  output = args.output
  if cfg is None:
    cfg = download("yolov3.cfg")
  if weights is None:
    weights = download("yolov3.weights")

  model = DarkNetConverter.read(cfg, weights).to_tf()
  if output != os.devnull:
    if flags.FLAGS.weights_only:
      model.save_weights(output)
    else:
      model.save(output)


_parser = _argparse_flags.ArgumentParser()
_makeParser(_parser)

if __name__ == "__main__":
  # I dislike Abseil's current help menu. I like the default Python one
  # better
  if "-h" in sys.argv or "--help" in sys.argv:
    _parser.parse_args(sys.argv[1:])
    exit()
  app.run(main)

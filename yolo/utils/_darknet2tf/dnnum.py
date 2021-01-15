#!/usr/bin/env python3
"Number the blocks in a DarkNet config file"

from absl import app
from absl.flags import argparse_flags
import argparse

from yolo.utils.downloads import file_manager


def _makeParser(parser):
  parser.add_argument(
      "filename",
      default=None,
      help="name of the config file. Defaults to YOLOv3",
      nargs="?",
      type=argparse.FileType("r"))


_parser = argparse_flags.ArgumentParser()
_makeParser(_parser)


def numberConfig(file):
  i = 0
  for line in file:
    if line.startswith("[") and "net" not in line:
      print(f"{i:4d}|{line}", end="")
      i += 1
    else:
      print(f"    |{line}", end="")


def main(argv, args=None):
  if args is None:
    args = _parser.parse_args(argv[1:])

  filename = args.filename
  if filename is None:
    with open(file_manager.download("yolov3.cfg")) as file:
      numberConfig(file)
  else:
    numberConfig(filename)


if __name__ == "__main__":
  app.run(main)

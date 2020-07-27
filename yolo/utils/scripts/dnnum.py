#!/usr/bin/env python3
"Number the blocks in a DarkNet config file"

from absl import app, flags
from absl.flags import argparse_flags
import argparse


def makeParser(parser):
    parser.add_argument(
        'filename',
        default=None,
        help='name of the config file. Defaults to YOLOv3',
        nargs='?',
        type=argparse.FileType('r'))


parser = argparse_flags.ArgumentParser()
makeParser(parser)


def numberConfig(file):
    i = 0
    for line in file:
        if line.startswith('[') and line != '[net]\n':
            print(f"{i:4d}|{line}", end='')
            i += 1
        else:
            print(f"    |{line}", end='')


def main(argv, args=None):
    if args is None:
        args = parser.parse_args(argv[1:])

    filename = args.filename
    if filename is None:
        from ..file_manager import download
        with open(download('yolov3.cfg')) as file:
            numberConfig(file)
    else:
        numberConfig(filename)


if __name__ == '__main__':
    app.run(main)

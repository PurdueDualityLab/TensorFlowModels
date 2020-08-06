#!/usr/bin/env python3
"Number the blocks in a DarkNet config file"

from __future__ import annotations

from absl import app, flags
from absl.flags import argparse_flags
import argparse

from typing import Iterable, Iterator, Union


def _makeParser(parser):
    parser.add_argument(
        'filename',
        default=None,
        help='name of the config file. Defaults to YOLOv3',
        nargs='?',
        type=argparse.FileType('r'))


_parser = argparse_flags.ArgumentParser()
_makeParser(_parser)


def numberConfig(file: Union[Iterable, Iterator]) -> None:
    """
    Number all of the lines in a file and print the result to the screen.

    Arguments:
        file: an iterable/iterator that contains all of the contents of a file
              Can be a text file itself, but not the path.
    """
    i = 0
    for line in file:
        if line.startswith('['):
            print(f"{i:4d}|{line}", end='')
            i += 1
        else:
            print(f"    |{line}", end='')


def main(argv, args=None):
    if args is None:
        args = _parser.parse_args(argv[1:])

    filename = args.filename
    if filename is None:
        from ..file_manager import download
        with open(download('yolov3.cfg')) as file:
            numberConfig(file)
    else:
        numberConfig(filename)


if __name__ == '__main__':
    app.run(main)

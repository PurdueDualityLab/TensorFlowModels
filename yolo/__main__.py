"""
Run a script from the yolo demos package:

python3 -m yolo dnnum yolov3.cfg

To see a full list of these commands, see
python3 -m yolo -h
"""

import importlib
import sys

try:
    from absl import app, flags
    from absl.flags import argparse_flags
except ImportError as e:
    raise ImportError(
        "Some requirements were not installed. See the README to see how to install the packages.") from e


from . import demos_testing


def makeParser(parser):
    subparsers = parser.add_subparsers(
        help='demo script to run', metavar='util', dest="_util")
    subparsers.required = True
    """
    for util in demos_testing.__all__:
        module = importlib.import_module('yolo.demos_testing.' + util)
        if hasattr(module, '_makeParser'):
            sub = subparsers.add_parser(util, help=module.__doc__)
            module._makeParser(sub)
    """


parser = argparse_flags.ArgumentParser(prog=f'{sys.executable} -m yolo')
makeParser(parser)


def main(argv):
    # This is somehow needed for argparse to work with Abseil
    # Couldn't tell you why ¯\_(ツ)_/¯
    if len(argv) == 1:
        parser.print_usage()
        exit()

    args = parser.parse_args(argv[1:])
    util = args._util

    module = importlib.import_module('yolo.demos_testing.' + util)
    module.main(argv, args=args)


if __name__ == '__main__':
    # I dislike Abseil's current help menu. I like the default Python one
    # better
    if '-h' in sys.argv or '--help' in sys.argv:
        parser.parse_args(sys.argv[1:])
        exit()
    app.run(main)

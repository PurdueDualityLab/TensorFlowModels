#!/usr/bin/env python3
"Convert a DarkNet config file into a Python literal file in a list of dictionaries format"

from absl import app, flags
from absl.flags import argparse_flags
import argparse
import json
from pprint import pprint
import sys


def _makeParser(parser):
    """
    Make a parser for the Abseil utility script. This is not the Darknet parser.
    """
    parser.add_argument(
        'config',
        default=None,
        help='name of the config file. Defaults to YOLOv3',
        nargs='?',
        type=argparse.FileType('r'))
    parser.add_argument(
        'dictsfile',
        default=sys.stdout,
        nargs='?',
        help='name of the Python literal file',
        type=argparse.FileType('w'))


_parser = argparse_flags.ArgumentParser()
_makeParser(_parser)


def parseValue(v):
    """
    Parse non-string literals found in darknet config files
    """
    if ',  ' in v:
        vals = v.split(',  ')
        return tuple(parseValue(v) for v in vals)
    elif ',' in v:
        vals = v.split(',')
        return tuple(parseValue(v) for v in vals)
    else:
        if '.' in v:
            try:
                return float(v.strip())
            except ValueError:
                return v
        else:
            try:
                return int(v.strip())
            except ValueError:
                return v


def convertConfigFile(configfile, break_script="###########"):
    """
    Convert an opened config file to a list of dictinaries.
    """
    mydict = None

    for line in configfile:
        if break_script is not None and line.startswith(break_script):
            yield mydict
            mydict = {"_type": "decoder_encoder_split"}
        elif line.startswith('['):
            if mydict is not None:
                yield mydict
            mydict = {}
            mydict['_type'] = line.strip('[] \n')
        else:
            line, *_ = line.split('#', 1)
            if line.strip() != '':
                k, v = line.split('=', 1)
                mydict[k.strip()] = parseValue(v)


def main(argv, args=None):
    if args is None:
        args = _parser.parse_args(argv[1:])

    config = args.config
    dictsfile = args.dictsfile

    if config is None:
        from ..file_manager import download
        with open(download('yolov3.cfg')) as config:
            output = list(convertConfigFile(config))
    else:
        output = list(convertConfigFile(config))

    pprint(output, dictsfile)


if __name__ == '__main__':
    app.run(main)

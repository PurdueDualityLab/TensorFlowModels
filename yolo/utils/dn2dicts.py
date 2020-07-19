#!/usr/bin/env python3
"Convert a DarkNet config file into a Python literal file in a list of dictionaries format"

from absl import app, flags
from absl.flags import argparse_flags
import argparse
import json
from pprint import pprint
import sys


def makeParser(parser):
    parser.add_argument('config', default=None, help='name of the config file. Defaults to YOLOv3', nargs='?', type=argparse.FileType('r'))
    parser.add_argument('dictsfile', default=sys.stdout, nargs='?', help='name of the Python literal file', type=argparse.FileType('w'))

parser = argparse_flags.ArgumentParser()
makeParser(parser)


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


def convertConfigFile(configfile):
    """
    Convert an opened config file to a list of dictinaries.
    """
    output = []
    mydict = None

    for line in configfile:
        if line.startswith('[') and line != '[net]\n':
            mydict = {}
            mydict['ltype'] = line.strip('[] \n')
            output.append(mydict)
        elif mydict is not None:
            line, *_ = line.strip().split('#', 1)
            if '=' in line:
                k, v = line.strip().split('=', 1)
                mydict[k] = parseValue(v)

    return output


def main(argv, args=None):
    if args is None:
        args = parser.parse_args(argv[1:])

    config = args.config
    dictsfile = args.dictsfile

    if config is None:
        with open('yolo/utils/yolov3.cfg') as config:
            output = convertConfigFile(config)
    else:
        output = convertConfigFile(config)

    pprint(output, dictsfile)

if __name__ == '__main__':
    app.run(main)

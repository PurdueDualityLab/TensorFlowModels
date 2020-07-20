#!/usr/bin/env python3
"Convert a DarkNet config file into a Python literal file in a list of dictionaries format"

from absl import app, flags
from absl.flags import FLAGS as args
import argparse
import json
from pprint import pprint

flags.DEFINE_string('config', 'yolov3.cfg', 'name of the config file')
flags.DEFINE_string('dictsfile', 'yolov3.py', 'name of the Python literal file')

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

def dict_check(dictionary, key, check_val):
    try:
        return prev_layer['filters'] == check_val
    except:
        return False
    return False

def convertConfigFile(configfile, break_script = "######################"):
    output = []
    mydict = None

    for line in configfile:
        if break_script != None and line == f"{break_script}\n":
            mydict = {"_type": "decoder_encoder_split"} 
            output.append(mydict)
        if line == '[net]\n':
            mydict = {} 
            mydict['_type'] = line.strip('[] \n')
            output.append(mydict)
        elif line.startswith('['): #and line != '[net]\n':
            mydict = {}
            mydict['_type'] = line.strip('[] \n')
            output.append(mydict)
        elif mydict is not None:
            line, *_ = line.strip().split('#', 1)
            if '=' in line:
                k, v = line.strip().split('=', 1)
                mydict[k] = parseValue(v)
    return output


def main(argv):
    config = args.config
    dictsfile = args.dictsfile
    with open(config) as configfile:
        output = convertConfigFile(configfile)
    with open(dictsfile, 'w') as dictsfilew:
        pprint(output, dictsfilew)
    return output

if __name__ == '__main__':
    app.run(main)

import struct
import numpy as np
import os
from configClasses import *
from dn2dicts import *

def get_size(path):
    """calculate image size changes"""
    data = os.stat(path)
    return data.st_size

def build_layer(layer_dict, file, prevlayer):
    """consturct layer and load weights from file"""
    bytes_read = 0
    if layer_dict['_type'] == 'convolutional':
        layer_dict.update({"w":prevlayer.shape[0], "h":prevlayer.shape[1], "c":prevlayer.shape[2]})
        layer = convCFG(**layer_dict)
        bytes_read = layer.load_weights(file)
    elif layer_dict['_type'] == 'net':
        l = {"_type":layer_dict["_type"], "w":layer_dict["width"], "h":layer_dict["height"], "c":layer_dict["channels"]}
        layer = placeCFG(**l)
    elif layer_dict['_type'] == 'shortcut' or layer_dict['_type'] == 'route' or layer_dict['_type'] == 'yolo':
        l = {"_type":layer_dict["_type"], "w":prevlayer.shape[0], "h":prevlayer.shape[1], "c":prevlayer.shape[2]}
        layer = placeCFG(**l)
    elif layer_dict['_type'] == 'upsample':
        l = {"w":prevlayer.shape[0], "h":prevlayer.shape[1], "c":prevlayer.shape[2], "stride": layer_dict["stride"]}
        layer = upsampleCFG(**l)
    else:
        layer = layer_dict
    
    print(layer_dict['_type'])
    #print(layer)
    return layer, bytes_read


def read_file(config, weights):
    """read the file ans construct weights nets"""
    bytes_read = 0

    major = read_n_int(1, weights)[0]
    bytes_read += 4
    minor = read_n_int(1, weights)[0]
    bytes_read += 4
    revision = read_n_int(1, weights)[0]
    bytes_read += 4

    if ((major * 10 + minor) >= 2):
        print("64 seen")
        iseen = read_n_long(1, weights, unsigned=True)[0]
        bytes_read += 8
    else:
        print("32 seen")
        iseen = read_n_int(1, weights, unsigned=True)[0]
        bytes_read += 8
    
    print(f"major: {major}")
    print(f"minor: {minor}")
    print(f"revision: {revision}")
    print(f"iseen: {iseen}")

    net = [None]
    for layer_dict in config:
        #try:
        layer, num_read = build_layer(layer_dict, weights, net[-1])
        if layer_dict["_type"] != 'yolo' and layer.shape[-1] != 255:
            net.append(layer)
        
        bytes_read += num_read
        #except:
            #break
    return net, bytes_read

def load_weights(config_file, weights_file):
    config = convertConfigFile(open(config_file))
    size = get_size(weights_file)
    weights = open(weights_file, "rb")
    print(weights)
    net, bytes_read = read_file(config, weights)
    print(f"bytes_read: {bytes_read}, original_size: {size}, final_position: {weights.tell()}")
    if (bytes_read != size):
        print("error: could not read the entire weights file")
    return net, bytes_read

config = "yolov3.cfg"
weights = "yolov3_416.weights"
net,_ = load_weights(config, weights)
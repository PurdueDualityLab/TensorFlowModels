import struct
import numpy as np
import os
from configClasses import *
from dn2dicts import *
import itertools


def get_size(path):
    """calculate image size changes"""
    data = os.stat(path)
    return data.st_size


def build_layer(layer_dict, file, prevlayer):
    """consturct layer and load weights from file"""
    bytes_read = 0
    if layer_dict['_type'] == 'convolutional':
        layer_dict.update(
            {"w": prevlayer.shape[0], "h": prevlayer.shape[1], "c": prevlayer.shape[2]})
        layer = convCFG(**layer_dict)
        bytes_read = layer.load_weights(file)
    elif layer_dict['_type'] == 'net':
        l = {
            "_type": layer_dict["_type"],
            "w": layer_dict["width"],
            "h": layer_dict["height"],
            "c": layer_dict["channels"]}
        layer = placeCFG(**l)
    elif layer_dict['_type'] == 'shortcut' or layer_dict['_type'] == 'route' or layer_dict['_type'] == 'yolo':
        l = {
            "_type": layer_dict["_type"],
            "w": prevlayer.shape[0],
            "h": prevlayer.shape[1],
            "c": prevlayer.shape[2]}
        layer = placeCFG(**l)
    elif layer_dict['_type'] == 'upsample':
        l = {
            "w": prevlayer.shape[0],
            "h": prevlayer.shape[1],
            "c": prevlayer.shape[2],
            "stride": layer_dict["stride"]}
        layer = upsampleCFG(**l)
    else:
        layer = layer_dict

    #print(f"reading: {layer_dict['_type']}          ", sep='      ', end = "\r", flush = True)
    #print(f"reading: {layer_dict['_type']}          ", sep='      ', end = "\r", flush = True)
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

    encoder = [None]
    decoder = []
    net = encoder
    outputs = [None]
    for layer_dict in config:
        if layer_dict["_type"] != "decoder_encoder_split":
            layer, num_read = build_layer(layer_dict, weights, net[-1])
            if layer_dict["_type"] != 'yolo' and layer.shape[-1] != 255:
                net.append(layer)
            else:
                outputs.append(layer)
        else:
            net = decoder
            decoder.append(encoder[-1])

        bytes_read += num_read
        print(f"bytes_read: {bytes_read}          ",
              sep='      ', end="\r", flush=True)
    #print(f"bytes_read: {bytes_read}          ")
    del encoder[0]
    del decoder[0]
    return encoder, decoder, outputs, bytes_read


def interleve_weights(block):
    weights_temp = []
    for layer in block:
        weights = layer.get_weights()
        weights = [tuple(weights[0:3]), tuple(weights[3:])]
        weights_temp.append(weights)
    top, bottom = tuple(zip(*weights_temp))
    weights = list(itertools.chain.from_iterable(top)) + \
        list(itertools.chain.from_iterable(bottom))
    return weights


def get_darknet53_tf_format(net, only_weights=True):
    combo_blocks = []
    for i in range(2):
        layer = net.pop(0)
        combo_blocks.append(layer)
    # ugly code i will document, very tired
    encoder = []
    while len(net) != 0:
        blocks = []
        layer = net.pop(0)
        while layer._type != "shortcut":
            blocks.append(layer)
            layer = net.pop(0)
        encoder.append(blocks)
    del combo_blocks[0]
    new_net = combo_blocks + encoder
    weights = []
    if only_weights:
        for block in new_net:
            if type(block) != list:
                weights.append(block.get_weights())
            else:
                weights.append(interleve_weights(block))
    print("converted/interleved weights for tensorflow format")
    return new_net, weights


def load_weights(config_file, weights_file):
    config = convertConfigFile(open(config_file))
    size = get_size(weights_file)
    weights = open(weights_file, "rb")
    encoder, decoder, outputs, bytes_read = read_file(config, weights)
    print(
        f"bytes_read: {bytes_read}, original_size: {size}, final_position: {weights.tell()}")
    if (bytes_read != size):
        print("error: could not read the entire weights file")
    return encoder, decoder, outputs, bytes_read


config = "yolov3.cfg"
weights = "yolov3_416.weights"
encoder, decoder, outputs, _ = load_weights(config, weights)
encoder, weights = get_darknet53_tf_format(encoder[:])

# printing to get formatting
# print()
# for item in weights:
#     print()
#     for weight in item:
#         print(weight.shape)

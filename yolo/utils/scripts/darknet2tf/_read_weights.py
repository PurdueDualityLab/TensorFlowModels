"""
This file contains the code to parse DarkNet weight files.
"""

from __future__ import annotations

import io
import numpy as np
import os

from typing import Union

from .config_classes import *
from ..dn2dicts import convertConfigFile
from ...file_manager import PathABC, get_size, open_if_not_open
from ...errors import with_origin


def split_list(lst, i):
    return lst[:i], lst[i:]

def build_layer(layer_dict, file, net):
    """consturct layer and load weights from file"""
    layer = layer_builder[layer_dict['_type']].from_dict(net, layer_dict)

    if file is not None:
        bytes_read = layer.load_weights(file)
    else:
        bytes_read = 0

    return layer, bytes_read


def read_file(config, weights):
    """read the file and construct weights net list"""
    bytes_read = 0

    if weights is not None:
        major, minor, revision = read_n_int(3, weights)
        bytes_read += 12

        if ((major * 10 + minor) >= 2):
            print("64 seen")
            iseen = read_n_long(1, weights, unsigned=True)[0]
            bytes_read += 8
        else:
            print("32 seen")
            iseen = read_n_int(1, weights, unsigned=True)[0]
            bytes_read += 4

        print(f"major: {major}")
        print(f"minor: {minor}")
        print(f"revision: {revision}")
        print(f"iseen: {iseen}")

    full_net = []
    split_index = -1
    for i, layer_dict in enumerate(config):
        if layer_dict["_type"] != "decoder_encoder_split":
            try:
                layer, num_read = build_layer(layer_dict, weights, full_net)
            except Exception as e:
                raise ValueError(f"Cannot read weights for layer [#{i}]") from e
            full_net.append(layer)
        else:
            split_index = i

        bytes_read += num_read

    return full_net, split_index, bytes_read


def read_weights(config_file: Union[PathABC, io.TextIOBase],
                 weights_file: Union[PathABC, io.RawIOBase, io.BufferedIOBase]):
    """
    Parse the config and weights files and read the DarkNet layer's encoder,
    decoder, and output layers. The number of bytes in the file is also returned.

    Args:
        config_file: str, path to yolo config file from Darknet
        weights_file: str, path to yolo weights file from Darknet

    Returns:
        A tuple containing the following components:
            encoder: the encoder as a list of layer Config objects
            decoder: the decoder as a list of layer Config objects
            outputs: the outputs as a list of layer Config objects
    """
    size = get_size(weights_file)
    with open_if_not_open(config_file) as config, \
         open_if_not_open(weights_file, "rb") as weights:
        config = convertConfigFile(config)
        full_net, split_index, bytes_read = read_file(config, weights)
        encoder, decoder = split_list(full_net, split_index)
        print('encoder')
        for e in encoder:
            print(f"{e.w} {e.h} {e.c}\t{e}")
        print('decoder')
        for e in decoder:
            print(f"{e.w} {e.h} {e.c}\t{e}")
        print(
            f"bytes_read: {bytes_read}, original_size: {size}, final_position: {weights.tell()}")
    if (bytes_read != size):
        raise IOError('error reading weights file')
    return encoder, decoder

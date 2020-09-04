"""
This file contains the code to parse DarkNet weight files.
"""

from __future__ import annotations

import io
import numpy as np
import os

from typing import Union

from .config_classes import *
from ._darknet_model import DarkNetModel
from ..dn2dicts import convertConfigFile
from ...file_manager import PathABC, get_size, open_if_not_open
from ...errors import with_origin


def split_list(lst, i):
    return lst[:i], lst[i:]

def build_layer(layer_dict, file, net):
    """consturct layer and load weights from file"""

    layer = layer_builder[layer_dict['_type']].from_dict(net, layer_dict)

    bytes_read = 0
    if file is not None:
        bytes_read = layer.load_weights(file)

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

    full_net = DarkNetModel()
    cfg_iter = iter(config)
    try:
        layer, _ = build_layer(next(cfg_iter), weights, full_net)
        assert isinstance(layer, netCFG), 'A [net] config must be the first configuation section.'
        full_net.net_cfg = layer
    except Exception as e:
        raise ValueError(f"Cannot read network configuration") from e

    for i, layer_dict in enumerate(cfg_iter):
        try:
            layer, num_read = build_layer(layer_dict, weights, full_net)
        except Exception as e:
            raise ValueError(f"Cannot read weights for layer [#{i}]") from e
        print(f"{weights.tell()} {layer}")
        full_net.append(layer)
        bytes_read += num_read
    return full_net, bytes_read


def read_weights(config_file: Union[PathABC, io.TextIOBase],
                 weights_file: Union[PathABC, io.RawIOBase, io.BufferedIOBase]) -> DarkNetModel:
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
        full_net, bytes_read = read_file(config, weights)
        print('full net: ')
        for e in full_net:
            print(f"{e.w} {e.h} {e.c}\t{e}")
        print(
            f"bytes_read: {bytes_read}, original_size: {size}, final_position: {weights.tell()}")
    if (bytes_read != size):
        raise IOError('error reading weights file')
    return full_net

"""
This file contains the layers (Config objects) that are used by the Darknet
config file parser.

For more details on the layer types and layer parameters, visit https://github.com/AlexeyAB/darknet/wiki/CFG-Parameters-in-the-different-layers

Currently, the parser is incomplete and we can only guarantee that it works for
models in the YOLO family (YOLOv3 and older).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import numpy as np


class Config(ABC):
    """
    The base class for all layers that are used by the parser. Each subclass
    defines a new layer type. Most nodes correspond to distinct layers that
    appear in the final network. [net] corresponds to the input to the model.

    Each subclass must be a @dataclass and must have the following fields:
    ```{python}
        _type: str = None
        w: int = field(init=True, repr=True, default=0)
        h: int = field(init=True, repr=True, default=0)
        c: int = field(init=True, repr=True, default=0)
    ```

    These fields are used when linking different layers together, but weren't
    included in the Config class due to limitations in the dataclasses package.
    (w, h, c) will correspond to the different input dimensions of a DarkNet
    layer: the width, height, and number of channels.
    """

    @property
    @abstractmethod
    def shape(self):
        '''
        Output shape of the layer. The output must be a 3-tuple of ints
        corresponding to the the width, height, and number of channels of the
        output.

        Returns:
            A tuple corresponding to the output shape of the layer.
        '''
        return

    def load_weights(self, files):
        '''
        Load the weights for the current layer from a file.

        Arguments:
            files: Name of the DarkNet weights file

        Returns:
            the number of bytes read.
        '''
        return 0

    def get_weights(self):
        '''
        Returns:
            a list of Numpy arrays consisting of all of the weights that
            were loaded from the weights file
        '''
        return []

    @classmethod
    def from_dict(clz, prevlayer, layer_dict):
        '''
        Create a layer instance from the previous layer and a dictionary
        containing all of the parameters for the DarkNet layer. This is how
        linking is done by the parser.
        '''
        l = {
            "_type": layer_dict["_type"],
            "w": prevlayer.shape[0],
            "h": prevlayer.shape[1],
            "c": prevlayer.shape[2]}
        return clz(**l)


class _LayerBuilder(dict):
    """
    This class defines a registry for the layer builder in the DarkNet weight
    parser. It allows for syntactic sugar when registering Config subclasses to
    the parser.
    """

    def __getitem__(self, key):
        try:
            return super().__getitem__(key)
        except KeyError as e:
            raise KeyError(f"Unknown layer type: {layer_type}") from e

    def register(self, *layer_types: str):
        '''
        Register a parser node (layer) class with the layer builder.
        '''
        def decorator(clz):
            for layer_type in layer_types:
                self[layer_type] = clz
            return clz
        return decorator

layer_builder = _LayerBuilder()


class Building_Blocks(ABC):
    @property
    @abstractmethod
    def shape(self):
        return

    @abstractmethod
    def interleave_weights(self):
        return

    def get_weights(self):
        return []


@layer_builder.register('conv', 'convolutional')
@dataclass
class convCFG(Config):
    _type: str = None
    w: int = field(init=True, repr=True, default=0)
    h: int = field(init=True, repr=True, default=0)
    c: int = field(init=True, repr=True, default=0)

    size: int = field(init=True, repr=True, default=0)
    stride: int = field(init=True, repr=True, default=0)
    pad: int = field(init=True, repr=False, default=0)
    filters: int = field(init=True, repr=True, default=0)
    activation: str = field(init=True, repr=False, default='linear')
    groups: int = field(init=True, repr=False, default=1)
    batch_normalize: int = field(init=True, repr=False, default=0)

    nweights: int = field(repr=False, default=0)
    biases: np.array = field(repr=False, default=None)
    weights: np.array = field(repr=False, default=None)
    scales: np.array = field(repr=False, default=None)
    rolling_mean: np.array = field(repr=False, default=None)
    rolling_variance: np.array = field(repr=False, default=None)

    def __post_init__(self):
        self.pad = int(self.pad) if self.size != 1 else 0
        self.nweights = int((self.c / self.groups) *
                            self.filters * self.size * self.size)
        return

    @property
    def shape(self):
        w = len_width(self.w, self.size, self.pad, self.stride)
        h = len_width(self.h, self.size, self.pad, self.stride)
        return (w, h, self.filters)

    def load_weights(self, files):
        bytes_read = self.filters
        self.biases = read_n_floats(self.filters, files)

        if self.batch_normalize == 1:
            self.scales = read_n_floats(self.filters, files)
            self.rolling_mean = read_n_floats(self.filters, files)
            self.rolling_variance = read_n_floats(self.filters, files)
            bytes_read += self.filters * 3

        # used as a guide:
        # https://github.com/thtrieu/darkflow/blob/master/darkflow/dark/convolution.py
        weights = read_n_floats(self.nweights, files)
        self.weights = weights.reshape(
            self.filters, self.c, self.size, self.size).transpose([2, 3, 1, 0])
        bytes_read += self.nweights
        # print(f"weights shape: {self.weights.shape}")
        return bytes_read * 4

    def get_weights(self, printing=False):
        if printing:
            print("[weights, biases, biases, scales, rolling_mean, rolling_variance]")
        if self.batch_normalize:
            return [
                self.weights,
                self.biases,
                self.scales,
                self.rolling_mean,
                self.rolling_variance]
        else:
            return [self.weights, self.biases]

    @classmethod
    def from_dict(clz, prevlayer, layer_dict):
        layer_dict.update(
            {"w": prevlayer.shape[0], "h": prevlayer.shape[1], "c": prevlayer.shape[2]})
        return clz(**layer_dict)


@layer_builder.register('shortcut', 'route', 'yolo')
@dataclass
class placeCFG(Config):
    _type: str = None
    w: int = field(init=True, default=0)
    h: int = field(init=True, default=0)
    c: int = field(init=True, default=0)

    @property
    def shape(self):
        return (self.w, self.h, self.c)


@layer_builder.register('net')
@dataclass
class netCFG(placeCFG):
    @classmethod
    def from_dict(clz, prevlayer, layer_dict):
        l = {
            "_type": layer_dict["_type"],
            "w": layer_dict["width"],
            "h": layer_dict["height"],
            "c": layer_dict["channels"]}
        return clz(**l)


@layer_builder.register('upsample')
@dataclass
class upsampleCFG(Config):
    _type: str = None
    w: int = field(init=True, default=0)
    h: int = field(init=True, default=0)
    c: int = field(init=True, default=0)

    stride: int = field(init=True, default=2)

    @property
    def shape(self):
        return (self.stride * self.w, self.stride * self.h, self.c)

    @classmethod
    def from_dict(clz, prevlayer, layer_dict):
        l = {
            "w": prevlayer.shape[0],
            "h": prevlayer.shape[1],
            "c": prevlayer.shape[2],
            "stride": layer_dict["stride"]}
        return clz(**l)


def len_width(n, f, p, s):
    '''
    n: height or width
    f: kernels height or width
    p: padding
    s: strides height or width
    '''
    return int(((n + 2 * p - f) / s) + 1)


def len_width_up(n, f, p, s):
    '''
    n: height or width
    f: kernels height or width
    p: padding
    s: strides height or width
    '''
    return int(((n - 1) * s - 2 * p + (f - 1)) + 1)


def read_n_floats(n, bfile):
    """c style read n float 32"""
    return np.fromfile(bfile, 'f4', n)


def read_n_int(n, bfile, unsigned=False):
    """c style read n int 32"""
    dtype = '<u4' if unsigned else '<i4'
    return np.fromfile(bfile, dtype, n)


def read_n_long(n, bfile, unsigned=False):
    """c style read n int 64"""
    dtype = '<u8' if unsigned else '<i8'
    return np.fromfile(bfile, dtype, n)

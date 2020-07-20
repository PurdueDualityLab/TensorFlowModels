from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import numpy as np
from dn2dicts import *
import struct


class Config(ABC):
    # @abstractmethod
    def __init__(self, **kwargs):
        return

    @property
    @abstractmethod
    def shape(self):
        return

    @abstractmethod
    def load_weights(self):
        return

    @abstractmethod
    def get_weights(parameter_list):
        pass


class Building_Blocks(ABC):
    @abstractmethod
    def __init__(self):
        return

    @property
    @abstractmethod
    def shape(self):
        return

    @abstractmethod
    def interleave_weights(self):
        return

    @abstractmethod
    def get_weights(parameter_list):
        pass


@dataclass
class convCFG(Config):
    _type: str = None
    size: int = field(init=True, repr=True, default=0)
    stride: int = field(init=True, repr=True, default=0)
    pad: int = field(init=True, repr=False, default=0)
    filters: int = field(init=True, repr=True, default=0)
    activation: str = field(init=True, repr=False, default='linear')
    groups: int = field(init=True, repr=False, default=1)
    batch_normalize: int = field(init=True, repr=False, default=0)
    w: int = field(init=True, repr=True, default=0)
    h: int = field(init=True, repr=True, default=0)
    c: int = field(init=True, repr=True, default=0)

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
        self.biases = np.array(read_n_floats(self.filters, files))

        if self.batch_normalize == 1:
            self.scales = np.array(read_n_floats(self.filters, files))
            self.rolling_mean = np.array(read_n_floats(self.filters, files))
            self.rolling_variance = np.array(
                read_n_floats(self.filters, files))
            bytes_read += self.filters * 3

        # used as a guide:
        # https://github.com/thtrieu/darkflow/blob/master/darkflow/dark/convolution.py
        self.weights = np.array(read_n_floats(self.nweights, files))
        self.weights = self.weights.reshape(
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


@dataclass
class placeCFG(Config):
    _type: str = None
    w: int = field(init=True, default=0)
    h: int = field(init=True, default=0)
    c: int = field(init=True, default=0)

    @property
    def shape(self):
        return (self.w, self.h, self.c)

    def load_weights(self):
        return 0

    def get_weights(parameter_list):
        return 0


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

    def load_weights(self):
        return 0

    def get_weights(parameter_list):
        return 0


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
    return list(struct.unpack("f" * n, bfile.read(4 * n)))


def read_n_int(n, bfile, unsigned=False):
    """c style read n int 32"""
    if unsigned:
        return list(struct.unpack("<" + "i" * n, bfile.read(4 * n)))
    else:
        return list(struct.unpack("<" + "i" * n, bfile.read(4 * n)))


def read_n_long(n, bfile, unsigned=False):
    """c style read n int 64"""
    if unsigned:
        return list(struct.unpack("<" + "Q" * n, bfile.read(8 * n)))
    else:
        return list(struct.unpack("<" + "q" * n, bfile.read(8 * n)))

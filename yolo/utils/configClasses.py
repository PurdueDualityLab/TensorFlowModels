from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import numpy as np
from .dn2dicts import *


class Config(ABC):
    @property
    @abstractmethod
    def shape(self):
        return

    def load_weights(self):
        return 0

    def get_weights(self):
        return []


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
        self.biases = read_n_floats(self.filters, files)

        if self.batch_normalize == 1:
            self.scales = read_n_floats(self.filters, files)
            self.rolling_mean = read_n_floats(self.filters, files)
            self.rolling_variance = read_n_floats(self.filters, files)
            bytes_read += self.filters * 3

        # used as a guide:
        # https://github.com/thtrieu/darkflow/blob/master/darkflow/dark/convolution.py
        self.weights = read_n_floats(self.nweights, files)
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
    return np.frombuffer(bfile.read(4 * n), np.dtype(f'{n}f4'))


def read_n_int(n, bfile, unsigned=False):
    """c style read n int 32"""
    buffer = bfile.read(4 * n)
    if n == 1:
        n = ''
    if unsigned:
        return np.frombuffer(buffer, np.dtype(f'<{n}u4'))
    else:
        return np.frombuffer(buffer, np.dtype(f'<{n}i4'))


def read_n_long(n, bfile, unsigned=False):
    """c style read n int 64"""
    buffer = bfile.read(8 * n)
    if n == 1:
        n = ''
    if unsigned:
        return np.frombuffer(buffer, np.dtype(f'<{n}u8'))
    else:
        return np.frombuffer(buffer, np.dtype(f'<{n}i8'))

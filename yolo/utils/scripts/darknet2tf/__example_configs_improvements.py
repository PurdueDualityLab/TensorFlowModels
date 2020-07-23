# This package may be used in the future. Not in use yet. It is just an example
# right now.

from abc import ABC, abstractmethod
from dataclasses import dataclass, fields, InitVar
from typing import *
import warnings

negInfInt = -1000000000

class Context:
    pass

class CFG(ABC):
    @abstractmethod
    def to_tfconfig(self) -> dict: ...

    def _load_weights(self, files) -> int:
        return 0

    def to_object(self):
        return self._layer_class(self.to_tfconfig())

    @classmethod
    def _from_section(clz, context: Context, section: dict) -> 'CFG':
        section2 = section.copy()
        for field in fields(clz):
            name = field.name
            if name in section:
                value = section[name]
                if not isinstance(value, field.type):
                    section2[name] = field.type(value)
        return clz(_context=context, **section2)
    @classmethod
    def _tfconfig_from_section(clz, context: Context, section: dict) -> dict:
        return clz._from_section(context, section).to_tfconfig()
    @classmethod
    def _code_from_section(clz, context: Context, section: dict) -> str:
        ... # TODO

@dataclass
class ConvCFG(CFG):
    filters: int
    size: int
    groups: int = 1
    stride_x: int = -1
    stride_y: int = -1
    stride: InitVar[int] = 1
    dilation: int = 1
    antialiasing: int = 0
    pad: int = 0
    padding: int = 0
    activation: str = "logistic" #change?
    assisted_excitation: float = 0
    share_index: InitVar[int] = negInfInt
    batch_normalize: int = 0
    cbn: int = 0
    binary: int = 0
    xnor: int = 0
    bin_output: int = 0
    sway: int = 0
    rotate: int = 0
    stretch: int = 0
    stretch_sway: int = 0
    flipped: int = 0
    dot: int = 0
    angle: float = 15
    grad_centr: int = 0
    reverse: float = 0


    _context: InitVar[Context] = None
    _layer_class: ClassVar[type] = None

    def __post_init__(self, stride, share_index, _context):
        # read stride
        if self.stride_x < 1 or self.stride_y < 1:
            if stride is None:
                raise AssertionError('Stride is not defined')
            if self.stride_x < 1:
                self.stride_x = stride
            if self.stride_y < 1:
                self.stride_y = stride
        else:
            warnings.warn('stride definition overwriten by stride_x and stride_y')

        # if size is 1, dilation is 1
        if self.size == 1:
            self.dilation = 1
            warnings.warn('dilation definition overwriten by size')

        # if pad != 0, padding = size//2
        if self.pad != 0:
            self.padding = self.size // 2
            warnings.warn('padding definition overwriten by pad')

        # if cbn != 0, batch_normalize = 2
        if self.cbn != 0:
            self.batch_normalize = 2
            warnings.warn('batch_normalize definition overwriten by cbn')

        if self.sway + self.rotate + self.stretch + self.stretch_sway > 1:
            raise AssertionError("Error: should be used only 1 param: sway=1, rotate=1 or stretch=1 in the [convolutional] layer")
        if any((self.sway, self.rotate, self.stretch, self.stretch_sway)) and self.size == 1:
            raise AssertionError("Error: params (sway=1, rotate=1 or stretch=1) should be used only with size >=3 in the [convolutional] layer")

    def to_tfconfig(self) -> dict:
        return None

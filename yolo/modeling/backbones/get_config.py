from __future__ import annotations

from typing import List, NewType, Tuple

try:
    from typing import final as _final
except ImportError:
    # Weak shim for Python 3.7 and older
    def _final(f):
        return f

#RawConfig = NewType('RawConfig', Tuple[str, int, int, int, int, str, bool, bool])
#RawConfig.__doc__ = "(name, numberinblock, filters, kernal_size, padding, strides, downsample, output)"
RawConfig = Tuple[str, int, int, int, int, str, bool, bool]

@_final
class BlockConfig:
    def __init__(
            self,
            layer,
            reps,
            filters,
            kernel_size,
            strides,
            padding,
            downsample,
            output):
        '''
        get layer config to make code more readable

        Args:
            layer: string layer name
            reps: integer for the number of times to repeat block
            filters: integer for the filter for this layer, or the output depth
            kernel_size: integer or none, if none, it implies that the the building block handles this automatically. not a layer input
            downsample: boolean, to down sample the input width and height
            output: boolean, true if the layer is required as an output
        '''
        self.name = layer
        self.repititions = reps
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.downsample = downsample
        self.output = output
        return

    def __repr__(self):
        return f"layer: {self.name}, repititions: {self.repititions}, filters: {self.filters}, padding: {self.padding}, strides: {self.strides}, kernel size: {self.kernel_size}, downsample: {self.downsample}, route/output: {self.output}\n"

    @staticmethod
    def from_dicts(config: List[RawConfig]) -> List['BlockConfig']:
        """
        Convert SpineNet style configurations into `BlockConfig` objects.

        Arguments:
            config: A list containing the arguments for each of the blocks in the model

        Returns:
            A list of `BlockConfig` objects corresponding to each tuple in config
        """
        specs = []
        for layer in config:
            specs.append(BlockConfig(*layer))
        return specs


build_block_specs = BlockConfig.from_dicts

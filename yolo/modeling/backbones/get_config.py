class BlockConfig(object):
    def __init__(self, layer, reps, filters, kernel_size, strides, padding,
                 downsample, output, name):
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
        self.output_name = name
        return

    def __repr__(self):
        return f"layer: {self.name}, repititions: {self.repititions}, filters: {self.filters}, padding: {self.padding}, strides: {self.strides}, kernel size: {self.kernel_size}, downsample: {self.downsample}, route/output: {self.output}\n"


class CSPBlockConfig(object):
    def __init__(self, layer, stack, reps, bottleneck, filters, kernel_size,
                 strides, padding, activation, route, output_name):
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
        self.layer = layer
        self.stack = stack
        self.repetitions = reps
        self.bottleneck = bottleneck
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.route = route
        self.output_name = output_name
        return

    def __repr__(self):
        return f"layer: {self.layer}, stack: {self.stack}, repititions: {self.repetitions}, filters: {self.filters}, bottleneck: {self.bottleneck}, route: {self.route}, output: {self.output_name}\n"


def build_block_specs(config):
    specs = []
    for layer in config:
        specs.append(BlockConfig(*layer))
    return specs


def csp_build_block_specs(config):
    specs = []
    for layer in config:
        specs.append(CSPBlockConfig(*layer))
    return specs

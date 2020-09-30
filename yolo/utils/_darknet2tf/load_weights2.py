from collections import defaultdict
from yolo.modeling.building_blocks import DarkConv
from .config_classes import convCFG


def split_converter(lst, i, j=None):
    if j is None:
        return lst.data[:i], lst.data[i:j], lst.data[j:]
    return lst.data[:i], lst.data[i:]


def load_weights_backbone(model, net):
    convs = []
    for layer in net:
        if isinstance(layer, convCFG):
            convs.append(layer)

    for layer in model.layers:
        if isinstance(layer, DarkConv):
            cfg = convs.pop(0)
            layer.set_weights(cfg.get_weights())
        else:
            for sublayer in layer.submodules:
                if isinstance(sublayer, DarkConv):
                    cfg = convs.pop(0)
                    sublayer.set_weights(cfg.get_weights())


def load_weights_v4head(model, net):
    convs = []
    for layer in net:
        if isinstance(layer, convCFG):
            convs.append(layer)

    blocks = []
    for layer in model.layers:
        if isinstance(layer, DarkConv):
            blocks.append([layer])
        else:
            block = []
            for sublayer in layer.submodules:
                if isinstance(sublayer, DarkConv):
                    block.append(sublayer)
            if block:
                blocks.append(block)

    # 4 and 0 have the same shape
    remap = [4, 6, 0, 1, 7, 2, 3, 5]
    old_blocks = blocks
    blocks = [old_blocks[i] for i in remap]

    for block in blocks:
        for layer in block:
            cfg = convs.pop(0)
            print(cfg)#, layer.input_shape)
            layer.set_weights(cfg.get_weights())
        print()

    print(convs)

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
            #layer.set_weights(cfg.get_weights())
            print(cfg, layer.input_shape)
        else:
            for sublayer in layer.submodules:
                if isinstance(sublayer, DarkConv):
                    cfg = convs.pop(0)
                    #sublayer.set_weights(cfg.get_weights())
                    print(cfg, sublayer.input_shape)

    print(convs)

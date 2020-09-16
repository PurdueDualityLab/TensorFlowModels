from collections import defaultdict
from yolo.modeling.building_blocks import DarkConv
from .config_classes import convCFG

def load_weights_backbone(model, net):
    convs = []
    for layer in net:
        if isinstance(layer, convCFG):
            convs.append(layer)

    i = 0
    for sublayer in model.submodules:
        if isinstance(sublayer, DarkConv):
            i += 1

    print(i, len(convs))

    for layer in model.layers:
        if isinstance(layer, DarkConv):
            cfg = convs.pop(0)
            layer.set_weights(cfg.get_weights())
        else:
            for sublayer in layer.submodules:
                if isinstance(sublayer, DarkConv):
                    cfg = convs.pop(0)
                    sublayer.set_weights(cfg.get_weights())

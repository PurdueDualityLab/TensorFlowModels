from collections import defaultdict
from .config_classes import get_primitive_tf_layer_name

def load_weights_backbone(model, net):
    varz_grouped = defaultdict(dict)
    varz = model.variables[::-1]
    for var in varz:
        cid, name = get_primitive_tf_layer_name(var)

        super_layer = var.name.split("/", 3)
        weight_name = super_layer[-1]
        if len(super_layer) == 3:
            super_layer = ''
        else:
            super_layer = super_layer[0]
        cid.insert(0, super_layer)
        cid = tuple(cid)

        varz_grouped[cid][weight_name] = var

    varz_sorted = list(varz_grouped.items())
    for l in net:
        l.load_tf(varz_sorted)

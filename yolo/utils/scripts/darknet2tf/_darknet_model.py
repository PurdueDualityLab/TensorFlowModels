from . import config_classes as _conf
import collections

class DarkNetModel(collections.UserList):
    net_cfg : _conf.netCFG
    def to_keras(self):
        layers = []
        for cfg in self:
            layers.append(cfg._new_keras_layer(layers))
        return layers[-1]

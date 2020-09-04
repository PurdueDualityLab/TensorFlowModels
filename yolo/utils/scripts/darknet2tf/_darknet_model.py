from . import config_classes as _conf
import collections

class DarkNetModel(collections.UserList):
    def to_tf(self):
        tensors = []
        for cfg in self:
            tensors.append(cfg.to_tf(tensors))
        return tensors#[-1]

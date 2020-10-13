

import dataclasses
from official.modeling import hyperparams
from official.modeling.hyperparams import config_definitions as cfg

@dataclasses.dataclass
class YoloCFG(hyperparams.Config):
    @property
    def boxes(self):
        boxes = []
        for box in self._boxes:
            f = []
            for b in box.split(","):
                f.append(int(b.strip()))
            boxes.append(f)
        return boxes
    
    @boxes.setter 
    def boxes(self, box_list):
        setter = []
        for value in box_list:
            value = str(list(value))
            setter.append(value[1:-1])
        self._boxes = setter

@dataclasses.dataclass
class TaskConfig(cfg.TaskConfig):
    @property
    def input_size(self):
        if self._input_size == None:
            return [None, None, 3]
        else:
            return self._input_size
    
    @input_size.setter 
    def input_size(self, input_size):
        self._input_size = input_size
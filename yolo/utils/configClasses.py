import abc
import dataclass
import numpy as np

class Config(abc):
    # @abstractmethod
    def __init__(self, **kwargs):
        return

    @property
    @abstractmethod
    def shape(self):
        return

    @abstractmethod
    def load_weights(self):
        return

    @abstractmethod
    def get_weigths(parameter_list):
        pass

class Building_Blocks(abc):
    @abstractmethod
    def __init__(self):
        return
    
    @property
    @abstractmethod
    def shape(self):
        return

    @abstractmethod
    def interleave_weights(self):
        return

    @abstractmethod
    def get_weigths(parameter_list):
        pass

@dataclass
class convCFG(Config):
    size:int = field(init = True, repr = True, default = 0)
    stride:int = field(init = True, repr = True, default = 0)
    pad:int = field(init = True, repr = True, default = 0)
    n:int = field(init = True, repr = True, default = 0)
    activation:str = field(init = True, default = 'linear')
    groups:int = field(init = True, default = 0)
    batch_normalize:int = field(init = True, default = 0)
    w:int = field(init = True, default = 0)
    h:int = field(init = True, default = 0)
    c:int = field(init = True, default = 0)
    nweights:int = field(repr = True, default = 0)
    biases:np.array: = field(repr = True, default = None)
    weigths:np.array: = field(repr = True, default = None)
    scales:np.array: = field(repr = True, default = None)
    rolling_mean:np.array: = field(repr = True, default = None)
    rolling_variance:np.array: = field(repr = True, default = None)




    
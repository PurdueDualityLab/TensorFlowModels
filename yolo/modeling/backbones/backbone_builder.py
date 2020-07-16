import tensorflow as tf 
import tensorflow.keras as ks

from yolo.modeling.building_blocks import *
from yolo.modeling.backbones.get_config import build_block_specs
from yolo.modeling.backbones.darknet_53 import darknet53_config

class YOLO_builder(ks.Model):
    def __init__(self, layer_specs, **kwargs):
        super(YOLO_builder, self).__init__(**kwargs)
        self.layers = layer_specs
        return

    def _build_struct(self, net):
        endpoints = dict()
        for layer in self.layerspecs:
            continue
        return
    
    def _build_block(self):
        
        return
    
    def _connect_routes(self):
        return
    
block_specs = build_block_specs(darknet53_config) # current model input



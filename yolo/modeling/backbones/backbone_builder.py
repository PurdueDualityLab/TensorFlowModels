import tensorflow as tf 
import tensorflow.keras as ks

from yolo.modeling.building_blocks import *
from yolo.modeling.backbones import *

#temporary
darknet53 = [
    ("DarkConv", 1, 32, 3, False, False, False), #1
    ("DarkRes", 1, 64, None, True, False, False), #3
    ("DarkRes", 1, 128, None, True, False, False), #3
    ("DarkRes", 1, 128, None, False, False, False), #2
    ("DarkRes", 1, 256, None, True, False, False), #3
    ("DarkRes", 7, 256, None, False, True, False), #14 route 61 last shortcut or last conv
    ("DarkRes", 1, 512, None, True, False, False), #3
    ("DarkRes", 7, 512, None, False, True, False), #14 route 61 last shortcut or last conv
    ("DarkRes", 1, 1024, None, True, False, False), #3
    ("DarkRes", 3, 1024, None, False, False, True), #6  #route  
 ] #52 layers 


class YOLO_builder(ks.Model):
    def __init__(self, **kwargs):
        super(YOLO_builder, self).__init__(**kwargs)
        return

    def _build_struct(self):
        endpoints = dict()
        return
    
    def _connect_routes(self):
        return
    


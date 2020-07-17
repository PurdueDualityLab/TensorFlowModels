import tensorflow as tf 
import tensorflow.keras as ks
from yolo.modeling.backbones.backbone_builder import backbone_builder
from yolo.modeling.backbones.detection_heads import *

class DarkNet53(ks.Model):
    def __init__(self, classes = 1000):
        self.back_bone = backbone_builder("darknet53")
        self.head = DarkNet53_head.classification(classes)
        super(DarkNet53, self).__init__()
        return
    
    def call(self, inputs):
        out_dict = self.back_bone(inputs)
        x = list(out_dict.keys())[-1]
        return self.head(x)
        

class Yolov3():
    def __init__(self):
        pass

class Yolov3_tiny():
    def __init__(self):
        pass

class Yolov3_spp():
    def __init__(self):
        pass

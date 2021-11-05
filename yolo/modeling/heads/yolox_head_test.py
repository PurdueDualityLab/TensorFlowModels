
import tensorflow as tf
from tensorflow.keras import Model
from yolo.modeling.backbones import darknet
from yolo.modeling.decoders.yolo_decoder import YoloFPN
from yolo.modeling.heads.yolox_head import YOLOXHead
# from yolo.modeling.heads.yolox_head import YOLOXLoss
class YOLOX(Model):
    """
    YOLOX model module. The module list is defined by create_yolov3_modules function.
    The network returns loss values from three YOLO layers during training
    and detection results during test.
    """

    def __init__(self, num_classes=80,backbone=None, head=None,act="silu"):
        super().__init__()
        if backbone is None:
            backbone = YoloFPN(activation=act)
        if head is None:
            head = YOLOXHead(num_classes,act=act)

        self.backbone = backbone
        self.head = head
        # self.yolo_loss=YOLOXLoss(num_classes=num_classes)

    def call(self,inputs,**kwargs):
        # fpn output content features of [dark3, dark4, dark5]
        # x=self.backbone(inputs)
        outputs=self.head(inputs)
        return outputs

if __name__=="__main__":
    from tensorflow.keras import Input
    from tensorflow.keras import Model
    yolox = YOLOXHead(num_classes=80)
    h, w = (640,640)
    inputs = Input(shape=(h, w,3))
    labels = Input(shape=(50, 6))
    outputs = yolox(inputs)
    model=Model(inputs,outputs)
    model.summary()

import tensorflow as tf
from tensorflow.keras import Model
from yolo.modeling.backbones import darknet
from yolo.modeling.decoders import yolo_decoder
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
            backbone = darknet(act=act)
        if head is None:
            head = YOLOXHead(num_classes,act=act)

        self.backbone = backbone
        self.head = head
        # self.yolo_loss=YOLOXLoss(num_classes=num_classes)

    def call(self,inputs,**kwargs):
        # fpn output content features of [dark3, dark4, dark5]
        x=self.backbone(inputs)
        outputs=self.head(x)
        return outputs

if __name__=="__main__":
    from tensorflow.keras import Input
    from tensorflow.keras import Model
    backbone = darknet(depth=0.67, width=0.75)
    head = YOLOXHead(num_classes=80, width=0.75)
    yolo = YOLOX(num_classes=80,backbone=backbone, head=head)
    h, w = (640,640)
    inputs = Input(shape=(h, w,3))
    # outputs=backbone(inputs)
    # model=Model(inputs,outputs)
    # model.summary()
    labels = Input(shape=( 50, 6))
    outputs = yolo(inputs)
    model=Model(inputs,outputs)
    model.summary()
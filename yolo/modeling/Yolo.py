import tensorflow as tf
import tensorflow.keras as ks
from typing import *

from yolo.modeling.backbones.Darknet import Darknet
from yolo.modeling.model_heads._Yolov4Head import Yolov4Head
from yolo.modeling.model_heads._Yolov3Head import Yolov3Head
from yolo.modeling.model_heads._Yolov4Neck import Yolov4Neck
from yolo.modeling.building_blocks import YoloLayer


from yolo.utils import DarkNetConverter
from yolo.utils.file_manager import download
from yolo.utils._darknet2tf.load_weights import split_converter
from yolo.utils._darknet2tf.load_weights2 import load_weights_backbone
from yolo.utils._darknet2tf.load_weights2 import load_weights_v4head
from yolo.utils._darknet2tf.load_weights import load_weights_dnBackbone
from yolo.utils._darknet2tf.load_weights import load_weights_dnHead

class Yolo(ks.Model):
    def __init__(
            self,
            backbone = None,
            neck = None,
            head = None,
            decoder = None,
            **kwargs):
        super().__init__(**kwargs)
        #model components
        self._backbone = backbone
        self._neck = neck
        self._head = head
        self._decoder = decoder

        return

    def build(self, input_shape):
        self._backbone.build(input_shape)
        nshape = self._backbone.output_shape
        if self._neck != None:
            self.neck.build(nshape)
            nshape = self._neck.output_shape
        self._head.build(nshape)
        super().build(input_shape)
        return 

    def call(self, inputs, training=False):
        maps = self._backbone(inputs)

        if self._neck != None:
            maps = self._neck(maps)

        raw_predictions = self._head(maps)
        if training:
            return {"raw_output": raw_predictions}
        else:
            predictions = self._decoder(raw_predictions)
            predictions.update({"raw_output": raw_predictions})
            return predictions
    
    @property
    def backbone(self):
        return self._backbone
    
    @property
    def neck(self):
        return self._neck
    
    @property
    def head(self):
        return self._head
    
    @property
    def decoder(self):
        return self._decoder
    
    def load_weights_from_dn(self,
                            dn2tf_backbone=True,
                            dn2tf_head=True,
                            config_file=None,
                            weights_file=None):
        if not self.built:
            self.build([None, None, None, 3])

        if dn2tf_backbone or dn2tf_head:
            if self.backbone.name == "cspdarknet53":
                if config_file == None:
                    config_file = download('yolov4.cfg')
                if weights_file  == None:
                    weights_file = download('yolov4.weights')
            elif self.backbone.name == "darknet53":
                if self.head.name == "regular":
                    if config_file == None:
                        config_file = download('yolov3.cfg')
                    if weights_file  == None:
                        weights_file = download('yolov3.weights')
                elif self.head.name == "spp":
                    if config_file == None:
                        config_file = download('yolov3-spp.cfg')
                    if weights_file  == None:
                        weights_file = download('yolov3-spp.weights')
            elif self.backbone.name == "cspdarknettiny":
                if config_file == None:
                    config_file = download('yolov4-tiny.cfg')
                if weights_file  == None:
                    weights_file = download('yolov4-tiny.weights')
            elif self.backbone.name == "darknettiny":
                if config_file == None:
                    config_file = download('yolov3-tiny.cfg')
                if weights_file  == None:
                    weights_file = download('yolov3-tiny.weights')
            list_encdec = DarkNetConverter.read(config_file, weights_file)
            splits = self.backbone.splits

            if len(splits.keys()) == 1:
                encoder, decoder = split_converter(list_encdec, splits["backbone_split"])
                neck = None
            elif len(splits.keys()) == 2:
                encoder, neck, decoder = split_converter(list_encdec, splits["backbone_split"], splits["neck_split"])


            if dn2tf_backbone: 
                load_weights_backbone(self.backbone, encoder) 
                self.backbone.trainable = False
        
            if dn2tf_head:
                if neck is not None:
                    load_weights_backbone(self._neck, neck)
                    self._neck.trainable = False
                
                if "csp" in self.backbone.name:
                    if self.head.name == "tinyv4":
                        load_weights_v4head(self._head, decoder, (3, 5, 0, 1, 6, 2, 4, 7))
                    else:
                        load_weights_v4head(self._head, decoder, (0, 4, 1, 2, 3))
                else:
                    load_weights_dnHead(self._head, decoder) 
                self._head.trainable = False



from yolo.modeling.backbones.Darknet import build_darknet
from official.vision.beta.modeling.backbones import factory 
from official.core import registry

def build_yolo_neck(input_specs, model_config, l2_regularization):
    if model_config.neck == None:
        return None

    model = Yolov4Neck(model = model_config.neck.name, input_shape = input_specs.shape, kernel_regularizer = l2_regularization)
    return model

def build_yolo_head(input_specs, model_config, parent_config, l2_regularization):
    box_cfg = parent_config.anchors.get()
    num_boxes = len(box_cfg.boxes)
    if model_config.head.version == "v3" or model_config.head.name == "tinyv4":
        model = Yolov3Head(classes = parent_config.num_classes, boxes = num_boxes, model = model_config.head.name, input_shape = input_specs.shape, kernel_regularizer = l2_regularization)
    elif model_config.head.version == "v4":
        model = Yolov4Head(classes = parent_config.num_classes, boxes = num_boxes, model = model_config.head.name, input_shape = input_specs.shape, kernel_regularizer = l2_regularization)
    return model

def build_yolo_decoder(model_config):
    anchor_cfg = model_config.anchors.get()
    model = YoloLayer(
        masks = anchor_cfg.masks.as_dict(),
        anchors = anchor_cfg.boxes,
        thresh = model_config.decoder.iou_thresh,
        cls_thresh = model_config.decoder.class_thresh,
        max_boxes = model_config.decoder.max_boxes,
        path_scale = anchor_cfg.path_scales.as_dict(),
        scale_xy = anchor_cfg.x_y_scales.as_dict(),
        use_nms = model_config.decoder.use_nms
    )
    return model

def build_yolo_default_loss(model_config):
    from yolo.modeling.functions.yolo_loss import Yolo_Loss
    loss_dict = {}
    anchors = model_config.anchors.get()
    masks = anchors.masks.as_dict()
    path_scales = anchors.path_scales.as_dict()
    x_y_scales = anchors.x_y_scales.as_dict()
    for key in masks.keys():
        loss_dict[key] = Yolo_Loss(classes = model_config.num_classes,
                                    anchors = anchors.boxes,
                                    ignore_thresh = model_config.decoder.ignore_thresh,
                                    loss_type = model_config.decoder.loss_type,
                                    path_key = key,
                                    mask = masks[key],
                                    scale_anchors = path_scales[key],
                                    scale_x_y = x_y_scales[key],
                                    use_tie_breaker = model_config.decoder.use_tie_breaker)
    return loss_dict

def build_yolo(input_specs, model_config, l2_regularization):
    print(model_config.as_dict())
    print(input_specs)
    print(l2_regularization) 

    base_config = model_config.base.get()
    backbone = factory.build_backbone(input_specs, base_config, l2_regularization)
    neck = build_yolo_neck(input_specs, base_config, l2_regularization)
    head = build_yolo_head(input_specs, base_config, model_config, l2_regularization)

    decoder = build_yolo_decoder(model_config)
    model = Yolo(backbone = backbone, neck = neck, head = head, decoder = decoder)
    model.build(input_specs.shape)

    losses = build_yolo_default_loss(model_config)

    return model, losses




import tensorflow as tf
import tensorflow.keras as ks
from typing import *

from yolo.modeling.backbones.Darknet import Darknet
from yolo.modeling.model_heads.YoloDecoder import YoloDecoder
from yolo.modeling.building_blocks import YoloLayer



class Yolo(ks.Model):
    def __init__(self,
                 backbone=None,
                 neck=None,
                 head=None,
                 decoder=None,
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


from yolo.modeling.backbones.Darknet import build_darknet
from official.vision.beta.modeling.backbones import factory
from official.core import registry


def build_yolo_decoder(input_specs, model_config, parent_config, l2_regularization):

    model = YoloDecoder(classes=model_config.num_classes,
                        boxes_per_level=3,
                        embed_spp=False,
                        embed_fpn=False,
                        max_level_process_len=None,
                        kernel_regularizer=l2_regularization,
                        path_process_len=6)
    return model


def build_yolo_filter(model_config):
    anchor_cfg = model_config.anchors.get()
    model = YoloLayer(masks=anchor_cfg.masks.as_dict(),
                      anchors=anchor_cfg.boxes,
                      thresh=model_config.decoder.iou_thresh,
                      cls_thresh=model_config.decoder.class_thresh,
                      max_boxes=model_config.decoder.max_boxes,
                      path_scale=anchor_cfg.path_scales.as_dict(),
                      scale_xy=anchor_cfg.x_y_scales.as_dict(),
                      use_nms=model_config.decoder.use_nms)
    return model


def build_yolo_default_loss(model_config):
    from yolo.modeling.functions.yolo_loss import Yolo_Loss
    loss_dict = {}
    anchors = model_config.anchors.get()
    masks = anchors.masks.as_dict()
    path_scales = anchors.path_scales.as_dict()
    x_y_scales = anchors.x_y_scales.as_dict()
    for key in masks.keys():
        loss_dict[key] = Yolo_Loss(
            classes=model_config.num_classes,
            anchors=anchors.boxes,
            ignore_thresh=model_config.decoder.ignore_thresh,
            loss_type=model_config.decoder.loss_type,
            path_key=key,
            mask=masks[key],
            scale_anchors=path_scales[key],
            scale_x_y=x_y_scales[key],
            use_tie_breaker=model_config.decoder.use_tie_breaker)
    return loss_dict


def build_yolo(input_specs, model_config, l2_regularization):
    print(model_config.as_dict())
    print(input_specs)
    print(l2_regularization)

    base_config = model_config.base.get()
    backbone = factory.build_backbone(input_specs, base_config, l2_regularization)
    head = build_yolo_decoder(input_specs, base_config, model_config, l2_regularization)
    filter = build_yolo_filter(model_config)

    model = Yolo(backbone=backbone, head=head, decoder=filter)
    model.build(input_specs.shape)

    losses = build_yolo_default_loss(model_config)

    return model, losses

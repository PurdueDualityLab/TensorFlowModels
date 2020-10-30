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
                 decoder=None,
                 filter=None,
                 **kwargs):
        super().__init__(**kwargs)
        #model components
        self._backbone = backbone
        self._neck = neck
        self._decoder = decoder
        self._filter = filter

        return

    def build(self, input_shape):
        self._backbone.build(input_shape)
        nshape = self._backbone.output_shape
        if self._neck != None:
            self.neck.build(nshape)
            nshape = self._neck.output_shape
        self._decoder.build(nshape)
        super().build(input_shape)
        return

    def call(self, inputs, training=False):
        maps = self._backbone(inputs)

        if self._neck != None:
            maps = self._neck(maps)

        raw_predictions = self._decoder(maps)
        if training:
            return {"raw_output": raw_predictions}
        else:
            predictions = self._filter(raw_predictions)
            predictions.update({"raw_output": raw_predictions})
            return predictions

    @property
    def backbone(self):
        return self._backbone

    @property
    def neck(self):
        return self._neck

    @property
    def decoder(self):
        return self._decoder

    @property
    def filter(self):
        return self._filter


from yolo.modeling.backbones.Darknet import build_darknet
from official.vision.beta.modeling.backbones import factory
from official.core import registry


def build_yolo_decoder(input_specs, model_config, l2_regularization):

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
                      thresh=model_config.filter.iou_thresh,
                      cls_thresh=model_config.filter.class_thresh,
                      max_boxes=model_config.filter.max_boxes,
                      path_scale=anchor_cfg.path_scales.as_dict(),
                      scale_xy=anchor_cfg.x_y_scales.as_dict(),
                      use_nms=model_config.filter.use_nms)
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
            ignore_thresh=model_config.filter.ignore_thresh,
            loss_type=model_config.filter.loss_type,
            path_key=key,
            mask=masks[key],
            scale_anchors=path_scales[key],
            scale_x_y=x_y_scales[key],
            use_tie_breaker=model_config.filter.use_tie_breaker)
    return loss_dict


def build_yolo(input_specs, model_config, l2_regularization):
    print(model_config.as_dict())
    print(input_specs)
    print(l2_regularization)

    backbone = factory.build_backbone(input_specs, model_config, l2_regularization)
    decoder = build_yolo_decoder(input_specs, model_config, l2_regularization)
    filter = build_yolo_filter(model_config)

    model = Yolo(backbone=backbone, decoder=decoder, filter=filter)
    model.build(input_specs.shape)

    losses = build_yolo_default_loss(model_config)

    return model, losses

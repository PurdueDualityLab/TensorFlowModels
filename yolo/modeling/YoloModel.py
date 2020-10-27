import tensorflow as tf
import tensorflow.keras as ks
from typing import *

from yolo.modeling.backbones.Darknet import Darknet
from yolo.modeling.model_heads.YoloDecoder import YoloDecoder
from yolo.modeling.building_blocks import YoloLayer

from yolo.utils import DarkNetConverter
from yolo.utils.file_manager import download
from yolo.utils._darknet2tf.load_weights import split_converter
from yolo.utils._darknet2tf.load_weights2 import load_weights_backbone
from yolo.utils._darknet2tf.load_weights2 import load_head
from yolo.utils._darknet2tf.load_weights import load_weights_dnBackbone
from yolo.utils._darknet2tf.load_weights import load_weights_dnHead


class Yolo(ks.Model):
    def __init__(self,
                 classes=80,
                 boxes_per_level=3,
                 boxes="default",
                 min_level=None,
                 max_level=5,
                 model_version="v4",
                 model_type="regular",
                 use_nms=True,
                 **kwargs):
        super().__init__(**kwargs)

        self._model_verison = model_version
        self._model_type = model_type
        self._use_nms = use_nms

        if boxes == "default":
            if model_type == "tiny":
                self._boxes = [(10, 14), (23, 27), (37, 58), (81, 82),
                               (135, 169), (344, 319)]
            elif model_version == "v4":
                self._boxes = [(12, 16), (19, 36), (40, 28), (36, 75),
                               (76, 55), (72, 146), (142, 110), (192, 243),
                               (459, 401)]
            else:
                self._boxes = [(10, 13), (16, 30), (33, 23), (30, 61),
                               (62, 45), (59, 119), (116, 90), (156, 198),
                               (373, 326)]

        if model_version == "v4":
            if model_type == "tiny":
                model_id = "cspdarknettiny"
            else:
                model_id = "cspdarknet53"
        else:
            if model_type == "tiny":
                model_id = "darknettiny"
            else:
                model_id = "darknet53"
        self._backbone = Darknet(model_id=model_id,
                                 max_level=max_level,
                                 min_level=min_level)

        if model_type == "tiny":
            self._head = YoloDecoder(classes=80,
                                     boxes_per_level=3,
                                     embed_spp=False,
                                     embed_fpn=False,
                                     max_level_process_len=2,
                                     path_process_len=1)
        elif model_version == "v4":
            self._head = YoloDecoder(classes=80,
                                     boxes_per_level=3,
                                     embed_spp=False,
                                     embed_fpn=True,
                                     max_level_process_len=None,
                                     path_process_len=6)
        elif model_type == "spp":
            self._head = YoloDecoder(classes=80,
                                     boxes_per_level=3,
                                     embed_spp=True,
                                     embed_fpn=False,
                                     max_level_process_len=None,
                                     path_process_len=6)
        else:
            self._head = YoloDecoder(classes=80,
                                     boxes_per_level=3,
                                     embed_spp=False,
                                     embed_fpn=False,
                                     max_level_process_len=None,
                                     path_process_len=6)
        return

    def build(self, input_shape):
        self._backbone.build(input_shape)
        nshape = self._backbone.output_shape
        self._head.build(nshape)
        masks, path_scales, x_y_scales = self._head.get_loss_attributes()

        self._decoder = YoloLayer(masks=masks,
                                  anchors=self._boxes,
                                  thresh=0.5,
                                  cls_thresh=0.5,
                                  max_boxes=200,
                                  path_scale=path_scales,
                                  scale_xy=x_y_scales,
                                  use_nms=self._use_nms)

        super().build(input_shape)
        return

    def call(self, inputs, training=False):
        maps = self._backbone(inputs)
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
                if weights_file == None:
                    weights_file = download('yolov4.weights')
            elif self.backbone.name == "darknet53":
                if not self.head._embed_spp:
                    if config_file == None:
                        config_file = download('yolov3.cfg')
                    if weights_file == None:
                        weights_file = download('yolov3.weights')
                else:
                    if config_file == None:
                        config_file = download('yolov3-spp.cfg')
                    if weights_file == None:
                        weights_file = download('yolov3-spp.weights')
            elif self.backbone.name == "cspdarknettiny":
                if config_file == None:
                    config_file = download('yolov4-tiny.cfg')
                if weights_file == None:
                    weights_file = download('yolov4-tiny.weights')
            elif self.backbone.name == "darknettiny":
                if config_file == None:
                    config_file = download('yolov3-tiny.cfg')
                if weights_file == None:
                    weights_file = download('yolov3-tiny.weights')
            list_encdec = DarkNetConverter.read(config_file, weights_file)
            splits = self.backbone.splits

            if len(splits.keys()) == 1:
                encoder, decoder = split_converter(list_encdec,
                                                   splits["backbone_split"])
                neck = None
            elif len(splits.keys()) == 2:
                encoder, neck, decoder = split_converter(
                    list_encdec, splits["backbone_split"],
                    splits["neck_split"])

            if dn2tf_backbone:
                load_weights_backbone(self.backbone, encoder)
                self.backbone.trainable = False

            if dn2tf_head:
                if neck is not None:
                    print(self.head.neck)
                    load_weights_backbone(self.head.neck, neck)
                    self.head.neck.trainable = False
                load_head(self.head.head,
                          decoder,
                          out_conv=self.head.output_depth)
                self.head.head.trainable = False


if __name__ == "__main__":
    model = Yolo(model_version="v3", model_type="regular")
    model.build([None, None, None, 3])
    model.load_weights_from_dn()
    model.summary()

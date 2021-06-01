from official.core import registry
import tensorflow as tf
import tensorflow.keras as ks
from typing import *

from yolo.configs import yolo

from official.vision.beta.modeling.backbones import factory
from yolo.modeling.backbones.darknet import build_darknet
from yolo.modeling.backbones.darknet import Darknet
from yolo.modeling.decoders.yolo_decoder import YoloDecoder
from yolo.modeling.heads.yolo_head import YoloHead
from yolo.modeling.layers.detection_generator import YoloLayer

# static base Yolo Models that do not require configuration
# similar to a backbone model id.

# this is done greatly simplify the model config
# the structure is as follows. model version, {v3, v4, v#, ... etc}
# the model config type {regular, tiny, small, large, ... etc}
YOLO_MODELS = {
    "v4":
        dict(
            regular=dict(
                embed_spp=False,
                use_fpn=True,
                max_level_process_len=None,
                path_process_len=6),
            tiny=dict(
                embed_spp=False,
                use_fpn=False,
                max_level_process_len=2,
                path_process_len=1),
            csp=dict(
                embed_spp=False,
                use_fpn=True,
                max_level_process_len=None,
                csp_stack=5,
                fpn_depth=5,
                path_process_len=6),
            csp_large=dict(
                embed_spp=False,
                use_fpn=True,
                max_level_process_len=None,
                csp_stack=7,
                fpn_depth=7,
                path_process_len=8,
                fpn_filter_scale=2),
        ),
    "v3":
        dict(
            regular=dict(
                embed_spp=False,
                use_fpn=False,
                max_level_process_len=None,
                path_process_len=6),
            tiny=dict(
                embed_spp=False,
                use_fpn=False,
                max_level_process_len=2,
                path_process_len=1),
            spp=dict(
                embed_spp=True,
                use_fpn=False,
                max_level_process_len=2,
                path_process_len=1),
        ),
}


class Yolo(ks.Model):
  """The YOLO model class."""

  def __init__(self,
               backbone=None,
               decoder=None,
               head=None,
               filter=None,
               **kwargs):
    """Detection initialization function.
    Args:
      backbone: `tf.keras.Model` a backbone network.
      decoder: `tf.keras.Model` a decoder network.
      head: `RetinaNetHead`, the RetinaNet head.
      filter: the detection generator.
      **kwargs: keyword arguments to be passed.
    """
    super(Yolo, self).__init__(**kwargs)

    self._config_dict = {
        'backbone': backbone,
        'decoder': decoder,
        'head': head,
        'filter': filter
    }

    # model components
    self._backbone = backbone
    self._decoder = decoder
    self._head = head
    self._filter = filter
    return

  def call(self, inputs, training=False):
    maps = self._backbone(inputs)
    decoded_maps = self._decoder(maps)
    raw_predictions = self._head(decoded_maps)
    if training:
      return {"raw_output": raw_predictions}
    else:
      # Post-processing.
      predictions = self._filter(raw_predictions)
      predictions.update({"raw_output": raw_predictions})
      return predictions

  @property
  def backbone(self):
    return self._backbone

  @property
  def decoder(self):
    return self._decoder

  @property
  def head(self):
    return self._head

  @property
  def filter(self):
    return self._filter

  def get_config(self):
    return self._config_dict

  @classmethod
  def from_config(cls, config):
    return cls(**config)


def build_yolo_decoder(input_specs, model_config: yolo.Yolo, l2_regularization):
  activation = model_config.decoder_activation if model_config.decoder_activation != "same" else model_config.norm_activation.activation
  subdivisions = 1

  if model_config.decoder.version is None:  # custom yolo
    model = YoloDecoder(
        input_specs,
        embed_spp=model_config.decoder.embed_spp,
        use_fpn=model_config.decoder.use_fpn,
        fpn_depth=model_config.decoder.fpn_depth,
        path_process_len=model_config.decoder.path_process_len,
        max_level_process_len=model_config.decoder.max_level_process_len,
        xy_exponential=model_config.decoder.xy_exponential,
        activation=activation,
        subdivisions=subdivisions,
        use_spatial_attention=model_config.use_sam,
        use_sync_bn=model_config.norm_activation.use_sync_bn,
        norm_momentum=model_config.norm_activation.norm_momentum,
        norm_epsilon=model_config.norm_activation.norm_epsilon,
        kernel_regularizer=l2_regularization)
    return model

  if model_config.decoder.type == None:
    model_config.decoder.type = "regular"

  if model_config.decoder.version not in YOLO_MODELS.keys():
    raise Exception(
        f"unsupported model version please select from {v3, v4}, \n\n or specify a custom decoder config using YoloDecoder in you yaml"
    )

  if model_config.decoder.type not in YOLO_MODELS[
      model_config.decoder.version].keys():
    raise Exception(
        f"unsupported model type please select from {YOLO_MODELS[model_config.decoder.version].keys()}, \n\n or specify a custom decoder config using YoloDecoder in you yaml"
    )

  base_model = YOLO_MODELS[model_config.decoder.version][
      model_config.decoder.type]
  base_dict = dict(
      activation=activation,
      subdivisions=subdivisions,
      use_spatial_attention=model_config.use_sam,
      use_sync_bn=model_config.norm_activation.use_sync_bn,
      norm_momentum=model_config.norm_activation.norm_momentum,
      norm_epsilon=model_config.norm_activation.norm_epsilon,
      kernel_regularizer=l2_regularization)

  base_model.update(base_dict)

  model = YoloDecoder(input_specs, **base_model)

  return model


def build_yolo_filter(model_config: yolo.Yolo, decoder: YoloDecoder, masks,
                      xy_scales, path_scales):
  model = YoloLayer(
      masks=masks,
      classes=model_config.num_classes,
      anchors=model_config._boxes,
      iou_thresh=model_config.filter.iou_thresh,
      nms_thresh=model_config.filter.nms_thresh,
      max_boxes=model_config.filter.max_boxes,
      nms_type=model_config.filter.nms_type,
      path_scale=path_scales,
      scale_xy=xy_scales,
      label_smoothing=model_config.filter.label_smoothing,
      pre_nms_points=model_config.filter.pre_nms_points,
      use_reduction_sum=model_config.filter.use_reduction_sum,
      truth_thresh=model_config.filter.truth_thresh.as_dict(),
      loss_type=model_config.filter.loss_type.as_dict(),
      max_delta=model_config.filter.max_delta.as_dict(),
      new_cords=model_config.filter.new_cords.as_dict(),
      iou_normalizer=model_config.filter.iou_normalizer.as_dict(),
      cls_normalizer=model_config.filter.cls_normalizer.as_dict(),
      obj_normalizer=model_config.filter.obj_normalizer.as_dict(),
      ignore_thresh=model_config.filter.ignore_thresh.as_dict(),
      objectness_smooth=model_config.filter.objectness_smooth.as_dict())
  return model


def build_yolo_head(input_specs, model_config: yolo.Yolo, l2_regularization):
  if hasattr(model_config, "subdivisions"):
    subdivisions = model_config.subdivisions
  else:
    subdivisions = 1

  head = YoloHead(
      min_level=model_config.min_level,
      max_level=model_config.max_level,
      classes=model_config.num_classes,
      boxes_per_level=model_config.boxes_per_scale,
      norm_momentum=model_config.norm_activation.norm_momentum,
      norm_epsilon=model_config.norm_activation.norm_epsilon,
      kernel_regularizer=l2_regularization)

  return head


def build_yolo(input_specs, model_config, l2_regularization, masks, xy_scales,
               path_scales):
  print(model_config.as_dict())
  print(input_specs)
  print(l2_regularization)

  backbone = factory.build_backbone(input_specs, model_config,
                                    l2_regularization)
  decoder = build_yolo_decoder(backbone.output_specs, model_config,
                               l2_regularization)
  head = build_yolo_head(decoder.output_specs, model_config, l2_regularization)
  filter = build_yolo_filter(model_config, head, masks, xy_scales, path_scales)

  model = Yolo(backbone=backbone, decoder=decoder, head=head, filter=filter)
  model.build(input_specs.shape)
  model.decoder.summary()
  model.summary()

  losses = filter.losses
  return model, losses

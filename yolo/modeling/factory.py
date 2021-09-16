from official.vision.beta.modeling.backbones import factory
from yolo.modeling.decoders.yolo_decoder import YoloDecoder
from yolo.modeling.heads.yolo_head import YoloHead
from yolo.modeling.layers.detection_generator import YoloLayer
from yolo.modeling.backbones.darknet import build_darknet

from yolo.modeling import yolo_model
from yolo.configs import yolo


def build_yolo_decoder(input_specs, model_config: yolo.Yolo, l2_regularization):
  activation = (
      model_config.decoder.activation
      if model_config.decoder.activation != "same" else
      model_config.norm_activation.activation)

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
        use_spatial_attention=model_config.use_sam,
        use_sync_bn=model_config.norm_activation.use_sync_bn,
        norm_momentum=model_config.norm_activation.norm_momentum,
        norm_epsilon=model_config.norm_activation.norm_epsilon,
        kernel_regularizer=l2_regularization)
    return model

  if model_config.decoder.type == None:
    model_config.decoder.type = "regular"

  if model_config.decoder.version not in yolo_model.YOLO_MODELS.keys():
    raise Exception(
        "unsupported model version please select from {v3, v4}, \n\n \
        or specify a custom decoder config using YoloDecoder in you yaml")

  if model_config.decoder.type not in yolo_model.YOLO_MODELS[
      model_config.decoder.version].keys():
    raise Exception("unsupported model type please select from \
        {yolo_model.YOLO_MODELS[model_config.decoder.version].keys()},\
        \n\n or specify a custom decoder config using YoloDecoder in you yaml")

  base_model = yolo_model.YOLO_MODELS[model_config.decoder.version][
      model_config.decoder.type]

  cfg_dict = model_config.decoder.as_dict()
  for key in base_model:
    if cfg_dict[key] is not None:
      base_model[key] = cfg_dict[key]

  base_dict = dict(
      activation=activation,
      use_spatial_attention=model_config.decoder.use_spatial_attention,
      use_separable_conv=model_config.decoder.use_separable_conv,
      use_sync_bn=model_config.norm_activation.use_sync_bn,
      norm_momentum=model_config.norm_activation.norm_momentum,
      norm_epsilon=model_config.norm_activation.norm_epsilon,
      kernel_regularizer=l2_regularization)

  base_model.update(base_dict)
  model = YoloDecoder(input_specs, **base_model)
  return model


def build_yolo_filter(model_config: yolo.Yolo, decoder: YoloDecoder, masks,
                      xy_scales, path_scales, anchor_boxes):
  model = YoloLayer(
      masks=masks,
      classes=model_config.num_classes,
      anchors=anchor_boxes,
      iou_thresh=model_config.detection_generator.iou_thresh,
      nms_thresh=model_config.detection_generator.nms_thresh,
      max_boxes=model_config.detection_generator.max_boxes,
      pre_nms_points=model_config.detection_generator.pre_nms_points,
      nms_type=model_config.detection_generator.nms_type,
      box_type=model_config.detection_generator.box_type.get(),
      path_scale=path_scales,
      scale_xy=xy_scales,
      label_smoothing=model_config.loss.label_smoothing,
      use_scaled_loss=model_config.loss.use_scaled_loss,
      update_on_repeat=model_config.loss.update_on_repeat,
      truth_thresh=model_config.loss.truth_thresh.get(),
      loss_type=model_config.loss.box_loss_type.get(),
      max_delta=model_config.loss.max_delta.get(),
      iou_normalizer=model_config.loss.iou_normalizer.get(),
      cls_normalizer=model_config.loss.cls_normalizer.get(),
      obj_normalizer=model_config.loss.obj_normalizer.get(),
      ignore_thresh=model_config.loss.ignore_thresh.get(),
      objectness_smooth=model_config.loss.objectness_smooth.get())
  return model


def build_yolo_head(input_specs, model_config: yolo.Yolo, l2_regularization):  
  min_level = min(map(int, input_specs.keys()))
  max_level = max(map(int, input_specs.keys()))
  head = YoloHead(
      min_level=min_level,
      max_level=max_level,
      classes=model_config.num_classes,
      boxes_per_level=model_config.boxes_per_scale,
      norm_momentum=model_config.norm_activation.norm_momentum,
      norm_epsilon=model_config.norm_activation.norm_epsilon,
      kernel_regularizer=l2_regularization,
      smart_bias=model_config.head.smart_bias)
  return head


def build_yolo(input_specs, 
               model_config, 
               l2_regularization, 
               masks, 
               xy_scales,
               path_scales, 
               anchor_boxes):

  backbone = factory.build_backbone(input_specs, 
                                    model_config.backbone,
                                    model_config.norm_activation,
                                    l2_regularization)
  decoder = build_yolo_decoder(backbone.output_specs, model_config,
                               l2_regularization)
  head = build_yolo_head(decoder.output_specs, model_config, l2_regularization)
  filter = build_yolo_filter(model_config, head, masks, 
                             xy_scales, path_scales, anchor_boxes)

  model = yolo_model.Yolo(
      backbone=backbone, decoder=decoder, head=head, filter=filter)
  model.build(input_specs.shape)

  losses = filter.losses
  return model, losses

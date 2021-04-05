import tensorflow as tf
import tensorflow.keras as ks

from centernet.modeling.backbones.hourglass import Hourglass, build_hourglass
from centernet.modeling.decoders.centernet_decoder import CenterNetDecoder
from centernet.modeling.layers.detection_generator import CenterNetLayer
from official.core import registry
from official.vision.beta.modeling.backbones import factory

# TODO: import prediction and filtering layers when made

class CenterNet(ks.Model):

  def __init__(self, 
               backbone=None,
               decoder=None,
               head=None,
               filter=None,
               **kwargs):
    super().__init__(**kwargs)
    # model components
    self._backbone = backbone
    self._decoder = decoder
    self._head = head
    self._filter = filter
    return

  def call(self, inputs, training=False):
    features = self._backbone(inputs)
    decoded_maps = self._decoder(features)

    if training:
      return {"raw_output": decoded_maps}
    else:
      predictions = self._filter(decoded_maps)
      predictions.update({"raw_output": decoded_maps})
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

def build_centernet_decoder(input_specs, task_config, num_inputs):
  # NOTE: For now just support the default config
  # model specific 
  heatmap_bias = task_config.model.base.decoder.heatmap_bias
  
  # task specific
  task_outputs = task_config._get_output_length_dict()
  model = CenterNetDecoder(
      input_specs = input_specs
      task_outputs=task_outputs,
      heatmap_bias=heatmap_bias,
      num_inputs=num_inputs)

  return model

def build_centernet_filter(model_config):
  return CenterNetLayer(max_detections=model_config.filter.max_detections,
                        peak_error=model_config.filter.peak_error,
                        peak_extract_kernel_size=model_config.filter.peak_extract_kernel_size,
                        use_nms=model_config.filter.use_nms,
                        center_thresh=model_config.filter.center_thresh,
                        iou_thresh=model_config.filter.iou_thresh,
                        class_offset=model_config.filter.class_offset)

def build_centernet_head(model_config):
  return None

def build_centernet(input_specs, task_config, l2_regularization):
  print(task_config.as_dict())
  print(input_specs)
  print(l2_regularization)
  model_config = task_config.model
  backbone = factory.build_backbone(input_specs, model_config.base,
                                    l2_regularization)

  decoder = build_centernet_decoder(backbone.output_specs, task_config, backbone._num_hourglasses)
  head = build_centernet_head(model_config)
  filter = build_centernet_filter(model_config)

  model = CenterNet(backbone=backbone, decoder=decoder, head=head, filter=filter)
  model.build(input_specs.shape)

  # TODO: uncommend when filter is implemented
  # losses = filter.losses
  losses = None
  return model, losses

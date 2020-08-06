from __future__ import annotations

import tensorflow as tf
import tensorflow.keras as ks
from yolo.modeling.backbones.backbone_builder import Backbone_Builder
from yolo.modeling.model_heads._Yolov3Head import Yolov3Head
from yolo.utils.file_manager import download
from yolo.utils import tf_shims
from yolo.utils.scripts.darknet2tf import read_weights, split_list
from yolo.utils.scripts.darknet2tf._load_weights import _load_weights_dnBackbone, _load_weights_dnHead


@tf_shims.ks_utils__register_keras_serializable(package='yolo')
class DarkNet53(ks.Model):
    """The Darknet Image Classification Network Using Darknet53 Backbone"""

    def __init__(
            self,
            classes=1000,
            load_backbone_weights=False,
            config_file=None,
            weights_file=None):
        """
        load the model and the sequential head so that the backbone can be applied for classification

        Model tested on ImageNet Tiny

        Args:
            classes: integer for how many classes can be predicted
            load_backbone_weights: bool, if true we will auto load the original darnet weights to use in the model
            config_file: str path for the location of the configuration file to use when decoding darknet weights
            weights_file: str path with the file containing the dark net weights

        """
        super(DarkNet53, self).__init__()
        self.backbone = Backbone_Builder("darknet53")
        self.head = ks.Sequential([
            ks.layers.GlobalAveragePooling2D(),
            ks.layers.Dense(classes, activation="sigmoid")
        ])
        if load_backbone_weights:
            if config_file is None:
                config_file = download('yolov3.cfg')
            if weights_file is None:
                weights_file = download('yolov3.weights')
            encoder, decoder = read_weights(config_file, weights_file)
            #encoder, _ = split_list(encoder, 76)
            _load_weights_dnBackbone(self.backbone, encoder)
        return

    def call(self, inputs):
        out_dict = self.backbone(inputs)
        x = out_dict[list(out_dict.keys())[-1]]
        return self.head(x)

    def get_summary(self):
        self.backbone.summary()
        self.head.build(input_shape=[None, None, None, 1024])
        self.head.summary()
        print(f"backbone trainable: {self.backbone.trainable}")
        print(f"head trainable: {self.head.trainable}")
        return


@tf_shims.ks_utils__register_keras_serializable(package='yolo')
class Yolov3(ks.Model):
    _updated_config = tf_shims.ks_Model___updated_config
    def __init__(self,
                 classes = 20,
                 boxes = 9,
                 type = 'regular',
                 **kwargs):
        """
        Args:
            classes: int for the number of available classes
            boxes: total number of boxes that are predicted by detection head
            type: the particular type of YOLOv3 model that is being constructed
                  regular, spp, or tiny
        """
        super().__init__(**kwargs)
        self._classes = classes
        self._boxes = boxes
        self._type = type

        if type == 'regular':
            self._backbone_name = "darknet53"
            self._head_name = "regular"
            self._model_name = 'yolov3'
            self._encoder_decoder_split_location = 76
        elif type == 'spp':
            self._backbone_name = "darknet53"
            self._head_name = "spp"
            self._model_name = 'yolov3-spp'
            self._encoder_decoder_split_location = 76
        elif type == 'tiny':
            self._backbone_name = "darknet_tiny"
            self._head_name = "tiny"
            self._model_name = 'yolov3-tiny'
            self._encoder_decoder_split_location = 76
            self._boxes = self._boxes//3 * 2
            print(self._boxes)
        else:
            raise ValueError(f"Unknown YOLOv3 type '{type}'")

    @classmethod
    def spp(clz, **kwargs):
        return clz(type='spp', **kwargs)

    @classmethod
    def tiny(clz, **kwargs):
        return clz(type='tiny', **kwargs)

    def build(self, input_shape=[None, None, None, 3]):
        self._backbone = Backbone_Builder(self._backbone_name)
        self._head = Yolov3Head(model = self._head_name, classes=self._classes, boxes=self._boxes)
        super().build(input_shape)

    def call(self, inputs):
        feature_maps = self._backbone(inputs)
        predictions = self._head(feature_maps)
        return predictions

    def load_weights_from_dn(self,
                             dn2tf_backbone = True,
                             dn2tf_head = False,
                             config_file=None,
                             weights_file=None):
        """
        load the entire Yolov3 Model for tensorflow

        example:
            load yolo with darknet wieghts for backbone
            model = Yolov3()
            model.build(input_shape = (1, 416, 416, 3))
            model.load_weights_from_dn(dn2tf_backbone = True, dn2tf_head = True)

        to be implemented
        example:
            load custom back bone weigths

        example:
            load custom head weigths

        example:
            load back bone weigths from tensorflow (our training)

        example:
            load head weigths from tensorflow (our training)

        Args:
            dn2tf_backbone: bool, if true it will load backbone weights for yolo v3 from darknet .weights file
            dn2tf_head: bool, if true it will load head weights for yolo v3 from darknet .weights file
            config_file: str path for the location of the configuration file to use when decoding darknet weights
            weights_file: str path with the file containing the dark net weights
        """
        if dn2tf_backbone or dn2tf_head:
            if config_file is None:
                config_file = download(self._model_name + '.cfg')
            if weights_file is None:
                weights_file = download(self._model_name + '.weights')
            encoder, decoder = read_weights(config_file, weights_file)
            #encoder, _ = split_list(encoder, self._encoder_decoder_split_location)

        if not self.built:
            net = encoder[0]
            self.build(input_shape = (1, *net.shape))

        if dn2tf_backbone:
            _load_weights_dnBackbone(self._backbone, encoder, mtype = self._backbone_name)

        if dn2tf_head:
            _load_weights_dnHead(self._head, decoder)

    def get_config(self):
        # used to store/share parameters to reconsturct the model
        return {
            "classes": self._classes,
            "boxes": self._boxes,
            "type": self._type,
            "layers": []
        }

    @classmethod
    def from_config(clz, config):
        config = config.copy()
        del config['layers']
        return clz(**config)


if __name__ == '__main__':
    model = Yolov3(type = 'spp', classes=80)
    model.build(input_shape = (None, None, None, 3))
    model.load_weights_from_dn(dn2tf_backbone = True, dn2tf_head = True, config_file=None, weights_file=None)
    # model.load_weights_from_dn(dn2tf_backbone = True, dn2tf_head = True, config_file="yolov3.cfg", weights_file="yolov3_416.weights")
    model.summary()

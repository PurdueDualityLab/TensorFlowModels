import tensorflow as tf
import tensorflow.keras as ks
from yolo.modeling.backbones.backbone_builder import Backbone_Builder
from yolo.modeling.model_heads._Yolov3Head import Yolov3Head
from yolo.modeling.building_blocks import YoloLayer
from yolo.utils.file_manager import download
from yolo.utils import DarkNetConverter
from yolo.utils._darknet2tf.load_weights import split_converter, load_weights_dnBackbone, load_weights_dnHead
import os


@ks.utils.register_keras_serializable(package='yolo')
class DarkNet53(ks.Model):
    """The Darknet Image Classification Network Using Darknet53 Backbone"""
    def __init__(self,
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
        self.backbone = Backbone_Builder("regular")
        self.head = ks.Sequential([
            ks.layers.GlobalAveragePooling2D(),
            ks.layers.Dense(classes, activation="sigmoid")
        ])
        if load_backbone_weights:
            if config_file is None:
                config_file = download('yolov3.cfg')
            if weights_file is None:
                weights_file = download('yolov3.weights')
            full_model = DarkNetConverter.read(config_file, weights_file)
            encoder, decoder = split_converter(full_model, 76)
            load_weights_dnBackbone(self.backbone, encoder)
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

    @property
    def input_image_size(self):
        return self.backbone.input_shape[1:3]

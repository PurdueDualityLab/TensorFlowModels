import tensorflow as tf
import tensorflow.keras as ks
from yolo.modeling.backbones.backbone_builder import Backbone_Builder
from yolo.modeling.model_heads._Yolov3Head import Yolov3Head
from ..utils.file_manager import download


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
            self._load_backbone_weights(config_file, weights_file)
        return

    def call(self, inputs):
        out_dict = self.backbone(inputs)
        x = out_dict[list(out_dict.keys())[-1]]
        return self.head(x)

    def _load_backbone_weights(self, config, weights):
        from yolo.utils.scripts.darknet2tf.get_weights import load_weights, get_darknet53_tf_format
        encoder, decoder, outputs = load_weights(config, weights)
        print(encoder, decoder, outputs)
        encoder, weight_list = get_darknet53_tf_format(encoder[:])
        print(
            f"\nno. layers: {len(self.backbone.layers)}, no. weights: {len(weight_list)}")
        for i, (layer, weights) in enumerate(
                zip(self.backbone.layers, weight_list)):
            print(
                f"loaded weights for layer: {i}  -> name: {layer.name}",
                sep='      ',
                end="\r")
            layer.set_weights(weights)
        self.backbone.trainable = False
        print(
            f"\nsetting backbone.trainable to: {self.backbone.trainable}\n")
        print(f"...training will only affect classification head...")
        return

    def get_summary(self):
        self.backbone.summary()
        self.head.build(input_shape=[None, None, None, 1024])
        self.head.summary()
        print(f"backbone trainable: {self.backbone.trainable}")
        print(f"head trainable: {self.head.trainable}")
        return


class Yolov3(ks.Model):
    def __init__(self, input_shape = [None, None, None, 3], **kwargs):
        super().__init__(**kwargs)

        self._input_shape = input_shape
        self._backbone = Backbone_Builder("darknet53")
        self._head = Yolov3Head("regular")

        inputs = ks.layers.Input(shape = self._input_shape[1:])
        feature_maps = self._backbone(inputs)
        predictions = self._head(feature_maps)
        super().__init__(inputs = inputs, outputs = predictions)


class Yolov3_tiny():
    def __init__(self, input_shape = [None, None, None, 3], **kwargs):
        super().__init__(**kwargs)

        self._input_shape = input_shape
        self._backbone = Backbone_Builder("darknet_tiny")
        self._head = Yolov3Head("tiny")
        
        inputs = ks.layers.Input(shape = self._input_shape[1:])
        feature_maps = self._backbone(inputs)
        predictions = self._head(feature_maps)
        super().__init__(inputs = inputs, outputs = predictions)


class Yolov3_spp():
    def __init__(self, input_shape = [None, None, None, 3], **kwargs):
        super().__init__(**kwargs)

        self._input_shape = input_shape
        self._backbone = Backbone_Builder("darknet53")
        self._head = Yolov3Head("spp")
        
        inputs = ks.layers.Input(shape = self._input_shape[1:])
        feature_maps = self._backbone(inputs)
        predictions = self._head(feature_maps)
        super().__init__(inputs = inputs, outputs = predictions)


model = Yolov3()
model.build(input_shape = None)
model.summary()

model.save("yolo")

tf.keras.utils.plot_model(model, to_file='Yolo.png', show_shapes=True, show_layer_names= False, expand_nested=True, dpi=96)

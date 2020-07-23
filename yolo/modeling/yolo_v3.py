import tensorflow as tf
import tensorflow.keras as ks
from yolo.modeling.backbones.backbone_builder import Backbone_Builder
from ..utils.file_manager import download

class DarkNet53(ks.Model):
    def __init__(self, classes = 1000, load_backbone_weights = False, config_file = None, weights_file = None):
        super(DarkNet53, self).__init__()
        self.backbone = Backbone_Builder("darknet53")
        self.head = ks.Sequential([
            ks.layers.GlobalAveragePooling2D(),
            ks.layers.Dense(classes, activation = "sigmoid")
        ])
        if load_backbone_weights:
            if config_file is None:
                config_file = download('yolov3', 'cfg')
            if weights_file is None:
                weights_file = download('yolov3', 'weights')
            self._load_backbone_weights(config_file, weights_file)
        return

    def call(self, inputs):
        out_dict = self.backbone(inputs)
        x = out_dict[list(out_dict.keys())[-1]]
        return self.head(x)

    def _load_backbone_weights(self, config, weights):
        from yolo.utils.scripts.darknet2tf.get_weights import load_weights, get_darknet53_tf_format
        encoder, decoder, outputs, _ = load_weights(config, weights)
        encoder, weight_list = get_darknet53_tf_format(encoder[:])
        print(f"\nno. layers: {len(self.backbone.layers)}, no. weights: {len(weight_list)}")
        for i, (layer, weights) in enumerate(zip(self.backbone.layers, weight_list)):
            print(f"loaded weights for layer: {i}  -> name: {layer.name}", sep='      ', end="\r")
            layer.set_weights(weights)
        self.backbone.trainable = False
        print(f"\nsetting back_bone.trainable to: {self.backbone.trainable}\n")
        print(f"...training will only affect classification head...")
        return

    def get_summary(self):
        self.backbone.summary()
        self.head.build(input_shape = [None, None, None, 1024])
        self.head.summary()
        print(f"backbone trainable: {self.backbone.trainable}")
        print(f"head trainable: {self.head.trainable}")
        return


class Yolov3():
    def __init__(self):
        pass

class Yolov3_tiny():
    def __init__(self):
        pass

class Yolov3_spp():
    def __init__(self):
        pass

x = tf.ones(shape=[1, 224, 224, 3], dtype = tf.float32)
model = DarkNet53(classes = 1000, load_backbone_weights = True, weights_file = "yolov3_416.weights")
model.get_summary()
y = model(x)

print(y.shape)
config = model.backbone.to_yaml()
print(config)


# print(tf.keras.utils.get_registered_name(model.backbone))

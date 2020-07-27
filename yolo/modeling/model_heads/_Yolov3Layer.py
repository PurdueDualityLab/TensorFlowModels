import time
import tensorflow as tf
import tensorflow.keras as ks
from yolo.modeling.building_blocks import DarkConv
from yolo.modeling.building_blocks import DarkRouteProcess
from yolo.modeling.building_blocks import DarkUpsampleRoute
# for testing
from yolo.modeling.backbones.backbone_builder import Backbone_Builder


# @ks.utils.register_keras_serializable(package='yolo')
class Yolov3Head(tf.keras.layers.Layer):
    def __init__(self, model_type, repetitions_override=None, **kwargs):
        self.type_dict = {"spp": 3, "tiny": 1, "regular": 3}
        self._model_type = model_type

        if repetitions_override is not None:
            self._repetitions = repetitions_override
        else:
            self._repetitions = self.type_dict[model_type]

        super().__init__(**kwargs)
        return

    def build(self, input_shape):
        self.routes = dict()
        self.upsamples = dict()
        self.prediction_heads = dict()

        self.filters = list(reversed(list(input_shape.keys())))

        if self._model_type == "tiny":
            filter_mod = 2
        else:
            filter_mod = 1

        for i, key in enumerate(self.filters):
            if i == 0 and self._model_type == "spp":
                self.routes[key] = DarkRouteProcess(
                    filters=key, repetitions=self._repetitions + 1, insert_spp=True)
            else:
                self.routes[key] = DarkRouteProcess(
                    filters=key // filter_mod,
                    repetitions=self._repetitions,
                    insert_spp=False)

            if i != len(self.filters) - 1:
                self.upsamples[key] = DarkUpsampleRoute(
                    filters=key // (4 * filter_mod))
                filter_mod = 1

            self.prediction_heads[key] = DarkConv(filters=255, kernel_size=(
                1, 1), strides=(1, 1), padding="same", activation=None)

        # print(self.routes)
        # print(self.upsamples)
        # print(self.prediction_heads)
        return

    def call(self, inputs):
        layer_in = inputs[self.filters[0]]
        outputs = dict()
        for i in range(len(self.filters)):
            x_prev, x = self.routes[self.filters[i]](layer_in)
            # print(x_prev.shape)
            if i + 1 < len(self.filters):
                x_next = inputs[self.filters[i + 1]]
                # print(x_next.shape)
                layer_in = self.upsamples[self.filters[i]](x_prev, x_next)
                # print(layer_in.shape)
            outputs[self.filters[i]
                    ] = self.prediction_heads[self.filters[i]](x)
        return outputs


model = Backbone_Builder("darknet53")
model.summary()
head = Yolov3Head("regular")

inputs = [tf.ones(shape=[1, 416, 416, 3], dtype=tf.float32) for i in range(60)]

start = time.time()
for x in inputs:
    y = model(x)
    z = head(y)
print(time.time() - start)

# for key in z.keys():
#     print(z[key].shape)

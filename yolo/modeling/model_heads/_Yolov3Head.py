import tensorflow as tf
import tensorflow.keras as ks
from yolo.modeling.building_blocks import DarkConv
from yolo.modeling.building_blocks import DarkRouteProcess
from yolo.modeling.building_blocks import DarkUpsampleRoute
# for testing
from yolo.modeling.backbones.backbone_builder import Backbone_Builder


@ks.utils.register_keras_serializable(package='yolo')
class Yolov3Head(tf.keras.Model):
    def __init__(self, model="regular", classes=80, boxes=9,
                 repetitions=None, filters=None, mod=1, **kwargs):

        self.type_dict = {
            "regular": {
                "filters": [
                    1024, 512, 256], "repetitions": 3, "mod": 1}, "spp": {
                "filters": [
                    1024, 512, 256], "repetitions": 3, "mod": 1}, "tiny": {
                        "filters": [
                            1024, 256], "repetitions": 1, "mod": 2}}
        self._model_name = model
        input_shape = dict()

        if model in self.type_dict.keys():
            self._filters = self.type_dict[model]["filters"]
            self._repetitions = self.type_dict[model]["repetitions"]
            self._mod = self.type_dict[model]["mod"]
        elif model is None:
            if filters is None or len(filters) == 0:
                raise Exception("unsupported model name")
            if repititions is None or repititions == 0:
                raise Exception("repititions cannot be None")
            self._filters = filters
            self._repetitions = repetitions
            self._mod = mod
        else:
            raise Exception("unsupported model name")

        self._classes = classes
        self._boxes = boxes
        self._boxes_per_layer = boxes // len(self._filters)

        self.pred_depth = (self._boxes_per_layer *
                           self._classes) + (self._boxes_per_layer * 5)

        inputs = dict()
        for count in self._filters:
            if count in inputs:
                raise Exception(
                    "2 filters have the same depth, this is not allowed")
            inputs[count] = ks.layers.Input(shape=[None, None, count])
            input_shape[count] = tf.TensorSpec([None, None, None, count])

        routes, upsamples, prediction_heads = self._get_layers()
        outputs = self._connect_layers(
            routes, upsamples, prediction_heads, inputs)
        super().__init__(inputs=inputs, outputs=outputs, name=self._model_name, **kwargs)
        self._input_shape = input_shape
        return

    def _get_layers(self):
        routes = dict()
        upsamples = dict()
        prediction_heads = dict()

        for i, filters in enumerate(self._filters):
            if i == 0 and self._model_name == "spp":
                routes[filters] = DarkRouteProcess(
                    filters=filters, repetitions=self._repetitions + 1, insert_spp=True)
            else:
                routes[filters] = DarkRouteProcess(
                    filters=filters // self._mod,
                    repetitions=self._repetitions,
                    insert_spp=False)

            if i != len(self._filters) - 1:
                upsamples[filters] = DarkUpsampleRoute(
                    filters=filters // (4 * self._mod))
                self.mod = 1

            prediction_heads[filters] = DarkConv(
                filters=self.pred_depth, kernel_size=(
                    1, 1), strides=(
                    1, 1), padding="same", activation=None)
        return routes, upsamples, prediction_heads

    def _connect_layers(self, routes, upsamples, prediction_heads, inputs):
        outputs = dict()
        layer_in = inputs[self._filters[0]]
        for i in range(len(self._filters)):
            x_prev, x = routes[self._filters[i]](layer_in)
            if i + 1 < len(self._filters):
                x_next = inputs[self._filters[i + 1]]
                layer_in = upsamples[self._filters[i]](x_prev, x_next)
            outputs[self._filters[i]] = prediction_heads[self._filters[i]](x)
        return outputs


# model = Backbone_Builder("darknet53")
# head = Yolov3Head("regular")
# model.summary()
# head.summary()

# model.build(input_shape = None)
# head.build(input_shape = None)

# x = tf.ones(shape=[1, 416, 416, 3], dtype = tf.float32)

# import time
# start = time.time()
# y = model(x)
# z = head(y)
# end = time.time() - start
# print((end * 60), end)

# print({key:value.shape for key, value in y.items()})
# print({key:value.shape for key, value in z.items()})
# import time
# inputs = [tf.ones(shape=[1, 416, 416, 3], dtype = tf.float32) for i in range(60)]

# start = time.time()
# for x in inputs:
#     y = model.predict(x)
#     z = head.predict(y)
# print(time.time() - start)

# for key in z.keys():
#     print(z[key].shape)


# tf.keras.utils.plot_model(model, to_file='backbone.png', show_shapes=True, show_layer_names= False, expand_nested=True, dpi=96)
# tf.keras.utils.plot_model(head, to_file='head.png', show_shapes=True, show_layer_names=False, expand_nested=True, dpi=96)

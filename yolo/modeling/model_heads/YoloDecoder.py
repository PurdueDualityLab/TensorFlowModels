import tensorflow as tf
import tensorflow.keras as ks
from yolo.modeling.building_blocks import DarkConv
from yolo.modeling.building_blocks import DarkRouteProcess
from yolo.modeling.building_blocks import DarkRoute
from yolo.modeling.building_blocks import DarkUpsampleRoute
# config independent


@ks.utils.register_keras_serializable(package='yolo')
class FPNTail(tf.keras.layers.Layer):
    def __init__(self,
                 filters=1,
                 upsample=True,
                 upsample_size=2,
                 activation="leaky",
                 use_sync_bn=False,
                 kernel_regularizer=None,
                 kernel_initializer='glorot_uniform',
                 bias_regularizer=None,
                 norm_epsilon=0.001,
                 norm_momentum=0.99,
                 **kwargs):

        self._filters = filters
        self._upsample = upsample
        self._upsample_size = upsample_size

        self._activation = "leaky" if activation == None else activation
        self._use_sync_bn = use_sync_bn
        self._norm_momentum = norm_momentum
        self._norm_epsilon = norm_epsilon
        self._kernel_initializer = kernel_initializer
        self._kernel_regularizer = kernel_regularizer
        self._bias_regularizer = bias_regularizer
        super().__init__(**kwargs)

    def build(self, input_shape):
        self._route_conv = DarkConv(
            filters=self._filters // 2,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="same",
            activation=self._activation,
            use_sync_bn=self._use_sync_bn,
            kernel_regularizer=self._kernel_regularizer,
            kernel_initializer=self._kernel_initializer,
            bias_regularizer=self._bias_regularizer,
            norm_epsilon=self._norm_epsilon,
            norm_momentum=self._norm_momentum)
        if self._upsample:
            self._process_conv = DarkConv(
                filters=self._filters // 4,
                kernel_size=(1, 1),
                strides=(1, 1),
                padding="same",
                activation=self._activation,
                use_sync_bn=self._use_sync_bn,
                kernel_regularizer=self._kernel_regularizer,
                kernel_initializer=self._kernel_initializer,
                bias_regularizer=self._bias_regularizer,
                norm_epsilon=self._norm_epsilon,
                norm_momentum=self._norm_momentum)
            self._upsampling_block = ks.layers.UpSampling2D(
                size=self._upsample_size)
        return

    def call(self, inputs):
        x_route = self._route_conv(inputs)
        if self._upsample:
            x = self._process_conv(x_route)
            x = self._upsampling_block(x)
            return x_route, x
        else:
            return x_route


@ks.utils.register_keras_serializable(package='yolo')
class YoloFPN(tf.keras.Model):
    def __init__(self,
                 fpn_path_len=4,
                 skip_list=[],
                 activation="leaky",
                 use_sync_bn=False,
                 norm_momentum=0.99,
                 norm_epsilon=0.001,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 **kwargs):

        super().__init__(**kwargs)
        self._fpn_path_len = fpn_path_len
        self._skip_list = skip_list

        self._activation = "leaky" if activation == None else activation
        self._use_sync_bn = use_sync_bn
        self._norm_momentum = norm_momentum
        self._norm_epsilon = norm_epsilon
        self._kernel_initializer = kernel_initializer
        self._kernel_regularizer = kernel_regularizer
        self._bias_regularizer = bias_regularizer
        return

    def get_raw_depths(self, minimum_depth):
        depths = []
        for _ in range(self._min_level, self._max_level + 1):
            depths.append(minimum_depth)
            minimum_depth *= 2
        return list(reversed(depths))

    def build(self, inputs):
        tf.print(inputs)
        """ use config dictionary to generate all important attributes for head construction """
        keys = [int(key) for key in inputs.keys()]
        self._min_level = min(keys)
        self._max_level = max(keys)
        self._min_depth = inputs[str(self._min_level)][-1]
        self._depths = self.get_raw_depths(self._min_depth)

        self.resamples = {}
        self.preprocessors = {}
        self.tails = {}
        for level, depth in zip(
                reversed(range(self._min_level, self._max_level + 1)),
                self._depths):
            print(level, depth)

            if level != self._max_level:
                self.resamples[str(level)] = DarkRoute(
                    filters=depth // 2,
                    activation=self._activation,
                    use_sync_bn=self._use_sync_bn,
                    kernel_regularizer=self._kernel_regularizer,
                    kernel_initializer=self._kernel_initializer,
                    bias_regularizer=self._bias_regularizer,
                    norm_epsilon=self._norm_epsilon,
                    norm_momentum=self._norm_momentum)
                self.preprocessors[str(level)] = DarkRouteProcess(
                    filters=depth,
                    repetitions=self._fpn_path_len,
                    insert_spp=False,
                    activation=self._activation,
                    use_sync_bn=self._use_sync_bn,
                    kernel_regularizer=self._kernel_regularizer,
                    kernel_initializer=self._kernel_initializer,
                    bias_regularizer=self._bias_regularizer,
                    norm_epsilon=self._norm_epsilon,
                    norm_momentum=self._norm_momentum)
            else:
                self.preprocessors[str(level)] = DarkRouteProcess(
                    filters=depth,
                    repetitions=self._fpn_path_len + 2,
                    insert_spp=True,
                    activation=self._activation,
                    use_sync_bn=self._use_sync_bn,
                    kernel_regularizer=self._kernel_regularizer,
                    kernel_initializer=self._kernel_initializer,
                    bias_regularizer=self._bias_regularizer,
                    norm_epsilon=self._norm_epsilon,
                    norm_momentum=self._norm_momentum)
            if level == self._min_level:
                self.tails[str(level)] = FPNTail(
                    filters=depth,
                    upsample=False,
                    activation=self._activation,
                    use_sync_bn=self._use_sync_bn,
                    kernel_regularizer=self._kernel_regularizer,
                    kernel_initializer=self._kernel_initializer,
                    bias_regularizer=self._bias_regularizer,
                    norm_epsilon=self._norm_epsilon,
                    norm_momentum=self._norm_momentum)
            else:
                self.tails[str(level)] = FPNTail(
                    filters=depth,
                    upsample=True,
                    activation=self._activation,
                    use_sync_bn=self._use_sync_bn,
                    kernel_regularizer=self._kernel_regularizer,
                    kernel_initializer=self._kernel_initializer,
                    bias_regularizer=self._bias_regularizer,
                    norm_epsilon=self._norm_epsilon,
                    norm_momentum=self._norm_momentum)
        return

    def call(self, inputs, training=False):
        outputs = {}
        layer_in = inputs[str(self._max_level)]
        for level in reversed(range(self._min_level, self._max_level + 1)):
            _, x = self.preprocessors[str(level)](layer_in)
            if level > self._min_level:
                x_route, x = self.tails[str(level)](x)
                x_next = inputs[str(level - 1)]
                layer_in = self.resamples[str(level - 1)]([x_next, x])
            else:
                x_route = self.tails[str(level)](x)
            outputs[str(level)] = x_route
        return outputs


@ks.utils.register_keras_serializable(package='yolo')
class YoloRoutedDecoder(ks.Model):
    def __init__(self,
                 classes=80,
                 boxes_per_level=3,
                 output_extras=0,
                 path_process_len=6,
                 max_level_process_len=None,
                 embed_spp=False,
                 skip_list=[],
                 activation="leaky",
                 use_sync_bn=False,
                 norm_momentum=0.99,
                 norm_epsilon=0.001,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 **kwargs):
        super().__init__(**kwargs)
        self._output_conv = (classes + output_extras + 5) * boxes_per_level
        self._path_process_len = path_process_len
        self._max_level_process_len = path_process_len if max_level_process_len == None else max_level_process_len

        self._embed_spp = embed_spp
        self._skip_list = skip_list

        self._activation = "leaky" if activation == None else activation
        self._use_sync_bn = use_sync_bn
        self._norm_momentum = norm_momentum
        self._norm_epsilon = norm_epsilon
        self._kernel_initializer = kernel_initializer
        self._kernel_regularizer = kernel_regularizer
        self._bias_regularizer = bias_regularizer

    def build(self, inputs):
        # tf.print(inputs)
        keys = [int(key) for key in inputs.keys()]
        self._min_level = min(keys)
        self._max_level = max(keys)
        self._min_depth = inputs[str(self._min_level)][-1]
        self._depths = self.get_raw_depths(self._min_depth)

        self.resamples = {}
        self.preprocessors = {}
        self.outputs = {}

        for level, depth in zip(
                reversed(range(self._min_level, self._max_level + 1)),
                self._depths):
            print(level, depth)
            if level == self._max_level:
                self.preprocessors[str(level)] = DarkRouteProcess(
                    filters=depth,
                    repetitions=self._max_level_process_len + 2 *
                    (1 if self._embed_spp else 0),
                    insert_spp=self._embed_spp,
                    activation=self._activation,
                    use_sync_bn=self._use_sync_bn,
                    kernel_regularizer=self._kernel_regularizer,
                    kernel_initializer=self._kernel_initializer,
                    bias_regularizer=self._bias_regularizer,
                    norm_epsilon=self._norm_epsilon,
                    norm_momentum=self._norm_momentum)
            else:
                self.resamples[str(level)] = DarkUpsampleRoute(
                    filters=depth // 2,
                    activation=self._activation,
                    use_sync_bn=self._use_sync_bn,
                    kernel_regularizer=self._kernel_regularizer,
                    kernel_initializer=self._kernel_initializer,
                    bias_regularizer=self._bias_regularizer,
                    norm_epsilon=self._norm_epsilon,
                    norm_momentum=self._norm_momentum)
                self.preprocessors[str(level)] = DarkRouteProcess(
                    filters=depth,
                    repetitions=self._path_process_len,
                    insert_spp=False,
                    activation=self._activation,
                    use_sync_bn=self._use_sync_bn,
                    kernel_regularizer=self._kernel_regularizer,
                    kernel_initializer=self._kernel_initializer,
                    bias_regularizer=self._bias_regularizer,
                    norm_epsilon=self._norm_epsilon,
                    norm_momentum=self._norm_momentum)
            self.outputs[str(level)] = DarkConv(
                filters=self._output_conv,
                kernel_size=(1, 1),
                strides=(1, 1),
                padding="same",
                use_bn=False,
                kernel_regularizer=self._kernel_regularizer,
                kernel_initializer=self._kernel_initializer,
                bias_regularizer=self._bias_regularizer,
                norm_epsilon=self._norm_epsilon,
                norm_momentum=self._norm_momentum,
                activation=None)
        return

    def get_raw_depths(self, minimum_depth):
        depths = []
        for _ in range(self._min_level, self._max_level + 1):
            depths.append(minimum_depth)
            minimum_depth *= 2
        return list(reversed(depths))

    def call(self, inputs, training=False):
        outputs = dict()
        layer_in = inputs[str(self._max_level)]
        for level in reversed(range(self._min_level, self._max_level + 1)):
            x_route, x = self.preprocessors[str(level)](layer_in)
            outputs[str(level)] = self.outputs[str(level)](x)
            if level > self._min_level:
                x_next = inputs[str(level - 1)]
                layer_in = self.resamples[str(level - 1)]([x_route, x_next])
        return outputs


@ks.utils.register_keras_serializable(package='yolo')
class YoloFPNDecoder(ks.Model):
    def __init__(self,
                 classes=80,
                 boxes_per_level=3,
                 output_extras=0,
                 path_process_len=6,
                 max_level_process_len=None,
                 embed_spp=False,
                 skip_list=[],
                 activation="leaky",
                 use_sync_bn=False,
                 norm_momentum=0.99,
                 norm_epsilon=0.001,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 **kwargs):
        super().__init__(**kwargs)
        self._output_conv = (classes + output_extras + 5) * boxes_per_level
        self._path_process_len = path_process_len
        self._max_level_process_len = 1 if max_level_process_len == None else max_level_process_len

        self._embed_spp = embed_spp
        self._skip_list = skip_list

        self._activation = "leaky" if activation == None else activation
        self._use_sync_bn = use_sync_bn
        self._norm_momentum = norm_momentum
        self._norm_epsilon = norm_epsilon
        self._kernel_initializer = kernel_initializer
        self._kernel_regularizer = kernel_regularizer
        self._bias_regularizer = bias_regularizer

    def get_raw_depths(self, minimum_depth):
        depths = []
        for _ in range(self._min_level, self._max_level + 1):
            depths.append(minimum_depth)
            minimum_depth *= 2
        return depths

    def build(self, inputs):
        tf.print(inputs)
        keys = [int(key) for key in inputs.keys()]
        self._min_level = min(keys)
        self._max_level = max(keys)
        self._min_depth = inputs[str(self._min_level)][-1]
        self._depths = self.get_raw_depths(self._min_depth)

        self.resamples = {}
        self.preprocessors = {}
        self.outputs = {}

        for level, depth in zip(range(self._min_level, self._max_level + 1),
                                self._depths):
            print(level, depth)
            if level == self._min_level:
                self.preprocessors[str(level)] = DarkRouteProcess(
                    filters=depth * 2,
                    repetitions=self._max_level_process_len + 2 *
                    (1 if self._embed_spp else 0),
                    insert_spp=self._embed_spp,
                    activation=self._activation,
                    use_sync_bn=self._use_sync_bn,
                    kernel_regularizer=self._kernel_regularizer,
                    kernel_initializer=self._kernel_initializer,
                    bias_regularizer=self._bias_regularizer,
                    norm_epsilon=self._norm_epsilon,
                    norm_momentum=self._norm_momentum)
            else:
                self.resamples[str(level)] = DarkRoute(
                    filters=depth,
                    downsample=True,
                    activation=self._activation,
                    use_sync_bn=self._use_sync_bn,
                    kernel_regularizer=self._kernel_regularizer,
                    kernel_initializer=self._kernel_initializer,
                    bias_regularizer=self._bias_regularizer,
                    norm_epsilon=self._norm_epsilon,
                    norm_momentum=self._norm_momentum)
                self.preprocessors[str(level)] = DarkRouteProcess(
                    filters=depth * 2,
                    repetitions=self._path_process_len,
                    insert_spp=False,
                    activation=self._activation,
                    use_sync_bn=self._use_sync_bn,
                    kernel_regularizer=self._kernel_regularizer,
                    kernel_initializer=self._kernel_initializer,
                    bias_regularizer=self._bias_regularizer,
                    norm_epsilon=self._norm_epsilon,
                    norm_momentum=self._norm_momentum)
            self.outputs[str(level)] = DarkConv(
                filters=self._output_conv,
                kernel_size=(1, 1),
                strides=(1, 1),
                padding="same",
                use_bn=False,
                kernel_regularizer=self._kernel_regularizer,
                kernel_initializer=self._kernel_initializer,
                bias_regularizer=self._bias_regularizer,
                norm_epsilon=self._norm_epsilon,
                norm_momentum=self._norm_momentum,
                activation=None)
        return

    def call(self, inputs, training=False):
        outputs = dict()
        layer_in = inputs[str(self._min_level)]
        for level in range(self._min_level, self._max_level + 1):
            x_route, x = self.preprocessors[str(level)](layer_in)
            if level < self._max_level:
                x_next = inputs[str(level + 1)]
                layer_in = self.resamples[str(level + 1)]([x_route, x_next])
            outputs[str(level)] = self.outputs[str(level)](x)
        return outputs


@ks.utils.register_keras_serializable(package='yolo')
class YoloDecoder(ks.Model):
    def __init__(self,
                 classes=80,
                 boxes_per_level=3,
                 output_extras=0,
                 embed_fpn=False,
                 fpn_path_len=4,
                 path_process_len=6,
                 max_level_process_len=None,
                 embed_spp=False,
                 skip_list=[],
                 activation="leaky",
                 use_sync_bn=False,
                 norm_momentum=0.99,
                 norm_epsilon=0.001,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 **kwargs):
        super().__init__(**kwargs)
        self._classes = classes
        self._boxes_per_level = boxes_per_level
        self._output_extras = output_extras
        self._embed_fpn = embed_fpn
        self._fpn_path_len = fpn_path_len
        self._path_process_len = path_process_len
        self._max_level_process_len = max_level_process_len
        self._embed_spp = embed_spp
        self._skip_list = skip_list

        self._activation = "leaky" if activation == None else activation
        self._use_sync_bn = use_sync_bn
        self._norm_momentum = norm_momentum
        self._norm_epsilon = norm_epsilon
        self._kernel_initializer = kernel_initializer
        self._kernel_regularizer = kernel_regularizer
        self._bias_regularizer = bias_regularizer

    def build(self, inputs):
        tf.print(inputs)
        keys = [int(key) for key in inputs.keys()]
        self._min_level = min(keys)
        self._max_level = max(keys)
        if self._embed_fpn:
            self._fpn = YoloFPN(fpn_path_len=self._fpn_path_len,
                                activation=self._activation,
                                use_sync_bn=self._use_sync_bn,
                                norm_momentum=self._norm_momentum,
                                norm_epsilon=self._norm_epsilon,
                                kernel_initializer=self._kernel_initializer,
                                kernel_regularizer=self._kernel_regularizer,
                                bias_regularizer=self._bias_regularizer)
            self._decoder = YoloFPNDecoder(
                classes=self._classes,
                boxes_per_level=self._boxes_per_level,
                output_extras=self._output_extras,
                path_process_len=self._path_process_len,
                max_level_process_len=self._max_level_process_len,
                embed_spp=self._embed_spp,
                skip_list=self._skip_list,
                activation=self._activation,
                use_sync_bn=self._use_sync_bn,
                norm_momentum=self._norm_momentum,
                norm_epsilon=self._norm_epsilon,
                kernel_initializer=self._kernel_initializer,
                kernel_regularizer=self._kernel_regularizer,
                bias_regularizer=self._bias_regularizer)
        else:
            self._fpn = None
            self._decoder = YoloRoutedDecoder(
                classes=self._classes,
                boxes_per_level=self._boxes_per_level,
                output_extras=self._output_extras,
                path_process_len=self._path_process_len,
                max_level_process_len=self._max_level_process_len,
                embed_spp=self._embed_spp,
                skip_list=self._skip_list,
                activation=self._activation,
                use_sync_bn=self._use_sync_bn,
                norm_momentum=self._norm_momentum,
                norm_epsilon=self._norm_epsilon,
                kernel_initializer=self._kernel_initializer,
                kernel_regularizer=self._kernel_regularizer,
                bias_regularizer=self._bias_regularizer)
        return

    def call(self, inputs, training=False):
        if self._embed_fpn:
            inputs = self._fpn(inputs)
        return self._decoder(inputs)

    def load_from_dn(self, neck, decoder):
        return

    @property
    def neck(self):
        return self._fpn

    @property
    def head(self):
        return self._decoder

    @property
    def output_depth(self):
        return (self._classes + self._output_extras +
                5) * self._boxes_per_level

    def get_loss_attributes(self,
                            xy_exponential=False,
                            exp_base=2,
                            xy_scale_base="default_value"):
        start = 0
        boxes = {}
        path_scales = {}
        scale_x_y = {}

        if xy_scale_base == "default_base":
            xy_scale_base = 0.05
            xy_scale_base = xy_scale_base / (
                self._boxes_per_level *
                (self._max_level - self._min_level + 1) - 1)
        elif xy_scale_base == "default_value":
            xy_scale_base = 0.00625

        for i in range(self._min_level, self._max_level + 1):
            boxes[str(i)] = list(range(start, self._boxes_per_level + start))
            path_scales[str(i)] = 2**i
            if xy_exponential:
                scale_x_y[str(i)] = 1.0 + xy_scale_base * (exp_base**i)
            else:
                scale_x_y[str(i)] = 1.0
            start += self._boxes_per_level
        return boxes, path_scales, scale_x_y

    def get_boxes(self):
        
        return boxes, path_scales, scale_x_y

def test():
    from yolo.modeling.backbones.Darknet import Darknet
    from yolo.modeling.backbones.Spinenet import SmallSpineNet

    inputs = ks.layers.Input(shape=[416, 416, 3])
    backbone = Darknet(model_id="darknettiny",
                       min_level=2,
                       max_level=5,
                       input_specs=(416, 416, 3))

    #backbone = SmallSpineNet(input_shape=(None, 416, 416, 3), min_level= 3, max_level=5)
    decoder = YoloDecoder(classes=80,
                          boxes_per_level=3,
                          embed_spp=False,
                          embed_fpn=True,
                          max_level_process_len=6,
                          path_process_len=6)

    y = backbone(inputs)
    print(y)
    outputs = decoder(y)
    model = ks.Model(inputs=inputs, outputs=outputs)
    model.build([None, 416, 416, 3])
    model.summary()

    print(backbone.output_shape)
    print(
        decoder.get_loss_attributes(xy_exponential=True,
                                    xy_scale_base="default_value"))


if __name__ == "__main__":
    test()

import tensorflow as tf
import tensorflow.keras as ks
from typing import *

import yolo.modeling.base_model as base_model
from yolo.modeling.backbones.backbone_builder import Backbone_Builder
from yolo.modeling.model_heads._Yolov3Head import Yolov3Head
from yolo.modeling.building_blocks import YoloLayer

from yolo.utils.file_manager import download
from yolo.utils import DarkNetConverter
from yolo.utils._darknet2tf.load_weights import split_converter, load_weights_dnBackbone, load_weights_dnHead


class Yolov3(base_model.Yolo):
    def __init__(
            self,
            input_shape=[None, None, None, 3],
            model="regular",  # options {regular, spp, tiny}
            classes=80,
            backbone=None,
            head=None,
            head_filter=None,
            weight_decay = 5e-4,
            clip_grads_norm = None,
            thresh: int = 0.45,
            class_thresh: int = 0.45,
            path_scales=None,
            x_y_scales=None,
            use_tie_breaker: bool = False,
            masks=None,
            anchors=None,
            max_boxes: int = 200,
            scale_boxes: int = 416,
            policy = None, 
            use_nms = True,
            using_rt = False,
            **kwargs):
        super().__init__(**kwargs)

        #required inputs
        self._input_shape = input_shape
        self._classes = classes
        self._type = model
        self._encoder_decoder_split_location = None
        self._built = False
        self._custom_aspects = False

        #setting the running policy
        if policy == None:
            if type(policy) != str:
                policy = policy.name
            self._og_policy = policy
            self._policy = tf.keras.mixed_precision.experimental.global_policy().name
            self.set_policy(policy=policy)
        else:
            self._og_policy = tf.keras.mixed_precision.experimental.global_policy().name
            self._policy = self._og_policy

        #filtering params
        self._thresh = thresh
        self._class_thresh = class_thresh
        self._max_boxes = max_boxes
        self._scale_boxes = scale_boxes
        self._x_y_scales = x_y_scales

        #init base params
        self._encoder_decoder_split_location = None
        self._boxes = anchors
        self._masks = masks
        self._path_scales = path_scales
        self._use_tie_breaker = use_tie_breaker

        #init models
        self.model_name = model
        self._model_name = None
        self._backbone_name = None
        self._backbone_cfg = backbone
        self._head_cfg = head
        self._head_filter_cfg = head_filter
        self._weight_decay = weight_decay
        self._use_nms = use_nms
        self._using_rt = using_rt

        self._clip_grads_norm = clip_grads_norm

        self.get_default_attributes()
        self._loss_fn = None
        self._loss_weight = None
        return

    @property
    def backbone():
        return self._backbone

    @property
    def head():
        return self._head

    def get_default_attributes(self):
        if self.model_name == "regular" or self.model_name == "spp":
            self._encoder_decoder_split_location = 76
            self._boxes = self._boxes or [(10, 13), (16, 30), (33, 23),
                                          (30, 61), (62, 45), (59, 119),
                                          (116, 90), (156, 198), (373, 326)]
            self._masks = self._masks or {
                5: [6, 7, 8],
                4: [3, 4, 5],
                3: [0, 1, 2]
            }
            self._path_scales = self._path_scales or {
                5: 32,
                4: 16,
                3: 8
            }
            self._x_y_scales = self._x_y_scales or {
                5: 1.0,
                4: 1.0,
                3: 1.0
            }
        elif self.model_name == "tiny":
            self._encoder_decoder_split_location = 14
            self._boxes = self._boxes or [(10, 14), (23, 27), (37, 58),
                                          (81, 82), (135, 169), (344, 319)]
            self._masks = self._masks or {5: [3, 4, 5], 3: [0, 1, 2]}
            self._path_scales = self._path_scales or {5: 32, 3: 8}
            self._x_y_scales = self._x_y_scales or {5: 1.0, 3: 1.0}
        return

    def get_summary(self):
        self._backbone.summary()
        self._head.summary()
        print(self._backbone.output_shape)
        print(self._head.output_shape)
        self.summary()
        return

    def build(self, input_shape):
        default_dict = {
            "regular": {
                "backbone": "regular",
                "head": "regular",
                "name": "yolov3"
            },
            "spp": {
                "backbone": "regular",
                "head": "spp",
                "name": "yolov3-spp"
            },
            "tiny": {
                "backbone": "tiny",
                "head": "tiny",
                "name": "yolov3-tiny"
            }
        }
        if self._backbone_cfg == None or isinstance(self._backbone_cfg, Dict):
            self._backbone_name = default_dict[self.model_name]["backbone"]
            if isinstance(self._backbone_cfg, Dict):
                default_dict[self.model_name]["backbone"] = self._backbone_cfg
            self._backbone = Backbone_Builder(
                name=default_dict[self.model_name]["backbone"],
                config=default_dict[self.model_name]["backbone"],
                input_shape=self._input_shape,
                weight_decay=self._weight_decay)
        else:
            self._backbone = self._backbone_cfg
            self._custom_aspects = True

        if self._head_cfg == None or isinstance(self._head_cfg, Dict):
            if isinstance(self._head_cfg, Dict):
                default_dict[self.model_name]["head"] = self._head_cfg
            self._head = Yolov3Head(
                model=default_dict[self.model_name]["head"],
                cfg_dict=default_dict[self.model_name]["head"],
                classes=self._classes,
                boxes=len(self._boxes),
                input_shape=self._input_shape,
                weight_decay=self._weight_decay)
        else:
            self._head = self._head_cfg
            self._custom_aspects = True

        if self._head_filter_cfg == None:
            self._head_filter = YoloLayer(masks=self._masks,
                                          anchors=self._boxes,
                                          thresh=self._thresh,
                                          cls_thresh=self._class_thresh,
                                          max_boxes=self._max_boxes,
                                          path_scale=self._path_scales,
                                          scale_xy=self._x_y_scales,
                                          use_nms=self._use_nms)
        else:
            self._head_filter = self._head_filter_cfg

        self._model_name = default_dict[self.model_name]["name"]
        self._backbone.build(input_shape)
        self._head.build(self._backbone.output_shape)
        self._built = True
        super().build(input_shape)
        return

    def call(self, inputs, training=False):
        feature_maps = self._backbone(inputs)
        raw_head = self._head(feature_maps)
        if training or self._using_rt:
            return {"raw_output": raw_head}
        else:
            predictions = self._head_filter(raw_head)
            return predictions

    def load_weights_from_dn(self,
                             dn2tf_backbone=True,
                             dn2tf_head=True,
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
        if not self._built:
            self.build(self._input_shape)

        if dn2tf_backbone or dn2tf_head:
            if config_file is None:
                config_file = download(self._model_name + '.cfg')
            if weights_file is None:
                weights_file = download(self._model_name + '.weights')
            list_encdec = DarkNetConverter.read(config_file, weights_file)
            encoder, decoder = split_converter(
                list_encdec, self._encoder_decoder_split_location)

        if dn2tf_backbone:
            load_weights_dnBackbone(self._backbone,
                                    encoder,
                                    mtype=self._backbone_name)

        if dn2tf_head:
            load_weights_dnHead(self._head, decoder)
        return


if __name__ == "__main__":
    import tensorflow_datasets as tfds
    from yolo.utils.testing_utils import prep_gpu
    from yolo.training.call_backs.PrintingCallBack import Printer
    prep_gpu()
    # train, info = tfds.load('coco',
    #                         split='train',
    #                         shuffle_files=True,
    #                         with_info=True)
    # test, info = tfds.load('coco',
    #                        split='validation',
    #                        shuffle_files=False,
    #                        with_info=True)

    model = Yolov3(model = "regular", policy="float32", use_tie_breaker=False)
    model.build(input_shape = [None, None, None, 3])
    model.summary()
    # model.load_weights_from_dn(dn2tf_head=False, weights_file="testing_weights/yolov3-regular.weights")

    # train, test = model.process_datasets(train, test, fixed_size = False , batch_size=1, jitter_im = 0.1, jitter_boxes = 0.005, _eval_is_training = False)
    # loss_fn = model.generate_loss(loss_type="ciou")

    # #optimizer = ks.optimizers.SGD(lr=1e-3)
    # optimizer = ks.optimizers.Adam(lr=1e-3)
    # optimizer = model.match_optimizer_to_policy(optimizer)
    # model.compile(optimizer=optimizer, loss=loss_fn)

    # try:
    #     model.fit(train, validation_data = test, epochs = 40, verbose = 1, shuffle = True)
    #     model.save_weights("testing_weights/yolov3/simple_test1")
    # except:
    #     model.save_weights("testing_weights/yolov3/simple_test1_early")
    # #model.evaluate(test)

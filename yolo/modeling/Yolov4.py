import tensorflow as tf
import tensorflow.keras as ks
from typing import *

import yolo.modeling.base_model as base_model
from yolo.modeling.backbones.csp_backbone_builder import CSP_Backbone_Builder
from yolo.modeling.model_heads._Yolov4Neck import Yolov4Neck
from yolo.modeling.model_heads._Yolov4Head import Yolov4Head
from yolo.modeling.building_blocks import YoloLayer

from yolo.utils.file_manager import download
from yolo.utils import DarkNetConverter
from yolo.utils._darknet2tf.load_weights import split_converter
from yolo.utils._darknet2tf.load_weights2 import load_weights_backbone, load_weights_v4head

class Yolov4(base_model.Yolo):
    def __init__(
            self,
            input_shape=[None, None, None, 3],
            model="regular",  # options {regular, spp, tiny}
            classes=80,
            backbone = None,
            neck = None,
            head = None,
            head_filter = None,
            masks = None,
            anchors = None,
            path_scales = None,
            x_y_scales = None,
            iou_thresh: int = 0.45,
            weight_decay = 5e-4,
            class_thresh: int = 0.45,
            use_nms = True,
            using_rt = False,
            max_boxes: int = 200,
            scale_boxes: int = 416,
            use_tie_breaker: bool = True,
            clip_grads_norm = None,
            policy=None,
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
        self._thresh = iou_thresh
        self._class_thresh = 0.45
        self._max_boxes = max_boxes
        self._scale_boxes = scale_boxes
        self._x_y_scales = x_y_scales

        #init base params
        self._encoder_decoder_split_location = None
        self._boxes = anchors
        self._masks = masks
        self._path_scales = path_scales
        self._use_tie_breaker = use_tie_breaker
        self._weight_decay = weight_decay

        #init models
        self.model_name = model
        self._model_name = None
        self._backbone_name = None
        self._backbone_cfg = backbone
        self._neck_cfg = neck
        self._head_cfg = head
        self._head_filter_cfg = head_filter
        self._use_nms = use_nms
        self._using_rt = using_rt

        self._clip_grads_norm = clip_grads_norm

        self.get_default_attributes()
        self._loss_fn = None
        self._loss_weight = None
        return

    def get_default_attributes(self):
        if self.model_name == "regular":
            self._encoder_decoder_split_location = 106
            self._boxes = self._boxes or [(12, 16), (19, 36), (40, 28), (36, 75),(76, 55), (72, 146), (142, 110),(192, 243), (459, 401)]
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
            self._x_y_scales = self._x_y_scales or {5: 1.05, 4: 1.1, 3: 1.2}

        return

    def get_summary(self):
        self._backbone.summary()
        self._neck.summary()
        self._head.summary()
        print(self._backbone.output_shape)
        print(self._neck.output_shape)
        print(self._head.output_shape)
        self.summary()
        return

    def build(self, input_shape):
        default_dict = {
            "regular": {
                "backbone": "regular",
                "neck": "regular",
                "head": "regular",
                "name": "yolov4"
            },
        }
        #if not self._built:
        if self._backbone_cfg == None or isinstance(self._backbone_cfg, Dict):
            self._backbone_name = default_dict[self.model_name]["backbone"]
            if isinstance(self._backbone_cfg, Dict):
                default_dict[self.model_name]["backbone"] = self._backbone_cfg
            self._backbone = CSP_Backbone_Builder(
                name=default_dict[self.model_name]["backbone"],
                config=default_dict[self.model_name]["backbone"],
                input_shape=self._input_shape,
                weight_decay=self._weight_decay)
        else:
            self._backbone = self._backbone_cfg
            self._custom_aspects = True

        if self._neck_cfg == None or isinstance(self._neck_cfg, Dict):
            if isinstance(self._neck_cfg, Dict):
                default_dict[self.model_name]["neck"] = self._neck_cfg
            self._neck = Yolov4Neck(
                name=default_dict[self.model_name]["neck"],
                cfg_dict=default_dict[self.model_name]["neck"],
                input_shape=self._input_shape,
                weight_decay=self._weight_decay)
        else:
            self._neck = self._neck_cfg
            self._custom_aspects = True

        if self._head_cfg == None or isinstance(self._head_cfg, Dict):
            if isinstance(self._head_cfg, Dict):
                default_dict[self.model_name]["head"] = self._head_cfg
            self._head = Yolov4Head(
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
        self._neck.build(self._backbone.output_shape)
        self._head.build(self._neck.output_shape)
        self._built = True
        super().build(input_shape)
        return

    def call(self, inputs, training=False):
        feature_maps = self._backbone(inputs)
        neck_maps = self._neck(feature_maps)
        raw_head = self._head(neck_maps)
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

        if dn2tf_backbone or dn2tf_neck or dn2tf_head:
            if config_file is None:
                config_file = download(self._model_name + '.cfg')
            if weights_file is None:
                weights_file = download(self._model_name + '.weights')
            list_encdec = DarkNetConverter.read(config_file, weights_file)
            encoder, neck, decoder = split_converter(
                list_encdec, self._encoder_decoder_split_location, 138)


        if dn2tf_backbone:
            #load_weights_dnBackbone(self._backbone, encoder, mtype = self._backbone_name)
            load_weights_backbone(self._backbone, encoder)
            self._backbone.trainable = False

        if dn2tf_head:
            load_weights_backbone(self._neck, neck)
            self._neck.trainable = False
            load_weights_v4head(self._head, decoder)
            self._head.trainable = False
        return


if __name__ == "__main__":
    import tensorflow_datasets as tfds
    from yolo.utils.testing_utils import prep_gpu
    from yolo.training.call_backs.PrintingCallBack import Printer
    prep_gpu() # must be called before loading a dataset
    # train, info = tfds.load('coco/2017',
    #                         split='train',
    #                         shuffle_files=True,
    #                         with_info=True)
    # test, info = tfds.load('coco/2017',
    #                        split='validation',
    #                        shuffle_files=False,
    #                        with_info=True)


    model = Yolov4(model = "regular", policy="float32", use_tie_breaker=True)
    model.build(model._input_shape)
    model.get_summary()
    # model.load_weights_from_dn(dn2tf_head=True)


    # train, test = model.process_datasets(train, test, batch_size=1, jitter_im = 0.1, jitter_boxes = 0.005, _eval_is_training = False)
    # loss_fn = model.generate_loss(loss_type="ciou")


    # #optimizer = ks.optimizers.SGD(lr=1e-3)
    # optimizer = ks.optimizers.Adam(lr=1e-3/32)
    # optimizer = model.match_optimizer_to_policy(optimizer)
    # model.compile(optimizer=optimizer, loss=loss_fn)

    # tensorboard = tf.keras.callbacks.TensorBoard()
    # model.evaluate(test, callbacks = [tensorboard])

    # try:
    #     model.fit(train, validation_data = test, epochs = 39, verbose = 1, shuffle = True)
    #     model.save_weights("testing_weights/yolov4/simple_test1")
    # except:
    #     model.save_weights("testing_weights/yolov4/simple_test1_early")

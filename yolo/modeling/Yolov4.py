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
            backbone=None,
            neck = None, 
            head=None,
            head_filter=None,
            masks=None,
            boxes=None,
            path_scales=None,
            x_y_scales=None,
            thresh: int = 0.45,
            weight_decay = 0.005,
            class_thresh: int = 0.45,
            max_boxes: int = 200,
            scale_boxes: int = 416,
            scale_mult: float = 1.0,
            use_tie_breaker: bool = True,
            clip_grads_norm = None, 
            policy="float32",
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
        if type(policy) != str:
            policy = policy.name
        self._og_policy = policy
        self._policy = tf.keras.mixed_precision.experimental.global_policy(
        ).name
        self.set_policy(policy=policy)

        #filtering params
        self._thresh = thresh
        self._class_thresh = 0.45
        self._max_boxes = max_boxes
        self._scale_boxes = scale_boxes
        self._scale_mult = scale_mult
        self._x_y_scales = x_y_scales

        #init base params
        self._encoder_decoder_split_location = None
        self._boxes = boxes
        self._masks = masks
        self._path_scales = path_scales
        self._use_tie_breaker = use_tie_breaker

        #init models
        self.model_name = model
        self._model_name = None
        self._backbone_name = None
        self.backbone = backbone
        self.neck = neck
        self.head = head
        self.head_filter = head_filter
        self._weight_decay = weight_decay

        self._clip_grads_norm = clip_grads_norm

        self.get_models()
        self._loss_fn = None
        self._loss_weight = None
        return

    def get_models(self):
        default_dict = {
            "regular": {
                "backbone": "darknet53",
                "neck": "neck",
                "head": "regular",
                "name": "yolov4"
            },
        }

        if self.model_name == "regular":
            self._encoder_decoder_split_location = 106
            self._boxes = self._boxes or [(12, 16), (19, 36), (40, 28), (36, 75),(76, 55), (72, 146), (142, 110),(192, 243), (459, 401)]
            self._masks = self._masks or {
                "1024": [6, 7, 8],
                "512": [3, 4, 5],
                "256": [0, 1, 2]
            }
            self._path_scales = self._path_scales or {
                "1024": 32,
                "512": 16,
                "256": 8
            }
            self._x_y_scales = self._x_y_scales or {"1024": 1.05, "512": 1.1, "256": 1.2}

        if self.backbone == None or isinstance(self.backbone, Dict):
            self._backbone_name = default_dict[self.model_name]["backbone"]
            if isinstance(self.backbone, Dict):
                default_dict[self.model_name]["backbone"] = self.backbone
            self.backbone = CSP_Backbone_Builder(
                name=default_dict[self.model_name]["backbone"],
                config=default_dict[self.model_name]["backbone"],
                input_shape=self._input_shape, 
                weight_decay=self._weight_decay)

        else:
            self._custom_aspects = True
        
        if self.neck == None or isinstance(self.neck, Dict):
            if isinstance(self.neck, Dict):
                default_dict[self.model_name]["neck"] = self.neck
            self.neck = Yolov4Neck(
                name=default_dict[self.model_name]["neck"],
                cfg_dict=default_dict[self.model_name]["neck"],
                input_shape=self._input_shape)
        else:
            self._custom_aspects = True

        if self.head == None or isinstance(self.head, Dict):
            if isinstance(self.head, Dict):
                default_dict[self.model_name]["head"] = self.head
            self.head =  Yolov4Head(
                model=default_dict[self.model_name]["head"],
                cfg_dict=default_dict[self.model_name]["head"],
                classes=self._classes,
                boxes=len(self._boxes),
                input_shape=self._input_shape)
        else:
            self._custom_aspects = True

        if self.head_filter == None:
            self.head_filter = YoloLayer(masks=self._masks,
                                         anchors=self._boxes,
                                         thresh=self._thresh,
                                         cls_thresh=self._class_thresh,
                                         max_boxes=self._max_boxes,
                                         scale_boxes=self._scale_boxes,
                                         scale_mult=self._scale_mult,
                                         path_scale=self._path_scales)


        self._model_name = default_dict[self.model_name]["name"]
        return

    def get_summary(self):
        self.backbone.summary()
        self.neck.summary()
        self.head.summary()
        print(self.backbone.output_shape)
        print(self.neck.output_shape)
        print(self.head.output_shape)
        return

    def build(self, input_shape):
        self.backbone.build(input_shape)
        self.neck.build(self.backbone.output_shape)
        self.head.build(self.neck.output_shape)
        #self.head_filter.build(self.head.output_shape)
        return

    def call(self, inputs, training=True):
        feature_maps = self.backbone(inputs)
        neck_maps = self.neck(feature_maps)
        raw_head = self.head(neck_maps)
        if training:
            return {"raw_output": raw_head}
        else:
            predictions = self.head_filter(raw_head)
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
            load_weights_backbone(self.backbone, encoder)
            self.backbone.trainable = False

        if dn2tf_head:
            load_weights_backbone(self.neck, neck)
            self.neck.trainable = False
            load_weights_v4head(self.head, decoder)
            self.head.trainable = False

        return


if __name__ == "__main__":
    import tensorflow_datasets as tfds
    from yolo.utils.testing_utils import prep_gpu
    from yolo.training.call_backs.PrintingCallBack import Printer
    prep_gpu() # must be called before loading a dataset
    train, info = tfds.load('coco',
                            split='train',
                            shuffle_files=True,
                            with_info=True)
    test, info = tfds.load('coco',
                           split='validation',
                           shuffle_files=False,
                           with_info=True)

    
    model = Yolov4(model = "regular", policy="mixed_float16", use_tie_breaker=True)
    model.get_summary()
    model.build(model._input_shape)
    model.load_weights_from_dn(dn2tf_head=False)

    train, test = model.process_datasets(train, test, batch_size=2, jitter_im = 0.1, jitter_boxes = 0.005, _eval_is_training = False)
    loss_fn = model.generate_loss(loss_type="ciou")

    #optimizer = ks.optimizers.SGD(lr=1e-3)
    optimizer = ks.optimizers.Adam(lr=1e-3)
    optimizer = model.match_optimizer_to_policy(optimizer)
    model.compile(optimizer=optimizer, loss=loss_fn)

    try:
        model.fit(train, validation_data = test, epochs = 40, verbose = 1, shuffle = True)
        model.save_weights("testing_weights/yolov4/simple_test1")
    except:
        model.save_weights("testing_weights/yolov4/simple_test1_early")


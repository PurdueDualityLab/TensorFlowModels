
import tensorflow as tf
import tensorflow.keras as ks
from yolo.modeling.backbones.csp_backbone_builder import CSP_Backbone_Builder
from yolo.modeling.model_heads._Yolov4Neck import Yolov4Neck
from yolo.modeling.model_heads._Yolov4Head import Yolov4Head
from yolo.modeling.building_blocks import YoloLayer
from yolo.utils.file_manager import download
from yolo.utils import tf_shims
from yolo.utils import DarkNetConverter
from yolo.utils._darknet2tf.load_weights import split_converter, load_weights_dnBackbone, load_weights_dnHead
from yolo.utils._darknet2tf.load_weights2 import load_weights_backbone, load_weights_v4head
import os

__all__ = ['CSPDarkNet53', 'Yolov4']

@ks.utils.register_keras_serializable(package='yolo')
class CSPDarkNet53(ks.Model):
    """The Darknet Image Classification Network Using Darknet53 Backbone"""
    _updated_config = tf_shims.ks_Model___updated_config

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
        super().__init__()
        self.backbone = CSP_Backbone_Builder("darknet53")
        self.head = ks.Sequential([
            ks.layers.GlobalAveragePooling2D(),
            ks.layers.Dense(classes, activation="sigmoid")
        ])
        if load_backbone_weights:
            if config_file is None:
                config_file = download('yolov4.cfg')
            if weights_file is None:
                weights_file = download('yolov4.weights')
            full_model = DarkNetConverter.read(config_file, weights_file)
            encoder, decoder = split_converter(full_model, 106)
            #load_weights_dnBackbone(self.backbone, encoder)
            load_weights_backbone(self.backbone, encoder)
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


@ks.utils.register_keras_serializable(package='yolo')
class Yolov4(ks.Model):
    _updated_config = tf_shims.ks_Model___updated_config
    def __init__(self,
                 input_shape = [None, None, None, 3],
                 model = 'regular',
                 classes = 80,
                 masks = None,
                 boxes = None,
                 policy = "float32",
                 scales = None,
                 path_scale = None,
                 **kwargs):
        """
        Args:
            classes: int for the number of available classes
            masks: a dictionary that gives names to sets of anchor boxes
            boxes: initial sizes of the bounded boxes
            type: the particular type of YOLOv3 model that is being constructed
                  regular, spp, or tiny
        """
        #required_inputs
        super().__init__(**kwargs)
        self._classes = classes
        self._type = model
        self._built = False
        self._input_shape = input_shape

        #setting the running policy
        if type(policy) != str:
            policy = policy.name
        self._og_policy = policy
        self._policy = tf.keras.mixed_precision.experimental.global_policy().name
        self.set_policy(policy=policy)

        #init model params
        if self._type == 'regular':
            self._backbone_name = "darknet53"
            self._neck_name = "name"
            self._head_name = "regular"
            self._model_name = 'yolov4'
            self._encoder_decoder_split_location = 106
            self._boxes = boxes or [(12, 16), (19, 36), (40, 28), (36, 75), (76, 55), (72, 146), (142, 110), (192, 243), (459, 401)]
            self._masks = masks or {"1024": [6,7,8], "512":[3,4,5], "256":[0,1,2]}
            self._x_y_scales = scales or {"1024": 1.05, "512":1.1, "256":1.2}
            self._path_scale = path_scale or {"1024": 32, "512": 16, "256": 8}
        elif self._type == 'tiny':
            self._backbone_name = "darknet_tiny"
            self._head_name = "tiny"
            self._model_name = 'yolov3-tiny'
            self._encoder_decoder_split_location = 14
            self._boxes = boxes or [(10,14),  (23,27),  (37,58), (81,82),  (135,169),  (344,319)]
            self._masks = masks or {"1024": [3,4,5], "256": [0,1,2]}
            self._x_y_scales = scales or {"1024": 1.05, "512":1.1, "256":1.2}
            self._path_scale = path_scale or {"1024": 32, "256": 8}
        else:
            raise ValueError(f"Unknown YOLOv3 type '{self._type}'")

        self._pred_filter = None
        return

    def build(self, input_shape=None):
        self._backbone = CSP_Backbone_Builder(self._backbone_name, input_shape = input_shape)
        self._neck = Yolov4Neck(name = self._neck_name, input_shape= input_shape)
        self._head = Yolov4Head(model = self._head_name, classes=self._classes, boxes=len(self._boxes), input_shape = input_shape)
        self._built = True
        if input_shape is not None and input_shape != self._input_shape:
            self._input_shape = input_shape
        super().build(input_shape)

    def call(self, inputs, training):
        feature_maps = self._backbone(inputs)
        neck_maps = self._neck(feature_maps)
        predictions = self._head(neck_maps)
        if self._pred_filter is not None:
            predictions = self._pred_filter(predictions)
        return predictions

    def load_weights_from_dn(self,
                             dn2tf_backbone = True,
                             dn2tf_neck = True,
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
        if dn2tf_backbone or dn2tf_neck or dn2tf_head:
            if config_file is None:
                config_file = download(self._model_name + '.cfg')
            if weights_file is None:
                weights_file = download(self._model_name + '.weights')
            list_encdec = DarkNetConverter.read(config_file, weights_file)
            encoder, neck, decoder = split_converter(list_encdec, self._encoder_decoder_split_location, 138)

        if not self._built:
            self.build(input_shape = (None, None, None, 3))

        if dn2tf_backbone:
            #load_weights_dnBackbone(self._backbone, encoder, mtype = self._backbone_name)
            load_weights_backbone(self._backbone, encoder)

        if dn2tf_neck:
            load_weights_backbone(self._neck, neck)

        if dn2tf_head:

            load_weights_v4head(self._head, decoder)

        return

    def preprocess_dataset(self, dataset:"Union[str, tfds.data.Dataset]", size:int=None, split='validation'):
        """
        Preprocesses (normalization and data augmentation) and batches the dataset.
        This is a convenience function that calls on
        yolo.dataloaders.preprocessing_functions.preprocessing, replacing the
        parameters with default values based on the anchor boxes and maskes
        passed into __init__.

        Args:
            dataset (str, tfds.data.Dataset): The Dataset you would like to preprocess.
                Can be replaced by the name of a dataset that is present in the
                TensorFlow Dataset library.
            size (int): Size of the dataset. If not specified, the cardinality
                will be calculated and used if it is finite.
            split: The type of split to make to create the dataset if dataset is
                specified as a string.

        Raises:
            SyntaxError:
                - Preprocessing type not found.
                - The given batch number for detection preprocessing is more than 1.
                - Number of batches cannot be less than 1.
                - Data augmentation split cannot be greater than 100.
            ValueError:
                - The dataset has unknown or infinite cardinality.
            WARNING:
                - Dataset is not a tensorflow dataset.
                - Detection Preprocessing may cause NotFoundError in Google Colab.
        """
        from yolo.dataloaders.preprocessing_functions import preprocessing
        if instanceof(dataset, str):
            import tensorflow_datasets as tfds
            dataset, Info = tfds.load(dataset, split=split, with_info=True, shuffle_files=True, download=False)
            if size is None:
                size = int(Info.splits[split].num_examples)
        if size is None:
            try:
                from tensorflow.data.experimental import cardinality
            except ImportError:
                size = dataset.cardinality()
            else:
                size = cardinality(dataset)
        if size < 0:
            raise ValueError("The dataset has unknown or infinite cardinality")
        return preprocessing(dataset, 100, "detection", size, 1, 80, False, anchors=self._boxes, masks=self._masks)

    def generate_loss(self, scale:float = 1.0, loss_type = "giou") -> "Dict[Yolo_Loss]":
        """
        Create loss function instances for each of the detection heads.

        Args:
            scale: the amount by which to scale the anchor boxes that were
                   provided in __init__
        """
        from yolo.modeling.functions.yolo_loss import Yolo_Loss
        loss_dict = {}
        for key in self._masks.keys():
            loss_dict[key] = Yolo_Loss(mask = self._masks[key],
                                       anchors = self._boxes,
                                       scale_anchors = self._path_scale[key],
                                       ignore_thresh = 0.7,
                                       truth_thresh = 1,
                                       loss_type=loss_type,
                                       path_key = key,
                                       scale_x_y=self._x_y_scales[key])
        return loss_dict

    def set_policy(self, policy = 'mixed_float16', save_weights_temp_name = "abn7lyjptnzuj918"):
        print(f"setting policy: {policy}")
        if self._policy == policy:
            return
        else:
            self._policy = policy
        from tensorflow.keras.mixed_precision import experimental as mixed_precision
        policy = mixed_precision.Policy(self._policy)
        mixed_precision.set_policy(policy)
        dtype = policy.compute_dtype
        tf.keras.backend.set_floatx(dtype)

        # save weights and and rebuild model, then load the weights if the model is built
        if self._built:
            self.save_weights(save_weights_temp_name)
            self.build(input_shape=self._input_shape)
            self.load_weights(save_weights_temp_name)
            os.system(f"rm {save_weights_temp_name}.*")
        return

    def set_prediction_filter(self,
            thresh:int = None,
            class_thresh:int = 0.45,
            max_boxes:int = 200,
            use_mixed:bool = True,
            scale_boxes:int = 416,
            scale_mult:float = 1.0):
        if use_mixed:
            self.set_policy(policy='mixed_float16')

        if thresh is None:
            if self._head_name == 'tiny':
                thresh = 0.5
            else:
                thresh = 0.45

        self._pred_filter = YoloLayer(masks = self._masks, anchors= self._boxes, thresh = thresh, cls_thresh = class_thresh, max_boxes = max_boxes, scale_boxes=scale_boxes, scale_mult=scale_mult, path_scale = self._path_scale)
        return

    def remove_prediction_filter(self):
        self.set_policy(policy=self._og_policy)
        self._pred_filter = None
        return

    @property
    def input_image_size(self):
        return self._backbone.input_shape[1:3]

    def get_config(self):
        # used to store/share parameters to reconsturct the model
        return {
            "classes": self._classes,
            "boxes": self._boxes,
            "masks": self._masks,
            "type": self._type,
            "layers": []
        }

    @classmethod
    def from_config(clz, config):
        config = config.copy()
        del config['layers']
        return clz(**config)

if __name__ == '__main__':
    model = Yolov4(model = 'regular', classes=80)
    model.build(input_shape = (None, None, None, 3))
    print(model.generate_loss())
    model.summary()

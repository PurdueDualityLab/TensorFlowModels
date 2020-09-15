
import tensorflow as tf
import tensorflow.keras as ks
from yolo.modeling.backbones.backbone_builder import Backbone_Builder
from yolo.modeling.model_heads._Yolov3Head import Yolov3Head
from yolo.modeling.building_blocks import YoloLayer
from yolo.utils.file_manager import download
from yolo.utils import tf_shims
from yolo.utils import DarkNetConverter
from yolo.utils._darknet2tf.load_weights import split_converter, load_weights_dnBackbone, load_weights_dnHead

__all__ = ['DarkNet53', 'Yolov3']


#@ks.utils.register_keras_serializable(package='yolo')
class DarkNet53(ks.Model):
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


#@ks.utils.register_keras_serializable(package='yolo')
class Yolov3(ks.Model):
    _updated_config = tf_shims.ks_Model___updated_config
    def __init__(self,
                 input_shape = [None, None, None, 3],
                 type = 'regular',
                 classes = 20,
                 masks = None,
                 boxes = None,
                 **kwargs):
        """
        Args:
            classes: int for the number of available classes
            masks: a dictionary that gives names to sets of anchor boxes
            boxes: initial sizes of the bounded boxes
            type: the particular type of YOLOv3 model that is being constructed
                  regular, spp, or tiny
        """

        self._classes = classes
        self._type = type
        self.built = False


        if type == 'regular':
            self._backbone_name = "darknet53"
            self._head_name = "regular"
            self._model_name = 'yolov3'
            self._encoder_decoder_split_location = 76
            self._boxes = boxes or [(10,13),  (16,30),  (33,23), (30,61),  (62,45),  (59,119), (116,90),  (156,198),  (373,326)]
            self._masks = masks or {"1024": [6,7,8], "512":[3,4,5], "256":[0,1,2]}
        elif type == 'spp':
            self._backbone_name = "darknet53"
            self._head_name = "spp"
            self._model_name = 'yolov3-spp'
            self._encoder_decoder_split_location = 76
            self._boxes = boxes or [(10,13),  (16,30),  (33,23), (30,61),  (62,45),  (59,119), (116,90),  (156,198),  (373,326)]
            self._masks = masks or {"1024": [6,7,8], "512":[3,4,5], "256":[0,1,2]}
        elif type == 'tiny':
            self._backbone_name = "darknet_tiny"
            self._head_name = "tiny"
            self._model_name = 'yolov3-tiny'
            self._encoder_decoder_split_location = 14
            self._boxes = boxes or [(10,14),  (23,27),  (37,58), (81,82),  (135,169),  (344,319)]
            self._masks = masks or {"1024": [3,4,5], "256": [0,1,2]}
        else:
            raise ValueError(f"Unknown YOLOv3 type '{type}'")

        self._original_input_shape = input_shape
        self._pred_filter = None
        super().__init__(**kwargs)
        return

    def build(self, input_shape=None):
        self._backbone = Backbone_Builder(self._backbone_name, input_shape = input_shape)
        self._head = Yolov3Head(model = self._head_name, classes=self._classes, boxes=len(self._boxes), input_shape = input_shape)

        self.built = True
        if input_shape is not None and input_shape != self._original_input_shape:
            self._original_input_shape = input_shape
        super().build(input_shape)

    def call(self, inputs):
        feature_maps = self._backbone(inputs)
        predictions = self._head(feature_maps)
        if self._pred_filter is not None:
            predictions = self._pred_filter(predictions)
        return predictions

    def load_weights_from_dn(self,
                             dn2tf_backbone = True,
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
        if dn2tf_backbone or dn2tf_head:
            if config_file is None:
                config_file = download(self._model_name + '.cfg')
            if weights_file is None:
                weights_file = download(self._model_name + '.weights')
            list_encdec = DarkNetConverter.read(config_file, weights_file)
            encoder, decoder = split_converter(list_encdec, self._encoder_decoder_split_location)

        if not self.built:
            net = encoder[0]
            self.build(input_shape = (1, *net.shape))

        if dn2tf_backbone:
            load_weights_dnBackbone(self._backbone, encoder, mtype = self._backbone_name)

        if dn2tf_head:
            load_weights_dnHead(self._head, decoder)

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

    def generate_loss(self, scale:float = 1.0) -> "Dict[Yolo_Loss]":
        """
        Create loss function instances for each of the detection heads.

        Args:
            scale: the amount by which to scale the anchor boxes that were
                   provided in __init__
        """
        from yolo.modeling.functions.yolo_loss import Yolo_Loss
        loss_dict = {}
        for key in masks:
            loss_dict[key] = Yolo_Loss(mask = self._masks[key],
                                anchors = self._boxes,
                                scale_anchors = scale,
                                ignore_thresh = 0.7,
                                truth_thresh = 1,
                                loss_type="giou")
        return loss_dict

    def set_prediction_filter(self,
            thresh:int = None,
            class_thresh:int = 0.45,
            max_boxes:int = 200,
            use_mixed:bool = True,
            scale_boxes:int = 416,
            scale_mult:float = 1.0
    ):
        from tensorflow.keras.mixed_precision import experimental as mixed_precision
        policy = mixed_precision.Policy('mixed_float16' if use_mixed else 'float32')
        mixed_precision.set_policy(policy)
        dtype = policy.compute_dtype

        if thresh is None:
            if self._head_name == 'tiny':
                thresh = 0.5
            else:
                thresh = 0.45

        self._pred_filter = YoloLayer(masks = self._masks, anchors= self._boxes, thresh = thresh, cls_thresh = class_thresh, max_boxes = max_boxes, dtype = dtype, scale_boxes=scale_boxes, scale_mult=scale_mult)

    def remove_prediction_filter(self):
        from tensorflow.keras.mixed_precision import experimental as mixed_precision
        policy = mixed_precision.Policy('float32')
        mixed_precision.set_policy(policy)
        dtype = policy.compute_dtype

        self._pred_filter = None

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
    model = Yolov3(type = 'spp', classes=80)
    model.build(input_shape = (None, None, None, 3))
    model.load_weights_from_dn(dn2tf_backbone = True, dn2tf_head = True, config_file=None, weights_file="yolov3_416.weights")
    model.summary()

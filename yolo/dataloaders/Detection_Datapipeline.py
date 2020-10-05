"""Detection parser."""
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_addons as tfa

from yolo.dataloaders.Parser import Parser
from yolo.dataloaders.decoder import Decoder

from yolo.dataloaders.ops.preprocessing_ops import _scale_image
from yolo.dataloaders.ops.preprocessing_ops import _get_best_anchor
from yolo.dataloaders.ops.preprocessing_ops import _jitter_boxes
from yolo.dataloaders.ops.preprocessing_ops import _translate_image

from yolo.dataloaders.ops.random_ops import _box_scale_rand
from yolo.dataloaders.ops.random_ops import _jitter_rand
from yolo.dataloaders.ops.random_ops import _translate_rand
from yolo.dataloaders.ops.random_ops import _rand_number

from yolo.utils.box_utils import _xcycwh_to_xyxy
from yolo.utils.box_utils import _xcycwh_to_yxyx
from yolo.utils.box_utils import _yxyx_to_xcycwh
from yolo.utils.loss_utils import build_grided_gt


class Detection_Parser(Parser):
    def __init__(self,
                 image_w=416,
                 image_h=416,
                 num_classes=80,
                 fixed_size=False,
                 jitter_im=0.1,
                 jitter_boxes=0.005,
                 net_down_scale=32,
                 max_process_size=608,
                 min_process_size=320,
                 pct_rand=0.5,
                 masks=None,
                 anchors=None):

        self._image_w = image_w
        self._image_h = image_h
        self._num_classes = num_classes
        self._fixed_size = fixed_size
        self._jitter_im = 0.0 if jitter_im == None else jitter_im
        self._jitter_boxes = 0.0 if jitter_boxes == None else jitter_boxes
        self._net_down_scale = net_down_scale
        self._max_process_size = max_process_size
        self._min_process_size = min_process_size
        self._pct_rand = pct_rand
        self._masks = {
            "1024": [6, 7, 8],
            "512": [3, 4, 5],
            "256": [0, 1, 2]
        } if masks == None else masks
        self._anchors = anchors  # use K means to find boxes if it is None
        return
    
    def _parse_train_data(self, data, is_training=True):
        randscale = self._image_w // self._net_down_scale
        if not self._fixed_size:
            randscale = tf.py_function(_box_scale_rand, [
                self._min_process_size // self._net_down_scale,
                self._max_process_size // self._net_down_scale, randscale,
                self._pct_rand
            ], tf.int32)

        if self._jitter_im != 0.0:
            translate_x, translate_y = tf.py_function(_translate_rand,
                                                      [self._jitter_im],
                                                      [tf.float32, tf.float32])
        else:
            translate_x, translate_y = 0.0, 0.0

        if self._jitter_boxes != 0.0:
            j_x, j_y, j_w, j_h = tf.py_function(
                _jitter_rand, [self._jitter_boxes],
                [tf.float32, tf.float32, tf.float32, tf.float32])
        else:
            j_x, j_y, j_w, j_h = 0.0, 0.0, 1.0, 1.0

        image = tf.image.resize(data["image"],
                                size=(randscale * 32,
                                      randscale * 32))  # Random Resize
        image = tf.image.random_brightness(image=image,
                                           max_delta=.1)  # Brightness
        image = tf.image.random_saturation(image=image, lower=0.75,
                                           upper=1.25)  # Saturation
        image = tf.image.random_hue(image=image, max_delta=.1)  # Hue
        image = tf.clip_by_value(image, 0.0, 1.0)

        image = tf.image.resize(image, 
                                size=(randscale * self._net_down_scale,
                                      randscale * self._net_down_scale)) # Random Resize
        image = _translate_image(image, translate_x, translate_y)
        boxes = _jitter_boxes(data["bbox"], translate_x, translate_y, j_x, j_y, j_w, j_h)
        return image, {"source_id": data["source_id"],
                        "bbox": boxes,
                        "classes": data["classes"],
                        "area": data["area"],
                        "is_crowd": data["is_crowd"],
                        "best_anchors": data["best_anchors"], 
                        "width": data["width"],
                        "height": data["height"],
                        "num_detections": data["num_detections"]}

    def _parse_eval_data(self, data):
        randscale = self._image_w // self._net_down_scale
        image = tf.image.resize(data["image"], 
                                size=(randscale * self._net_down_scale,
                                      randscale * self._net_down_scale)) # Random Resize
        image = tf.clip_by_value(image, 0.0, 1.0)
        return image, {"source_id": data["source_id"],
                        "bbox": data["bbox"],
                        "classes": data["classes"],
                        "area": data["area"],
                        "is_crowd": data["is_crowd"],
                        "best_anchors": data["best_anchors"], 
                        "width": data["width"],
                        "height": data["height"],
                        "num_detections": data["num_detections"]}

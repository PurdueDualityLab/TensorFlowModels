import tensorflow as tf
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import copy

from .preprocessing_ops import _scale_image, _get_yolo_box, _get_best_anchor, _jitter_boxes, _translate_image, _get_tf_box, _build_grided_gt
from .random_ops import _box_scale_rand, _jitter_rand, _translate_rand

class YoloParser():
    def __init__(self, 
                image_w = 416,
                image_h = 416, 
                num_classes = 80, 
                fixed_size = False, 
                jitter_im = 0.1,
                jitter_boxes = 0.005, 
                net_down_scale = 32,
                path_scales = None,  
                max_process_size = 608, 
                min_process_size = 320, 
                pct_rand = 0.5, 
                use_tie_breaker = True,
                masks = None,
                anchors = None):

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
        self._path_scales = path_scales
        self._use_tie_breaker = use_tie_breaker
        self._masks = {"1024": [6,7,8], "512":[3,4,5], "256":[0,1,2]} if masks == None else masks
        self._anchors = anchors # use K means to find boxes if it is None
        return 
    
    def _unbatched_processing(self, data):
        image = _scale_image(data["image"], square = True, square_w = self._max_process_size)
        boxes = _get_yolo_box(data["objects"]["bbox"])
        classes = tf.one_hot(data["objects"]["label"], depth = self._num_classes)
        best_anchor = _get_best_anchor(boxes, self._anchors, self._image_w)
        return {"image": image, "bbox": boxes, "label": classes, "best_anchor": best_anchor}
    
    def _batched_processing(self, data, is_training = True):
        randscale = self._image_w//self._net_down_scale
        if not self._fixed_size and is_training:
            randscale = tf.py_function(_box_scale_rand, [self._min_process_size//self._net_down_scale, 
                                                         self._max_process_size//self._net_down_scale,
                                                         randscale, 
                                                         self._pct_rand], tf.int32)

        if self._jitter_im != 0.0 and is_training: 
            translate_x, translate_y = tf.py_function(_translate_rand, [self._jitter_im], [tf.float32, tf.float32])
        else:
            translate_x, translate_y = 0.0, 0.0
        
        if self._jitter_boxes != 0.0 and is_training: 
            j_x, j_y, j_w, j_h = tf.py_function( _jitter_rand, [self._jitter_boxes], [tf.float32, tf.float32, tf.float32, tf.float32])
        else:
            j_x, j_y, j_w, j_h = 0.0, 0.0, 1.0, 1.0
        
        image = tf.image.resize(data["image"], size = (randscale * 32, randscale * 32)) # Random Resize
        image = tf.image.random_brightness(image=image, max_delta=.1) # Brightness
        image = tf.image.random_saturation(image=image, lower = 0.75, upper=1.25) # Saturation
        image = tf.image.random_hue(image=image, max_delta=.1) # Hue
        image = tf.clip_by_value(image, 0.0, 1.0)

        image = tf.image.resize(image, size = (randscale * self._net_down_scale, randscale * self._net_down_scale))
        image = _translate_image(image , translate_x, translate_y)
        boxes = _jitter_boxes(data["bbox"], translate_x, translate_y, j_x, j_y, j_w, j_h)
        label = tf.concat([boxes, data["label"], data["best_anchor"]], axis = -1)
        return {"image": image, "label": label, "randscale": randscale}

    def _label_format_gt(self, data):
        masks = self._masks
        randscale = data["randscale"]
        for key in masks.keys():
            masks[key] = _build_grided_gt(data["label"], tf.convert_to_tensor(self._masks[key], dtype= tf.float32), randscale, self._use_tie_breaker)
            if self._path_scales == None:
                randscale *= 2
            else:
                randscale = randscale * self._path_scales[key]
        return data["image"], masks

    def unbatched_process_fn(self, is_training):
        def parse(tensor_set):
            return self._unbatched_processing(tensor_set)
        return parse
    
    def batched_process_fn(self, is_training):
        def parse(tensor_set):
            return self._batched_processing(tensor_set, is_training = is_training)
        return parse
    
    def build_gt(self, is_training):
        def parse(tensor_set):
            return self._label_format_gt(tensor_set)
        return parse

if __name__ == "__main__":
    import tensorflow_datasets as tfds
    coco, info = tfds.load('coco', split = 'train', with_info= True)

    parser = YoloParser(use_tie_breaker=False, anchors=[(10,13),  (16,30),  (33,23), (30,61),  (62,45),  (59,119), (116,90),  (156,198),  (373,326)])
    process_1 = parser.unbatched_process_fn(is_training = True)
    process_2 = parser.batched_process_fn(is_training = False)
    process_3 = parser.build_gt(is_training = True)
    coco = coco.map(process_1).padded_batch(10)
    coco = coco.map(process_2)
    coco = coco.map(process_3)

    for k in coco.take(10):
        print()
        # boxes = _get_tf_box(k["label"][..., :4])
        # image = tf.image.draw_bounding_boxes(k["image"], boxes, [[1.0, 0.0, 0.0]])
        # tf.print(k["randscale"], k["label"][..., :4])
        # plt.imshow(image.numpy()[0])
        # plt.show()
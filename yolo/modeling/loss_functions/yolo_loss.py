import tensorflow as tf 
import tensorflow.keras as ks 
from tensorflow.keras import backend as K

def array_index_list(index, array):
    return [array[i] for i in index]


'''
y_pred structure: [...class..., p(object), b_x, b_y, b_w, b_h]

example of 1 bounding box with 3 classes and a shape of h = 2, w = 2,

full_vector = [[[c_1_1, c_1_2],
              [c_1_3, c_1_4]],
              [[c_2_1, c_2_2],
              [c_2_3, c_2_4]],
              [[c_3_1, c_3_2],
              [c_3_3, c_3_4]],

              [[p_1, p_2],
              [p_3, p_4]],

              [[bx_1, bx_2],
              [bx_3, bx_4]],
              [[by_1, by_2],
              [by_3, by_4]],
              [[bw_1, bw_2],
              [bw_3, bw_4]],
              [[bh_1, bh_2],
              [bh_3, bh_4]]]

depth_only = [class_1, class_2, class_3, p(object), b_x, b_y, b_w, b_h]

if more than n bounding boxs concatnate full vector n times 

for m batches, we need add a dimention to the front [m, None, None, n * (#classes + 4 + 1)]
'''


class Yolo_Loss(ks.losses.Loss):
    def __init__(self, mask, anchors, classes, num, ignore_thresh, truth_thresh, random ,reduction = tf.keras.losses.Reduction.AUTO, name=None, **kwargs):
        self._anchors = array_index_list(mask, anchors)
        self._classes = classes
        self._num = len(mask)
        self._ignore_thresh = ignore_thresh
        self._truth_thresh = truth_thresh
        self._iou_thresh = 1 # recomended use = 0.213 in [yolo]
        self._random = 0 if random == None else random

        self._batch_size = None
        self._vector_shape = self._num*(self._classes + 4 + 1)
        print([self._batch_size, None, None, self._vector_shape])
        super(Yolo_Loss, self).__init__(reduction = reduction, name = name)
        pass

    def _get_splits(self, tensor):
        return 

    def call(self, y_true, y_pred):
        pass 

    def get_config(self):
        pass

k = Yolo_Loss(mask = [3, 4, 5], 
              anchors = [(10,13),  (16,30),  (33,23),  (30,61),  (62,45),  (59,119),  (116,90),  (156,198),  (373,326)], 
              classes = 80, 
              num = 9, 
              ignore_thresh = 0.7,
              truth_thresh = 1, 
              random = 1)

ones = tf.ones_initializer()
n = 3
classes = 3

x = tf.Variable(initial_value = ones((2, 13, 13, n * (classes + 5)), dtype = tf.float32))
y = tf.Variable(initial_value = ones((2, 13, 13, n * (classes + 5)), dtype = tf.float32))

# option_find_float_quiet(linked_list, key, defaults_value) -> same as above but print nothing
# option_find_float(linked_list, key, defaults_value) -> same as above but print nothing
# option_find_int_quiet(linked_list, key, defaults_value) -> same as above but print nothing
# option_find_int(linked_list, key, defaults_value) -> same as above but print nothing

# find the value with the linked_list key, if nothing is found return default_value
# if _quite in the name, print nothing


# lw = layer input width 
# lh = layer input hight 
# h = image hight 
# w = image width
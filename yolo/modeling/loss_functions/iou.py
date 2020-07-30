import tensorflow as tf
import tensorflow.keras as ks
from tensorflow.keras import backend as K
'''
float delta_yolo_box(box truth, float *x, float *biases, int n, int index, int i, int j, int lw, int lh, int w, int h, float *delta, float scale, int stride)
{
    box pred = get_yolo_box(x, biases, n, index, i, j, lw, lh, w, h, stride);
    float iou = box_iou(pred, truth);

    float tx = (truth.x*lw - i);
    float ty = (truth.y*lh - j);
    float tw = log(truth.w*w / biases[2*n]);
    float th = log(truth.h*h / biases[2*n + 1]);

    delta[index + 0*stride] = scale * (tx - x[index + 0*stride]);
    delta[index + 1*stride] = scale * (ty - x[index + 1*stride]);
    delta[index + 2*stride] = scale * (tw - x[index + 2*stride]);
    delta[index + 3*stride] = scale * (th - x[index + 3*stride]);
    return iou;
}
lw, lh layer width, layer height
w, h width, height
biases
'''

@tf.function
def overlap(x1, w1, x2, w2):
    l1 = x1 - w1 / 2
    l2 = x2 - w2 / 2
    left = K.maximum(l1, l2)
    '''
    if l1 < l2:
        left = l2
    else:
        left = l1
    right = 0
    '''
    r1 = x1 + w1 / 2
    r2 = x2 + w2 / 2
    '''
    if r1 < r2:
        right = r1
    else:
        right = r2
    '''
    right = K.minimum(r1, r2)
    return right - left

@tf.function
def intersection(box_pred, box_truth):
    #box = [x, y, width, height]
    w = overlap(box_pred[0], box_pred[2], box_truth[0], box_truth[2])
    h = overlap(box_pred[1], box_pred[3], box_truth[1], box_truth[3])
    w = K.relu(w)
    h = K.relu(h)
    return w * h

@tf.function
def area(box):
    return box[2] * box[3]

@tf.function
def union(box_pred, box_truth):
    inter = intersection(box_pred, box_truth)
    return area(box_pred) + area(box_truth) - inter

@tf.function
def iou(box_pred, box_truth):
    return intersection(box_pred, box_truth) / union(box_pred, box_truth)
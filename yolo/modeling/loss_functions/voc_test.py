import tensorflow_datasets as tfds
import tensorflow as tf
import tensorflow.keras.backend as K
import sys
import numpy as np
from yolo.modeling.loss_functions.iou import *


RANDOM_SEED = tf.random.Generator.from_seed(int(np.random.uniform(low=300, high=9000)))

@tf.function
def convert_to_yolo(box):
    with tf.name_scope("convert_box"):
        ymin, xmin, ymax, xmax = tf.split(box, 4, axis = -1)
        # add a dimention check
        x_center = (xmax + xmin)/2
        y_center = (ymax + ymin)/2
        width = xmax - xmin
        height = ymax - ymin
        box = tf.concat([x_center, y_center, width, height], axis = -1)
    return box

@tf.function
def build_yolo_box(image, boxes):
    box_list = []
    with tf.name_scope("yolo_box"):
        image = tf.convert_to_tensor(image)
        boxes = convert_to_yolo(boxes)
    return boxes

def build_gt(y_true, anchors, size):
    size = tf.cast(size, dtype = tf.float32)

    anchor_xy = y_true[..., 0:2]
    true_wh = y_true[..., 2:4]

    anchors = tf.convert_to_tensor(anchors, dtype=tf.float32)/size
    anchors = tf.transpose(anchors, perm=[1, 0])
    anchor_xy = tf.tile(tf.expand_dims(anchor_xy, axis = -1), [1,1, tf.shape(anchors)[-1]])
    anchors = tf.tile(tf.expand_dims(anchors, axis = 0), [tf.shape(anchor_xy)[0], 1, 1])
    anchors = K.concatenate([anchor_xy, anchors], axis = 1)
    anchors = tf.transpose(anchors, perm = [2, 0, 1])

    truth_comp = tf.tile(tf.expand_dims(y_true[..., 0:4], axis = -1), [1,1, tf.shape(anchors)[0]])
    truth_comp = tf.transpose(truth_comp, perm = [2, 0, 1])

    box = tf.split(truth_comp, num_or_size_splits=4, axis = -1)
    anchor_box = tf.split(anchors, num_or_size_splits=4, axis = -1)    

    iou_anchors = tf.cast(K.argmax(iou(box, anchor_box), axis = 0), dtype = tf.float32)
    y_true = K.concatenate([y_true, iou_anchors], axis = -1)
    return y_true

def preprocess(data, anchors, width, height):
    #image
    image = tf.cast(data["image"], dtype=tf.float32)
    image = tf.image.resize(image, size = (width, height))

    # deal with jitter here

    # box, variable to get shape
    boxes = data["objects"]["bbox"]
    boxes = build_yolo_box(image, boxes)
    # classes = tf.one_hot(data["objects"]["label"], depth = 20)
    classes = tf.cast(tf.expand_dims(data["objects"]["label"], axis = -1), dtype = tf.float32)
    label = tf.concat([boxes, classes], axis = -1)

    # constant
    label = build_gt(label, anchors, width)
    
    #idk if this works
    label = tf.random.shuffle(label, seed=None)

    return image, label

def py_func_rand():
    jitter = np.random.uniform(low = -0.3, high = 0.3)
    randscale = np.random.randint(low = 10, high = 19) 
    return jitter, randscale

@tf.function
def get_random(image, label, masks):
    masks = tf.convert_to_tensor(masks, dtype= tf.float32)
    jitter, randscale = tf.py_function(py_func_rand, [], [tf.float32, tf.int32])
    image = tf.image.resize(image, size = (randscale * 32, randscale * 32))
    
    # bounding boxs
    if tf.math.equal(tf.shape(masks)[0], 3):
        value1 = build_grided_gt(label, masks[0], randscale)
        #tf.print()
        value2 = build_grided_gt(label, masks[1], randscale * 2)
        #tf.print()
        value3 = build_grided_gt(label, masks[2], randscale * 4)
        ret_dict = {1024: value1, 512: value2, 256: value3}
    elif tf.math.equal(tf.shape(masks)[0], 2):
        value1 = build_grided_gt(label, masks[0], randscale)
        value2 = build_grided_gt(label, masks[1], randscale * 4)
        ret_dict = {1024: value1, 512: value2, 256: value2}
    else:
        value1 = build_grided_gt(label, masks[0], randscale)
        value2 = build_grided_gt(label, masks[1], randscale * 2)
        value3 = build_grided_gt(label, masks[2], randscale * 4)
        ret_dict = {1024: value1, 512: value2, 256: value3}

    # we need to get this to work!!!!
    # loss_dict = []
    # index = 1024
    # for i in range(tf.shape(masks)[0]):
    #     loss_dict.append(build_grided_gt(label, masks[i], randscale))
    #     randscale *= 2
    #     index //= 2
    #tf.print(ret_dict)
    return image, ret_dict

@tf.function
def build_grided_gt(y_true, mask, size):
    batches = tf.shape(y_true)[0]
    num_boxes = tf.shape(y_true)[1]
    len_masks = tf.shape(mask)[0]

    finshape = tf.convert_to_tensor([batches, size, size, len_masks * tf.shape(y_true)[-1]])
    full = tf.zeros([batches, size, size, len_masks, tf.shape(y_true)[-1]])
    depth_track = tf.zeros((batches, size, size, len_masks), dtype=tf.int32)

    x = tf.cast(y_true[..., 0] * tf.cast(size, dtype = tf.float32), dtype = tf.int32)
    y = tf.cast(y_true[..., 1] * tf.cast(size, dtype = tf.float32), dtype = tf.int32)
    anchors = tf.repeat(tf.expand_dims(y_true[..., -1], axis = -1), len_masks, axis = -1)

    update_index = tf.TensorArray(tf.int32, size=0, dynamic_size=True)
    update = tf.TensorArray(tf.float32, size=0, dynamic_size=True)

    i = 0
    for batch in range(batches):
        for box_id in range(num_boxes):
            if tf.math.equal(y_true[batch, box_id, 3], 0) :
                continue
            index = tf.math.equal(anchors[batch, box_id], mask)
            if K.any(index):
                p = tf.cast(K.argmax(tf.cast(index, dtype = tf.int8)), dtype = tf.int32)
                
                # start code for tie breaker, temp check performance
                uid = 1
                used = depth_track[batch, x[batch, box_id], y[batch, box_id], p]
                count = 0
                while tf.math.equal(used, 1) and tf.math.less(count, 3):
                    uid = 2
                    count += 1
                    p = (p + 1)%3
                    used = depth_track[batch, x[batch, box_id], y[batch, box_id], p]
                    
                if tf.math.equal(used, 1):
                    tf.print("skipping")
                    continue
                depth_track = tf.tensor_scatter_nd_update(depth_track, [(batch, x[batch, box_id], y[batch, box_id], p)], [uid])
                #end code for tie breaker

                update_index = update_index.write(i, [batch, x[batch, box_id], y[batch, box_id], p])
                test = K.concatenate([y_true[batch, box_id, 0:4], tf.convert_to_tensor([1.]), y_true[batch, box_id, 4:-1]])
                update = update.write(i, test)
                i += 1

            """
            used can be:
                0 not used
                1 used with the correct anchor
                2 used with offset anchor
            
            if used is 0 or 2: 
                do not enter tie breaker (count = 0)
                edit box index with the most recent box

            if tie breaker was used:
                set used to 2
            else:
                set used to 1

            E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:741] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
            raised likely due to a memory issue? reduced batch size to 2 and it solved the problem? odd

            W tensorflow/core/grappler/optimizers/loop_optimizer.cc:906] Skipping loop optimization for Merge node with control input: cond/branch_executed/_11
            idk should look into this

            18 seconds for 2000 images 
            """

    if tf.math.greater(update_index.size(), 0):
        update_index = update_index.stack()
        update = update.stack()
        full = tf.tensor_scatter_nd_add(full, update_index, update)
    #tf.print(K.sum(full))
    full = tf.reshape(full, finshape)
    #tf.print(K.sum(full))
    return full

def load_dataset(skip = 0, batch_size = 10):
    dataset,info = tfds.load('voc', split='train', with_info=True, shuffle_files=True)
    #dataset = dataset.skip(skip).take(batch_size * 100)
    dataset = dataset.map(lambda x: preprocess(x, [(10,13),  (16,30),  (33,23),  (30,61),  (62,45),  (59,119),  (116,90),  (156,198),  (373,326)], 416, 416)).padded_batch(batch_size)
    dataset = dataset.map(lambda x, y: get_random(x, y, masks = [[0, 1, 2],[3, 4, 5],[6, 7, 8]]))
    return dataset


if __name__ == "__main__":
    print("")
    #with tf.device("/CPU:0"):
    dataset = load_dataset()

    import time
    start = time.time()
    for image, label in dataset:
        for key in label.keys():
            # for k in label[key]:
            #     for j in k:
            #         for i in j:
            #             if K.sum(i) > 0:
            #                 print(i.numpy().tolist(), end = "\n")
            print(label[key].shape)
        count = 0
    end = time.time() - start
    print(end)

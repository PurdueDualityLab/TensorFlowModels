import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_addons as tfa
from tensorflow_addons.image import utils as img_utils
import matplotlib.pyplot as plt
import numpy as np
import sys
import time
import tensorflow.keras.backend as K


# Global Variable to introduce randomness among each element of a batch
RANDOM_SEED = tf.random.Generator.from_seed(int(np.random.uniform(low=300, high=9000)))


def image_scaler(image):
    """Image Normalization.
    Args:
        image(tensorflow.python.framework.ops.Tensor): The image.
    Returns:
        A Normalized Function.
    """
    image = tf.convert_to_tensor(image)
    image = image / 255
    return image

def py_func_rand():
    """Image Normalization.
    Returns:
        jitter(tensorflow.python.framework.ops.Tensor): A random number generated
            from a uniform distrubution between -0.3 and 0.3.
        randscale(tensorflow.python.framework.ops.Tensor): A random integer between
            -10 and 19.
    """
    jitter = np.random.uniform(low = -0.3, high = 0.3)
    randscale = np.random.randint(low = 10, high = 19)
    return jitter, randscale

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
            if tf.math.equal(y_true[batch, box_id, 3], 0): # VALUE ERROR is (None, 25)
                continue
            index = tf.math.equal(anchors[batch, box_id], mask)
            if K.any(index):
                if  (x[batch, box_id] >= 0 and x[batch, box_id] <= Size) and (y[batch, box_id] >= 0 and y[batch, box_id] <= 1): 
                    p = tf.cast(K.argmax(tf.cast(index, dtype = tf.int8)), dtype = tf.int32)

                    # # start code for tie breaker, temp check performance
                    # uid = 1
                    # used = depth_track[batch, x[batch, box_id], y[batch, box_id], p]
                    # count = 0
                    # while tf.math.equal(used, 1) and tf.math.less(count, 3):
                    #     uid = 2
                    #     count += 1
                    #     p = (p + 1)%3
                    #     used = depth_track[batch, x[batch, box_id], y[batch, box_id], p]
                    # if tf.math.equal(used, 1):
                    #     tf.print("skipping")
                    #     continue
                    # depth_track = tf.tensor_scatter_nd_update(depth_track, [(batch, x[batch, box_id], y[batch, box_id], p)], [uid])
                    # #end code for tie breaker

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
    
    tf.print("gtsum 1", K.sum(y_true))
    tf.print("gtsum full",K.sum(full))
    return full

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
def box_iou(box_1, box_2):
    # K.expand_dims()
    box1_xy = box_1[..., :2]
    box1_wh = box_1[..., 2:4]
    box1_mins = box1_xy - box1_wh / 2.
    box1_maxes = box1_xy + box1_wh / 2.

    box2_xy = box_2[..., :2]
    box2_wh = box_2[..., 2:4]
    box2_mins = box2_xy - box2_wh / 2.
    box2_maxes = box2_xy + box2_wh / 2.

    intersect_mins = K.maximum(box1_mins, box2_mins)
    intersect_maxes = K.minimum(box1_maxes, box2_maxes)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, K.zeros_like(intersect_mins))
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    box1_area = box1_wh[..., 0] * box1_wh[..., 1]
    box2_area = box2_wh[..., 0] * box2_wh[..., 1]
    iou = tf.math.divide_no_nan(intersect_area, (box1_area + box2_area - intersect_area))
    return iou

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

    iou_anchors = tf.cast(K.argmax(box_iou(truth_comp, anchors), axis = 0), dtype = tf.float32)

    #tf.print(iou_anchors)
    y_true = K.concatenate([y_true, K.expand_dims(iou_anchors, axis = -1)], axis = -1)
    return y_true

def _angles_to_projective_transforms(angle, image_w, image_h):
    """Generate projective transform matrix for tfa.image.transform.
    Args:
        angle(tensorflow.python.framework.ops.EagerTensor): The rotation angle.
        image_w(tensorflow.python.framework.ops.EagerTensor): The width of the image.
        image_h(tensorflow.python.framework.ops.EagerTensor): The height of the image.
    Returns:
        projective transform matrix(tensorflow.python.framework.ops.EagerTensor)
    """
    with tf.name_scope("rotate_parent"):
        angle_or_angles = tf.convert_to_tensor(angle, name="angles", dtype=tf.dtypes.float32)
        angles = angle_or_angles[None]
        x_offset = ((image_w - 1) - (tf.math.cos(angles) * (image_w - 1) - tf.math.sin(angles) * (image_h - 1))) / 2.0
        y_offset = ((image_h - 1)- (tf.math.sin(angles) * (image_w - 1) + tf.math.cos(angles) * (image_h - 1))) / 2.0
        num_angles = tf.shape(angles)[0]
    return tf.concat([tf.math.cos(angles)[:, None],-tf.math.sin(angles)[:, None],x_offset[:, None],tf.math.sin(angles)[:, None],tf.math.cos(angles)[:, None],y_offset[:, None],tf.zeros((1, 2))],axis=1)

def _rotate(image, angle):
    """Generates a rotated image with the use of tfa.image.transform
    Args:
        image(tensorflow.python.framework.ops.Tensor): The image.
        angle(tensorflow.python.framework.ops.EagerTensor): The rotation angle.
    Returns:
        The rotated image.
    """
    with tf.name_scope("rotate"):
        image = tf.convert_to_tensor(image)
        img = img_utils.to_4D_image(image)
        ndim = image.get_shape().ndims
        image_h = tf.cast(img.shape[0], tf.dtypes.float32)
        image_w = tf.cast(img.shape[1], tf.dtypes.float32)
        rotation_key = _angles_to_projective_transforms(angle, image_w, image_h)
        output = tfa.image.transform(img, rotation_key, interpolation="NEAREST")
    return img_utils.from_4D_image(output, ndim)

def _rand_number(low, high):
    """Generates a random number along a uniform distrubution.
    Args:
        low(tensorflow.python.framework.ops.Tensor): Minimum Value of the Distrubution.
        high(tensorflow.python.framework.ops.EagerTensor): Maximum Value of the Distrubution.
    Returns:
        A tensor of the specified shape filled with random uniform values.
    """
    global RANDOM_SEED # Global Variable defined at the beginning of the file.
    return RANDOM_SEED.uniform(minval= low, maxval= high, shape = (), dtype=tf.float32)

def _classification_data_augmentation(datapoint, num_of_classes):
    """Augments image by performing Random Zoom, Resize with Pad, Random Rotate,
        Random Brightness Distortion, Random Saturation Distortion, Random Hue Distortion
        and finally normalizing the image.
    Args:
        datapoint (dict): A Dictionaty that holds the image as well as other relevant
            information.
    Returns:
        Either Image and Label or Image and Object.
    """
    # Data Augmentation values intializations.
    image = datapoint['image']
    image = tf.cast(image, tf.float32)
    w = tf.cast(tf.shape(image)[0], tf.float32)
    h = tf.cast(tf.shape(image)[1], tf.int32)
    low = tf.cast(128, tf.dtypes.float32)[None]
    high = tf.cast(448, tf.dtypes.float32)[None]
    scale = tf.py_function(_rand_number, [low, high], [tf.float32])
    aspect = tf.py_function(_rand_number, [.75, 1.25], [tf.float32])
    deg = tf.py_function(_rand_number, [-7.0, 7.0], [tf.float32])
    scale = tf.cast(scale, dtype= tf.int32)[0][0]
    deg = tf.cast(deg, dtype=tf.float32)[0]
    aspect = tf.cast(aspect, dtype=tf.float32)[0]
    nh = tf.cast(w/aspect, dtype= tf.int32)
    nw = tf.cast(w, dtype= tf.int32)
    # Data Augmentation Functions.
    image = tf.image.resize(image, size = (nw, nh))
    image = tf.image.resize_with_crop_or_pad(image, target_height = scale, target_width = scale) # Zoom
    image = tf.image.resize_with_pad(image, target_width=224, target_height=224) # Final Output Shape
    image = _rotate(image, deg) # Rotate
    image = tf.image.random_brightness(image=image, max_delta=.75) # Brightness
    image = tf.image.random_saturation(image=image, lower = 0.75, upper=1.25) # Saturation
    image = tf.image.random_hue(image=image, max_delta=.1) # Hue
    image = tf.clip_by_value(image / 255, 0, 1) # Normalize
    if "objects" in datapoint:
        return image, datapoint['objects']
    else:
        return image, tf.one_hot(datapoint['label'],num_of_classes)

def _priming_data_augmentation(datapoint, num_of_classes):
    """Augments image by performing Random Zoom, Resize with Pad, and
        finally normalizing the image.
    Args:
        datapoint (dict): A Dictionaty that holds the image as well as other relevant
            information.
    Returns:
        Either Image and Label or Image and Object.
    """
    # Data Augmentation values intializations.
    image = datapoint['image']
    image = tf.cast(image, tf.float32)
    w = tf.cast(tf.shape(image)[0], tf.float32)
    h = tf.cast(tf.shape(image)[1], tf.int32)
    low = tf.cast(448, tf.dtypes.float32)[None]
    high = tf.cast(512, tf.dtypes.float32)[None]
    scale = tf.py_function(_rand_number, [low, high], [tf.float32])
    scale = tf.cast(scale, dtype= tf.int32)[0][0]
    # Data Augmentation Functions.
    image = tf.image.resize_with_crop_or_pad(image, target_height = scale, target_width = scale) # Zoom
    image = tf.image.resize_with_pad(image, target_width=448, target_height=448) # Final Output Shape
    image = image / 255 #Normalize
    if "objects" in datapoint:
        return image, datapoint['objects']
    else:
        return image, tf.one_hot(datapoint['label'],num_of_classes)

def _detection_data_augmentation(image, label, masks):
    masks = tf.convert_to_tensor(masks, dtype= tf.float32)
    # Image Jitter
    jitter, randscale = tf.py_function(py_func_rand, [], [tf.float32, tf.int32])
    # image_jitter = tf.concat([jitter, jitter], axis = 0)
    # image_jitter.set_shape([2])
    # image = tfa.image.translate(image, image_jitter)
    # # Bounding Box Jitter
    # tf.print(tf.shape(label))
    # x = tf.math.add(label[..., 0], jitter)
    # x = tf.expand_dims(x, axis = -1)
    # tf.print(tf.shape(x))
    # y = tf.math.add(label[..., 1], jitter)
    # y = tf.expand_dims(y, axis = -1)
    # tf.print(tf.shape(y))
    # rest = label[..., 2:]
    # tf.print(tf.shape(rest))
    # label = tf.concat([x,y,rest], axis = -1)
    # tf.print(tf.shape(label))
    # Other Data Augmentation
    image = tf.image.resize(image, size = (randscale * 32, randscale * 32)) # Random Resize
    image = tf.image.random_brightness(image=image, max_delta=.1) # Brightness
    image = tf.image.random_saturation(image=image, lower = 0.75, upper=1.25) # Saturation
    image = tf.image.random_hue(image=image, max_delta=.1) # Hue
    # building bounding boxs
    #! SUBJECT TO CHANGE ***
    #! -----------------
    if tf.math.equal(tf.shape(masks)[0], 3):
        value1 = build_grided_gt(label, masks[0], randscale)
        value2 = build_grided_gt(label, masks[1], randscale * 2)
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

    return image, ret_dict

def _normalize(datapoint, h, w, num_of_classes):
    """Normalizes the image by resizing it to the desired output shape
    Args:
        datapoint (dict): A Dictionaty that holds the image as well as other relevant
            information.
    Returns:
        normalize (dict): A Normalized Image alongside the mapped information.
    """
    image = datapoint['image']
    image = tf.cast(image, tf.float32)
    image = tf.image.resize_with_pad(image, target_width=h, target_height=w) # Final Output Shape
    image = image / 255 # Normalize
    if "objects" in datapoint:
        return image, datapoint['objects']
    else:
        return image, tf.one_hot(datapoint['label'],num_of_classes)

def _detection_normalize(data, anchors, width, height):
    """Normalizes the image by doing random resizing required for detection.
    Args:
        datapoint (dict): A Dictionaty that holds the image as well as other relevant
            information.
    Returns:
        normalize (dict): A Normalized Image alongside the mapped information.
    """
    image = tf.cast(data["image"], dtype=tf.float32)
    image = tf.image.resize(image, size = (608, 608))
    boxes = data["objects"]["bbox"]
    boxes = build_yolo_box(image, boxes)
    classes = tf.one_hot(data["objects"]["label"], depth = 80)
    label = tf.concat([boxes, classes], axis = -1)
    label = build_gt(label, anchors, width)
    image = image / 255 # Normalize
    return image, label

def preprocessing(dataset, data_augmentation_split, preprocessing_type, size, batch_size, num_of_classes, shuffle_flag):
    """Preprocesses (normalization and data augmentation) and batches the dataset.
    Args:
        dataset (tfds.data.Dataset): The Dataset you would like to preprocess.
        data_augmentation_split (int): The percentage of the dataset that is data
            augmented.
        preprocessing_type (str): The type of preprocessing should be conducted
            and is dependent on the type of training.
        size (int): The size of the dataset being passed into preprocessing.
        batch_size (int): The size of the each batch.
        num_of_classes (int): The number of classes found within the dataset.
        shuffle_flag (bool): This is a Flag that determines whether to or not to shuffle
            within the function.
    Returns:
        dataset (tfds.data.Dataset): A shuffled dataset that includes images that
            have been data augmented
    Raises:
        SyntaxError:
            - Preprocessing type not found.
            - The given batch number for detection preprocessing is more than 1.
            - Number of batches cannot be less than 1.
            - Data augmentation split cannot be greater than 100.
        TypeError:
            - Data augmentation split must be an integer.
            - Preprocessing type must be an string.
            - Size must be an integer.
            - Number of batches must be an integer.
            - Shuffle flag must be a boolean.
        WARNING:
            - Dataset is not a tensorflow dataset.
            - Detection Preprocessing may cause NotFoundError in Google Colab.
    """
    # TypeError Raising
    if  hasattr(dataset, 'element_spec')== False:
        print("WARNING: Dataset may not a tensorflow dataset.")
    if type(data_augmentation_split) is not int:
        raise TypeError("Data augmentation split must be an integer.")
    if type(preprocessing_type) is not str:
        raise TypeError("Preprocessing type must be an string.")
    if type(size) is not int:
        raise TypeError("Size must be an integer.")
    if type(batch_size) is not int:
        raise TypeError("Number of batches must be an integer.")
    if type(shuffle_flag) is not bool:
        raise TypeError("Shuffle flag must be a boolean.")
    # SyntaxError Raising
    if preprocessing_type.lower() != "detection" and preprocessing_type.lower() != "classification" and preprocessing_type.lower() != "priming":
        raise SyntaxError("Preprocessing type not found.")
    if batch_size < 1:
        raise SyntaxError("Batch Size cannot be less than 1.")
    if data_augmentation_split > 100:
        raise SyntaxError("Data augmentation split cannot be greater than 100.")
    if 'google.colab' in sys.modules == True and preprocessing_type.lower() != "detection":
        print("WARNING: Detection Preprocessing may cause NotFoundError in Google Colab. Please try running on local machine.")

    # Spliting the dataset based off of user defined split.
    data_augmentation_split = int((data_augmentation_split/100)*size)
    non_preprocessed_split = size - data_augmentation_split
    data_augmentation_dataset = dataset.take(data_augmentation_split)
    remaining = dataset.skip(data_augmentation_split)
    non_preprocessed_split = remaining.take(non_preprocessed_split)
    # Data Preprocessing functions based off of selected preprocessing type.
    
    # Detection Preprocessing
    if preprocessing_type.lower() == "detection":
        dataset = data_augmentation_dataset.concatenate(non_preprocessed_split)
        if shuffle_flag == True:
            dataset = dataset.shuffle(size)
        dataset = dataset.map(lambda x: _detection_normalize(x, [(10,13),  (16,30),  (33,23),  (30,61),  (62,45),  (59,119),  (116,90),  (156,198),  (373,326)], 416, 416), num_parallel_calls = tf.data.experimental.AUTOTUNE).batch(int(batch_size))
        dataset = dataset.map(lambda x, y: _detection_data_augmentation(x, y, masks = [[0, 1, 2],[3, 4, 5],[6, 7, 8]]), num_parallel_calls = tf.data.experimental.AUTOTUNE).prefetch(tf.data.experimental.AUTOTUNE)
    # Classification Preprocessing
    elif preprocessing_type.lower() == "classification":
        # Preprocessing functions applications.
        non_preprocessed_split = non_preprocessed_split.map(lambda x: _normalize(x, 224, 224, num_of_classes), num_parallel_calls = tf.data.experimental.AUTOTUNE)
        data_augmentation_dataset = data_augmentation_dataset.map(lambda x: _classification_data_augmentation(x, num_of_classes), num_parallel_calls= tf.data.experimental.AUTOTUNE)
        # Dataset concatenation, shuffling, batching, and prefetching.
        dataset = data_augmentation_dataset.concatenate(non_preprocessed_split)
        if shuffle_flag == True:
            dataset = dataset.shuffle(size)
        dataset = dataset.padded_batch(int(batch_size)).prefetch(tf.data.experimental.AUTOTUNE)
    # Priming Preprocessing
    elif preprocessing_type.lower() == "priming":
        # Preprocessing functions applications.
        non_preprocessed_split = non_preprocessed_split.map(lambda x: _normalize(x, 448, 448, num_of_classes), num_parallel_calls = tf.data.experimental.AUTOTUNE)
        data_augmentation_dataset = data_augmentation_dataset.map(lambda x: _priming_data_augmentation(x, num_of_classes), num_parallel_calls= tf.data.experimental.AUTOTUNE)
        # Dataset concatenation, shuffling, batching, and prefetching.
        dataset = data_augmentation_dataset.concatenate(non_preprocessed_split)
        if shuffle_flag == True:
            dataset = dataset.shuffle(size)
        dataset = dataset.padded_batch(int(batch_size)).prefetch(tf.data.experimental.AUTOTUNE)
    
    return dataset
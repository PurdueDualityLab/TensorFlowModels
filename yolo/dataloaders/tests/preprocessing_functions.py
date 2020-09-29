import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_addons as tfa
from tensorflow_addons.image import utils as img_utils
import matplotlib.pyplot as plt
import numpy as np
import sys
import time
import tensorflow.keras.backend as K


<<<<<<< HEAD:yolo/dataloaders/tests/preprocessing_functions.py
=======
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
    randscale = np.random.randint(low = 10, high = 19)
    jitter_x = np.random.uniform(low = -0.1, high = 0.1)
    jitter_y = np.random.uniform(low = -0.1, high = 0.1)
    jitter_cx = 0.0
    jitter_cy = 0.0
    jitter_bw = np.random.uniform(low = -.05, high = .05) + 1.0
    jitter_bh = np.random.uniform(low = -.05, high = .05) + 1.0
    return jitter_x, jitter_y, jitter_cx, jitter_cy, jitter_bw, jitter_bh, randscale 

>>>>>>> master:yolo/dataloaders/preprocessing_functions.py
@tf.function
def build_grided_gt(y_true, mask, size):
    """
    convert ground truth for use in loss functions
    Args: 
        y_true: tf.Tensor[] ground truth [box coords[0:4], classes_onehot[0:-1], best_fit_anchor_box]
        mask: list of the anchor boxes choresponding to the output, ex. [1, 2, 3] tells this layer to predict only the first 3 anchors in the total. 
        size: the dimensions of this output, for regular, it progresses from 13, to 26, to 52
    
    Return:
        tf.Tensor[] of shape [batch, size, size, #of_anchors, 4, 1, num_classes]
    """
    batches = tf.shape(y_true)[0]
    num_boxes = tf.shape(y_true)[1]
    len_masks = tf.shape(mask)[0]

    # finshape = tf.convert_to_tensor([batches, size, size, len_masks * tf.shape(y_true)[-1]])
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
            if K.all(tf.math.equal(y_true[batch, box_id, 2:4], 0)):
                continue
            if K.any(tf.math.less(y_true[batch, box_id, 0:2], 0.0)) or K.any(tf.math.greater_equal(y_true[batch, box_id, 0:2], 1.0)): 
<<<<<<< HEAD:yolo/dataloaders/tests/preprocessing_functions.py
=======
                #tf.print("outer vals: ",y_true[batch, box_id, 0:2])
>>>>>>> master:yolo/dataloaders/preprocessing_functions.py
                continue
            index = tf.math.equal(anchors[batch, box_id], mask)
            if K.any(index):
                p = tf.cast(K.argmax(tf.cast(index, dtype = tf.int32)), dtype = tf.int32)
                
                # start code for tie breaker, temp check performance 
                # find the index of the box
                uid = 1
                used = depth_track[batch, y[batch, box_id], x[batch, box_id], p]
                count = 0
                # check if the next anchor is used used == 1, if so find another box 
                while tf.math.equal(used, 1) and tf.math.less(count, 3):
                    uid = 2
                    count += 1
                    p = (p + 1)%3
                    used = depth_track[batch, x[batch, box_id], y[batch, box_id], p]
                if tf.math.equal(used, 1):
                    tf.print("skipping")
                    continue
                # set the current index to used  = 2, to indicate that it is occupied by something that should not be there, so if another box fits that anchor
                # it will be prioritized over the current box.
                depth_track = tf.tensor_scatter_nd_update(depth_track, [(batch, y[batch, box_id], x[batch, box_id], p)], [uid])
                #end code for tie breaker

                # write the box to the update list 
                # the boxes output from yolo are for some reason have the x and y indexes swapped for some reason, I am not sure why 
                """peculiar"""
                update_index = update_index.write(i, [batch, y[batch, box_id], x[batch, box_id], p])
                value = K.concatenate([y_true[batch, box_id, 0:4], tf.convert_to_tensor([1.]), y_true[batch, box_id, 4:-1]])
                update = update.write(i, value)
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

    # if the seize of the update list is not 0, do an update, other wise, no boxes and pass an empty grid
    if tf.math.greater(update_index.size(), 0):
        update_index = update_index.stack()
        update = update.stack()
        full = tf.tensor_scatter_nd_add(full, update_index, update)
    return full

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


def _detection_data_augmentation(image, label, masks, fixed_size = True, jitter_im = False):
    """
    for each mask in masks, compute a output ground truth grid
    
    Args: 
        image: tf.tensor image to manipulate 
        label: the ground truth of the boxes [batch, 4, 1, num_classes]
        masks: dictionary for the index of the anchor to use at each scale, the number of keys should be the 
               same as the number of prediction your yolo configuration will make. 
             
               ex: yolo regular: -> change to this format
                {256: [0,1,2], 512: [3,4,5], 1024: [6,7,8]}
    
    return: 
        tf.Tensor: for the image with jitter computed 
        dict{tf.tensor}: output grids for proper yolo predictions
    
    """
    # Image Jitter
    jitter_x, jitter_y, jitter_cx, jitter_cy, jitter_bw, jitter_bh, randscale = tf.py_function(py_func_rand, [], [tf.float32,  tf.float32,tf.float32,  tf.float32,tf.float32,  tf.float32, tf.int32,])
    if fixed_size:
        randscale = 13
    
    if jitter_im == True:
        image_jitter = tf.concat([jitter_x, jitter_y], axis = 0)
        image_jitter.set_shape([2])
        image = tfa.image.translate(image, image_jitter * tf.cast(tf.shape(image)[1], tf.float32))
        # Bounding Box Jitter
<<<<<<< HEAD:yolo/dataloaders/tests/preprocessing_functions.py
=======
        #tf.print(tf.shape(label))
>>>>>>> master:yolo/dataloaders/preprocessing_functions.py
        x = tf.math.add(label[..., 0], jitter_x + jitter_cx)
        x = tf.expand_dims(x, axis = -1)
        y = tf.math.add(label[..., 1], jitter_y + jitter_cy)
        y = tf.expand_dims(y, axis = -1)
        w = label[..., 2] * jitter_bw
        w = tf.expand_dims(w, axis = -1)
        h = label[..., 3] * jitter_bh
        h = tf.expand_dims(h, axis = -1)

        rest = label[..., 4:]
        label = tf.concat([x,y,w,h,rest], axis = -1)
<<<<<<< HEAD:yolo/dataloaders/tests/preprocessing_functions.py
    
=======
>>>>>>> master:yolo/dataloaders/preprocessing_functions.py
    # Other Data Augmentation
    image = tf.image.resize(image, size = (randscale * 32, randscale * 32)) # Random Resize
    image = tf.image.random_brightness(image=image, max_delta=.1) # Brightness
    image = tf.image.random_saturation(image=image, lower = 0.75, upper=1.25) # Saturation
    image = tf.image.random_hue(image=image, max_delta=.1) # Hue
    
    for key in masks.keys():
        masks[key] = build_grided_gt(label, tf.convert_to_tensor(masks[key], dtype= tf.float32), randscale)
        randscale *= 2
        
    return image, masks

# def _detection_data_augmentation(image, label, masks, fixed_size = True, jitter_im = False):
#     """
#     for each mask in masks, compute a output ground truth grid
    
#     Args: 
#         image: tf.tensor image to manipulate 
#         label: the ground truth of the boxes [batch, 4, 1, num_classes]
#         masks: dictionary for the index of the anchor to use at each scale, the number of keys should be the 
#                same as the number of prediction your yolo configuration will make. 
             
#                ex: yolo regular: -> change to this format
#                 {256: [0,1,2], 512: [3,4,5], 1024: [6,7,8]}
    
#     return: 
#         tf.Tensor: for the image with jitter computed 
#         dict{tf.tensor}: output grids for proper yolo predictions
    
#     """

#     #masks = tf.convert_to_tensor(masks, dtype= tf.float32)
#     # Image Jitter
#     jitter, randscale = tf.py_function(py_func_rand, [], [tf.float32, tf.int32])
#     if fixed_size:
#         randscale = 13
    
#     if jitter_im == True:
#         image_jitter = tf.concat([jitter, jitter], axis = 0)
#         image_jitter.set_shape([2])
#         image = tfa.image.translate(image, image_jitter)
#         # Bounding Box Jitter
#         #tf.print(tf.shape(label))
#         x = tf.math.add(label[..., 0], jitter)
#         x = tf.expand_dims(x, axis = -1)
#         y = tf.math.add(label[..., 1], jitter)
#         y = tf.expand_dims(y, axis = -1)
#         rest = label[..., 2:]
#         label = tf.concat([x,y,rest], axis = -1)
#     # Other Data Augmentation
#     image = tf.image.resize(image, size = (randscale * 32, randscale * 32)) # Random Resize
#     image = tf.image.random_brightness(image=image, max_delta=.1) # Brightness
#     image = tf.image.random_saturation(image=image, lower = 0.75, upper=1.25) # Saturation
#     image = tf.image.random_hue(image=image, max_delta=.1) # Hue

#     for key in masks.keys():
#         masks[key] = build_grided_gt(label, tf.convert_to_tensor(masks[key], dtype= tf.float32), randscale)
#         randscale *= 2

#     return image, masks

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
    """Normalizes the image by resizing it to the desired output shape
    Args:
        datapoint (dict): A Dictionaty that holds the image as well as other relevant
            information.
        
        h: the default height to scale images of variable shape to in order to batch the images  
        w: the default width to scale images of variable shape to in order to batch the images
        number of classes: the number of classes that can be predicted for each object 
    Returns:
        tf.Tensor (image): scaled and resized image for input into model, prior to batching
        tf.Tensor (label): the label of the bounding boxes with the best anchor asscoiated with it known for each ground truth box 
    """
    image = tf.cast(data["image"], dtype=tf.float32)
    image = tf.image.resize(image, size = (608, 608))
    boxes = data["objects"]["bbox"]
    boxes = get_yolo_box(boxes)
    classes = tf.one_hot(data["objects"]["label"], depth = 80)
    label = tf.concat([boxes, classes], axis = -1)
    label = build_gt(label, anchors, width)
    image = image / 255 # Normalize
    return image, label

<<<<<<< HEAD:yolo/dataloaders/tests/preprocessing_functions.py
def preprocessing(dataset, num_of_classes, batch_size, size, data_augmentation_split = 100, preprocessing_type = "detection", shuffle_flag = False, anchors = None, masks = None, fixed = False, jitter = False):
=======
def preprocessing(dataset, data_augmentation_split, preprocessing_type, size, batch_size, num_of_classes, shuffle_flag = False, anchors = None, masks = None, fixed = False, jitter = False):
>>>>>>> master:yolo/dataloaders/preprocessing_functions.py
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
        dataset = dataset.map(lambda x: _detection_normalize(x, anchors, 416, 416), num_parallel_calls = tf.data.experimental.AUTOTUNE).padded_batch(int(batch_size))
        dataset = dataset.map(lambda x, y: _detection_data_augmentation(x, y, masks = masks, fixed_size=fixed, jitter_im= jitter), num_parallel_calls = tf.data.experimental.AUTOTUNE)#.prefetch(10)
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
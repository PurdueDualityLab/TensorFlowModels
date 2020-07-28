import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_addons as tfa
from tensorflow_addons.image import utils as img_utils
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import numpy as np
import sys
import time

# Global Variable to introduce randomness among each element of a batch
RANDOM_SEED = tf.random.Generator.from_seed(int(np.random.uniform(low=300, high=9000)))

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
        image_h = tf.cast(img.shape[1], tf.dtypes.float32)
        image_w = tf.cast(img.shape[2], tf.dtypes.float32)
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
    # Global Variable defined at the beginning of the file.
    global RANDOM_SEED
    return RANDOM_SEED.uniform(minval= low, maxval= high, shape = (), dtype=tf.float32)

def _preprocessing_selection(choice):
    """Returns the requested data augmentation function required for the training
        specfied.

    Args:
        choice(str): The type of training the user would like to use.

    Returns:
        function: A function for data augmentation for the specfic training specified.
    """

    def classification(datapoint):
        """Augments image by performing Random Zoom, Resize with Pad, Random Rotate,
        Random Brightness Distortion, Random Saturation Distortion, Random Hue Distortion
        and finally normalizing the image.

        Args:
            datapoint (dict): A Dictionaty that holds the image as well as other relevant
                information.

        Returns:
            Either Image and Label or Image and Object.
        """

        # Generates Random Variables that will be used within the Data Augmentation Function.
        image = datapoint['image']
        image = tf.cast(image, tf.float32)
        w = tf.cast(image.shape[1], tf.float32)
        h = tf.cast(image.shape[2], tf.int32)
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

        # Return
        if "object" in datapoint:
            return image, datapoint['object']
        else:
            return image, datapoint['label']

    def priming(datapoint):
        """Augments image by performing Random Zoom, Resize with Pad, and
            finally normalizing the image.

        Args:
            datapoint (dict): A Dictionaty that holds the image as well as other relevant
                information.

        Returns:
            Either Image and Label or Image and Object.
        """

        # Generates Random Variables that will be used within the Data Augmentation Function.
        image = datapoint['image']
        image = tf.cast(image, tf.float32)
        w = tf.cast(image.shape[1], tf.float32)
        h = tf.cast(image.shape[2], tf.int32)
        low = tf.cast(448, tf.dtypes.float32)[None]
        high = tf.cast(512, tf.dtypes.float32)[None]
        scale = tf.py_function(_rand_number, [low, high], [tf.float32])
        scale = tf.cast(scale, dtype= tf.int32)[0][0]

        # Data Augmentation Functions.
        image = tf.image.resize_with_crop_or_pad(image, target_height = scale, target_width = scale) # Zoom
        image = tf.image.resize_with_pad(image, target_width=448, target_height=448) # Final Output Shape
        image = image / 255 #Normalize

        # Return
        if "object" in datapoint:
            return image, datapoint['object']
        else:
            return image, datapoint['label']

    def detection(datapoint):
        """Augments image by performing Random Resize with Pad, Random Brightness Distortion,
        Random Saturation Distortion, Random Hue Distortion and finally normalizing the image.

        Args:
            datapoint (dict): A Dictionaty that holds the image as well as other relevant
                information.

        Returns:
            Either Image and Label or Image and Object.
        """

        # Generates Random Variables that will be used within the Data Augmentation Function.
        image = datapoint['image']
        image = tf.cast(image, tf.float32)
        low = tf.cast(128, tf.dtypes.float32)[None]
        high = tf.cast(448, tf.dtypes.float32)[None]
        resize_num = tf.py_function(_rand_number, [10.0, 19.0], [tf.float32])
        resize_num = tf.cast(resize_num, dtype= tf.int32)[0]*32

        # Data Augmentation Functions.
        image = tf.image.resize_with_pad(image, target_width=resize_num, target_h=resize_num) # Random Resize
        image = tf.image.random_brightness(image=image, max_delta=.75) # Brightness 
        image = tf.image.random_saturation(image=image, lower = 0.75, upper=1.25) # Saturation
        image = tf.image.random_hue(image=image, max_delta=.1) # Hue
        image = image / 255 # Normalize

        # Return
        if "object" in datapoint:
            return image, datapoint['object']
        else:
            return image, datapoint['label']

    if choice.lower() == "detection":
        return detection
    elif choice.lower() == "classification":
        return classification
    elif choice.lower() == "priming":
        return priming

def _normalize_selection(h, w):
    """Returns the requested normalization function required for the width and height
        specified

    Args:
        h (int): Height of desired output image.
        w (int): Width of desired output image.

    Returns:
        function: A function for normalize for the specfic training specified.
    """
    def normalize(datapoint):
        """Normalizes the image by resizing it to the desired output shape

        Args:
            datapoint (dict): A Dictionaty that holds the image as well as other relevant
                information.

        Returns:
            normalize (dict): A Normalized Image alongside the mapped information.
        """
        image = datapoint['image']
        image = tf.cast(image, tf.float32)

        # Normalization Functions.
        image = tf.image.resize_with_pad(image, target_width=h, target_height=w) # Final Output Shape
        image = image / 255 # Normalize

        # Return
        if "object" in datapoint:
            return image, datapoint['object']
        else:
            return image, datapoint['label']

    return normalize

def _detection_normalize(datapoint):
    """Normalizes the image by doing random resizing required for detection.

    Args:
        datapoint (dict): A Dictionaty that holds the image as well as other relevant
            information.

    Returns:
        normalize (dict): A Normalized Image alongside the mapped information.
    """
    # Generates Random Variables that will be used within the Normalization Function.
    image = datapoint['image']
    image = tf.cast(image, tf.float32)
    low = tf.cast(128, tf.dtypes.float32)[None]
    high = tf.cast(448, tf.dtypes.float32)[None]
    resize_num = tf.py_function(_rand_number, [10.0, 19.0], [tf.float32])
    resize_num = tf.cast(resize_num, dtype= tf.int32)[0]*32

    # Normalization Functions.
    image = tf.image.resize_with_pad(image, target_width=resize_num, target_h=resize_num) # Final Output Shape
    image = image / 255 # Normalize

    # Return
    if "object" in datapoint:
        return image, datapoint['object']
    else:
        return image, datapoint['label']

# MAIN FUNCTION TO USE FOR PREPROCESSING
# SAMPLE USE CASES CAN BE FOUND IN THE benchmark_preprocessing.py
def preprocessing(dataset, data_augmentation_split, preprocessing_type, size, num_of_batches):
    """Preprocesses (normalization and data augmentation) and batches the dataset.

    Args:
        dataset (tfds.data.Dataset): The Dataset you would like to preprocess.
        data_augmentation_split (int): The percentage of the dataset that is data
            augmented.
        preprocessing_type (str): The type of preprocessing should be conducted
            and is dependent on the type of training.
        size (int): The size of the dataset being passed into preprocessing.
        num_of_batches (int): The number of batches you would like the return
            dataset to be split into.

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
            - Dataset is not a tensorflow dataset.
            - Data augmentation split must be an integer.
            - Preprocessing type must be an string.
            - Size must be an integer.
            - Number of batches must be an integer.
    """
    if isinstance(dataset, tf.python.data.ops.dataset_ops.DatasetV1Adapter) == False:
        raise TypeError("Dataset is not a tensorflow dataset.")
    if type(data_augmentation_split) is not int:
        raise TypeError("Data augmentation split must be an integer.")
    if type(preprocessing_type) is not str:
        raise TypeError("Preprocessing type must be an string.")
    if type(size) is not int:
        raise TypeError("Size must be an integer.")
    if type(num_of_batches) is not int:
        raise TypeError("Number of batches must be an integer.")

    if preprocessing_type.lower() != "detection" and preprocessing_type.lower() != "classification" and preprocessing_type.lower() != "priming":
        raise SyntaxError("Preprocessing type not found.")
    if num_of_batches != 1 and preprocessing_type.lower() == "detection":
        raise SyntaxError("For detection preprocessing, number of batches must be 1.")
    if num_of_batches < 1:
        raise SyntaxError("Number of batches cannot be less than 1.")
    if data_augmentation_split > 100:
        raise SyntaxError("Data augmentation split cannot be greater than 100.")

    # Spliting the dataset based off of user defined split
    data_augmentation_split = int((data_augmentation_split/100)*size)
    non_preprocessed_split = size - data_augmentation_split
    data_augmentation_dataset = dataset.take(data_augmentation_split)
    remaining = dataset.skip(data_augmentation_split)
    non_preprocessed_split = remaining.take(non_preprocessed_split)

    # Data Augmentation
    preprocessing_function = _preprocessing_selection(preprocessing_type)
    data_augmentation_dataset = data_augmentation_dataset.map(preprocessing_function, num_parallel_calls= tf.data.experimental.AUTOTUNE)

    # Normalization
    if preprocessing_type.lower() == "detection":
        non_preprocessed_split = non_preprocessed_split.map(_detection_normalize, num_parallel_calls = tf.data.experimental.AUTOTUNE)

    elif preprocessing_type.lower() == "classification":
        normalize = _normalize_selection(224, 224)
        non_preprocessed_split = non_preprocessed_split.map(normalize, num_parallel_calls = tf.data.experimental.AUTOTUNE)

    elif preprocessing_type.lower() == "priming":
        normalize = _normalize_selection(448, 448)
        non_preprocessed_split = non_preprocessed_split.map(normalize, num_parallel_calls = tf.data.experimental.AUTOTUNE)

    # Preparing the return dataset through concatentaion and shuffling
    dataset= data_augmentation_dataset.concatenate(non_preprocessed_split)
    dataset = dataset.shuffle(size)
    dataset = dataset.batch(int(size/num_of_batches)).prefetch(tf.data.experimental.AUTOTUNE)
    return dataset
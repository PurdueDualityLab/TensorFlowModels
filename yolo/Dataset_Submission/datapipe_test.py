import YOLO_Classification_Input as cls
import YOLO_Priming_Input as prime
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import time

def preprocessing(dataset, num_of_classes, batch_size, size, data_augmentation_split = 100, preprocessing_type = "detection", shuffle_flag = False, anchors = None, masks = None, fixed = False, jitter = False):
    data_augmentation_split = int((data_augmentation_split/100)*size)
    non_preprocessed_split = size - data_augmentation_split
    data_augmentation_dataset = dataset.take(data_augmentation_split)
    remaining = dataset.skip(data_augmentation_split)
    non_preprocessed_split = remaining.take(non_preprocessed_split)
    # Data Preprocessing functions based off of selected preprocessing type.
    if preprocessing_type.lower() == "classification":
        # Preprocessing functions applications.
        Dataset_Parser = prime.Priming_Parser([224,224], 50) # ADD CONFIG FILE SUPPORT
        _normalize, _data_augmentation = Dataset_Parser.parse_fn(is_training = False), Dataset_Parser.parse_fn(is_training = True) 
        non_preprocessed_split = non_preprocessed_split.map(_normalize, num_parallel_calls = tf.data.experimental.AUTOTUNE)
        data_augmentation_dataset = data_augmentation_dataset.map(_data_augmentation, num_parallel_calls= tf.data.experimental.AUTOTUNE)
        # Dataset concatenation, shuffling, batching, and prefetching.
        dataset = data_augmentation_dataset.concatenate(non_preprocessed_split)
        if shuffle_flag == True:
            dataset = dataset.shuffle(size)
        dataset = dataset.padded_batch(int(batch_size)).prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

Train, Info = tfds.load('rock_paper_scissors', split='train', with_info=True)
Size = int(Info.splits['train'].num_examples)
# This is the function that you have to call inorder to preprocess and batch the dataset.
# For more info refer to preprocessing_functions.py.
Train = preprocessing(Train, 3, 50, Size, data_augmentation_split = 100, preprocessing_type = "classification", shuffle_flag = False)
# benchmarking
start = time.time()
count = 0
train_ds = Train.take(1)
for i,j in train_ds:
    for image in i:
        plt.imshow(image)
        plt.show()
        arr = image
        count += 1
end = time.time()
print(f'total time: {end-start}')
print(f'num_img: {count}')
print(f'size: {Size}')
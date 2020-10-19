import tensorflow as tf
import tensorflow_datasets as tfds
from yolo.dataloaders.YOLO_Priming_Input import Priming_Parser as Classification_Parser
from yolo.dataloaders.ImageNetDecoder import ImageNetDecoder

def preprocessing(dataset, num_of_classes, batch_size, size, data_augmentation_split = 100, preprocessing_type = "detection", shuffle_flag = False, anchors = None, masks = None, fixed = False, jitter = False):
    data_augmentation_split = int((data_augmentation_split/100)*size)
    non_preprocessed_split = size - data_augmentation_split
    data_augmentation_dataset = dataset.take(data_augmentation_split)
    remaining = dataset.skip(data_augmentation_split)
    non_preprocessed_split = remaining.take(non_preprocessed_split)
    Dataset_Parser = Classification_Parser([224,224], 50) # ADD CONFIG FILE SUPPORT
    _normalize, _data_augmentation = Dataset_Parser.parse_fn(is_training = False), Dataset_Parser.parse_fn(is_training = True) 
    non_preprocessed_split = non_preprocessed_split.map(_normalize, num_parallel_calls = tf.data.experimental.AUTOTUNE)
    data_augmentation_dataset = data_augmentation_dataset.map(_data_augmentation, num_parallel_calls= tf.data.experimental.AUTOTUNE)
    dataset = data_augmentation_dataset.concatenate(non_preprocessed_split)
    if shuffle_flag == True:
        dataset = dataset.shuffle(size)
    dataset = dataset.padded_batch(int(batch_size)).prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

import time
import matplotlib.pyplot as plt
Train, Info = tfds.load('imagenet_a', decoders={
        'image': tfds.decode.SkipDecoding(),
    }, split='test', with_info=True)
decoder = ImageNetDecoder()
ds = Train.map(decoder.decode)
Size = int(Info.splits['test'].num_examples)
# This is the function that you have to call inorder to preprocess and batch the dataset.
# For more info refer to preprocessing_functions.py.
ds = preprocessing(ds, 3, 50, Size, data_augmentation_split = 30, preprocessing_type = "classification", shuffle_flag = False)
# benchmarking
start = time.time()
count = 0
print(ds)

train_ds = ds.take(1)

for i,j in train_ds:
    print(j)
    for ind, image in enumerate(i):
        plt.imshow(image)
        plt.show()
        arr = image
        count += 1
end = time.time()
print(f'total time: {end-start}')
print(f'num_img: {count}')
print(f'size: {Size}')
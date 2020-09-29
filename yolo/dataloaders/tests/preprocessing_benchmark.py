# WILL BE DELETED WHEN RELEASING PUBLICLY
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
import sys
import time
#** Below I am importing the preprocessing functions.
import preprocessing_functions

#** Below this is how you import a custom dataset, here is an example of importing imagenet. **
builder = tfds.ImageFolder('/local/b/cam2/data/ILSVRC2012_Classification/')
Train = builder.as_dataset(split='train', shuffle_files=False)
Size = int (builder.info.splits['train'].num_examples)

#** Below this is how you import a dataset if it exists in the tfds catalog. **
# Train, Info = tfds.load('rock_paper_scissors', split='train', with_info=True, shuffle_files=False, data_dir = "/local/b/cam2/data/ILSVRC2012_Classification/", download=False)
# Size = int(Info.splits['train'].num_examples)

# This is the function that you have to call inorder to preprocess and batch the dataset.
# For more info refer to preprocessing_functions.py.
Train = preprocessing_functions.preprocessing(Train, 50,"classification", Size, 1000, 1001, 224) # https://www.calculatorsoup.com/calculators/math/factors.php -> Check to find a good batch size

# benchmarking
start = time.time()
count = 0
for x in range(Size):
    train_ds = Train.take(1)
    for i,j in train_ds:
        for image in i:
            arr = image
            count += 1
        break
    break
end = time.time()

print(f'total time: {end-start}')
print(f'num_img: {count}')
print(f'size: {Size}')

# STATS
# Preprocessing 1 batch of imagenet2012 takes 3.3 seconds.

import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_addons as tfa
from tensorflow_addons.image import utils as img_utils
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import numpy as np
import sys
import time
import preprocessing_functions


Train, Info = tfds.load('rock_paper_scissors', split='train', with_info=True, shuffle_files=False)
Size = int(Info.splits['train'].num_examples)
Train = preprocessing_functions.preprocessing(Train, 50,"classification", Size, 30) # https://www.calculatorsoup.com/calculators/math/factors.php -> Check to find a good batch size

#benchmark
start = time.time()
count = 0
for x in range(30):
    train_ds = Train.take(1)
    print(count)
    count += 1
stop = time.time()
print(stop - start)
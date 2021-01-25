import tensorflow_datasets as tfds

path = "/media/vbanna/DATA_SHARE/tfds"
dataset = "coco/2017_panoptic"

train = tfds.load(name = dataset, download = True, data_dir = path, shuffle_files = True, split = "train")
print(train)
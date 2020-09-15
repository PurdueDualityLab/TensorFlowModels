#Depricated
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_virtual_device_configuration(gpu, [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*5)])
        logical = tf.config.experimental.list_logical_devices('GPU')
        print(f"{len(gpus)} Physical GPUs {len(logical)} Logical GPUs")
    except RuntimeError as e:
        print(e)
import tensorflow.keras as ks
import tensorflow_datasets as tfds
from yolo.modeling.yolo_v3 import DarkNet53
from yolo.training.callbacks import Darknet_Classification_LR
from yolo.training.callbacks import poly_schedule
from yolo.dataloaders import preprocessing_functions
from yolo.training.callbacks import config

mirrored_strategy = tf.distribute.MirroredStrategy()

EPOCHS = 160
BATCH_SIZE = 128

#---------------
builder = tfds.ImageFolder('/home/data/ilsvrc/ILSVRC/ILSVRC2012_Classification')
dataset = builder.as_dataset(split='train',shuffle_files=True)
size = int(builder.info.splits["train"].num_examples)
print(size)
Validation_Split = int((10/100)*size)
Validation = dataset.take(Validation_Split)
remaining = dataset.skip(Validation_Split)
Train = remaining.take(size - Validation_Split)

train_size = size - Validation_Split
print(train_size)
max_batches = int((EPOCHS*train_size)/BATCH_SIZE)
batches_per_epoch = int(max_batches / EPOCHS)
config["max_batches"] = max_batches
Train = preprocessing_functions.preprocessing(Train,50,"classification", train_size, batches_per_epoch, 1000, 224)
Validation = preprocessing_functions.preprocessing(Validation,0,"classification",Validation_Split,1001,1000,224)


# Callbacks -----------------------
lr_callback = Darknet_Classification_LR(poly_schedule)
# Change file path
checkpoint_filepath = '/home/carrot/Garden/'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_acc',
    mode='max',
    save_best_only=True)

early_stopping = ks.callbacks.EarlyStopping(monitor="val_loss",patience=3,verbose=1)
NaN_loss = ks.callbacks.TerminateOnNaN()
# -------------------------------

loss = ks.losses.CategoricalCrossentropy()
metrics = 'accuracy'
with mirrored_strategy.scope():
    optimizer = ks.optimizers.SGD(lr=0.1,momentum=0.9,decay=0.005)
    model = DarkNet53(classes = 1000)
    model.build(input_shape = [None, None, None, 3])
    model.compile(optimizer = optimizer, loss = loss, metrics = [metrics])

model.summary()
model.fit(Train,validation_data=Validation, epochs = EPOCHS,callbacks=[lr_callback,model_checkpoint_callback,early_stopping])
#### ARGUMENT PARSER ####
from absl import app, flags
from absl.flags import argparse_flags
import sys

parser = argparse_flags.ArgumentParser()

parser.add_argument(
    'model',
    metavar='model',
    default='regular',
    choices=('regular', 'spp', 'tiny'),
    type=str,
    help='Name of the model. Defaults to regular. The options are ("regular", "spp", "tiny")',
    nargs='?'
)

parser.add_argument(
    'vidpath',
    default=None,
    type=str,
    help='Path of the video stream to process. Defaults to the webcam.',
    nargs='?'
)

parser.add_argument(
    '--webcam',
    default=0,
    type=int,
    help='ID number of the webcam to process as a video stream. This is only used if vidpath is not specified. Defaults to 0.',
    nargs='?'
)

if __name__ == '__main__':
    if '-h' in sys.argv or '--help' in sys.argv:
        parser.parse_args(sys.argv[1:])
        exit()

#### MAIN CODE ####

import tensorflow as tf 
from absl import app
from typing import * 

from yolo.utils.testing_utils

def configure_gpus(gpus:Union[List, str] = None, memory_limit:Union[int, str] = None):
    """
    
    """
    visible_gpus = tf.config.experimental.list_physical_devices('GPU')
    if len(visible_gpus) == 0:
        return visible_gpus
    if gpus == "all" or gpus == None:
        gpus = visible_gpus
    elif type(gpus) == str:
        gpus = [gpus]

    try:
        if memory_limit == "full" or memory_limit == None:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        else:
            # limit maximum memory usage for the list of GPU's passed in
            # memory limit on all devices must be the same error will be thrown
            for gpu in gpus:
                tf.config.experimental.set_virtual_device_configuration(gpu, [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except Runtime as e:
        print(e)
    return logical_gpus, gpus

def _train_eager(model_version:str = "v3", 
                 model_type:str = "regular", 
                 classes:int = 80, 
                 boxes:int = None,
                 masks:Dict = None, 
                 anchors:List = None,
                 save_weight_directory:str = "weights", 
                 gpus:Union[List, str] = None, 
                 memory_limit:Union[int, str] = None, 
                 dataset:Union[str, tf.data.Dataset] = "coco", 
                 epochs:int = 270, 
                 batch_size:int = 32, 
                 metrics_full:bool = True):
    logical_gpus, gpus = configure_gpus(gpus=gpus, memory_limit=memory_limit)
    strategy = tf.distribute.MirroredStrategy(devices=logical_gpus)

    # get list of anchors and masks
    with strategy.scope():
        # build the given model
        # generate the loss functions
        # load datasets 
        # generate the metrics 
        # generate the callbacks for learning rate
        # init early stop callback 
        # init check points callback
        # init tensorboard callback
    return

def main(args, argv = None):
    prep_gpu()
    _train_eager()
    return

if __name__ == "__main__":
    app.run(main)
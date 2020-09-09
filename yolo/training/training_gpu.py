import tensorflow as tf 
from absl import app
from typing import * 

from yolo.utils.testing_utils import prep_gpu, build_model_partial

def configure_gpus(gpus:Union[List, str], memory_limit:Union[int, Dict]):
    if gpus == "all":
        gpus = tf.config.experimental.list_physical_devices('GPU')
    
    try:
        for gpu in gpus:
            return 
    except Runtime as e:
        print(e)
    return 


def _train_eager():
    stratagy = tf.distribute.MirroredStrategy()
    build_model_partial(split = validation, load_head= False)

    return

def main(args, argv = None):
    prep_gpu()
    _train_eager()
    return

if __name__ == "__main__":
    app.run(main)
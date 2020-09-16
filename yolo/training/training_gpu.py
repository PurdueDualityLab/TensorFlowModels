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

def _get_anchors():
    return {"v3": {"regular": [(10,13),  (16,30),  (33,23), (30,61),  (62,45),  (59,119), (116,90),  (156,198),  (373,326)], 
                   "spp": [(10,13),  (16,30),  (33,23), (30,61),  (62,45),  (59,119), (116,90),  (156,198),  (373,326)], 
                   "tiny": [(10,14),  (23,27),  (37,58), (81,82),  (135,169),  (344,319)]}, 
            "v4": {"regular": [(12, 16), (19, 36), (40, 28), (36, 75), (76, 55), (72, 146), (142, 110), (192, 243), (459, 401)]}}

def _get_masks():
    return {"v3": {"regular": {"1024": [6,7,8], "512":[3,4,5], "256":[0,1,2]}, 
                   "spp": {"1024": [6,7,8], "512":[3,4,5], "256":[0,1,2]}, 
                   "tiny": {"1024": [3,4,5], "256": [0,1,2]}}, 
            "v4": {"regular": {"1024": [6,7,8], "512":[3,4,5], "256":[0,1,2]}}}

def _build_model(model_version, model_type, classes, anchors, masks, scales = None, input_shape = (None, None, None, 3)):
    if model_version == "v3":
        from yolo.modeling.yolo_v3 import Yolov3
        model = Yolov3(model=model_type, classes=classes, boxes=anchors, masks=masks, input_shape = input_shape)
    elif model_version == "v4":
        from yolo.modeling.yolo_v4 import Yolov4
        model = Yolov4(model=model_type, classes=classes, boxes=anchors, masks=masks, input_shape = input_shape, scales=scales)
    else:
        raise ImportError("unsupported model")
    return model

def _train_yolo( model_version:str = "v3", 
                 model_type:str = "regular", 
                 classes:int = 80, 
                 masks:Dict = None, 
                 anchors:List = None,
                 scales:Dict = None, 
                 save_weight_directory:str = "weights", 
                 gpus:Union[List, str] = None, 
                 memory_limit:Union[int, str] = None, 
                 dataset:Union[str, tf.data.Dataset] = "coco", 
                 split_train: "train",
                 split_val: "validation",
                 epochs:int = 270, 
                 batch_size:int = 32, 
                 jitter:bool = True, 
                 fixed_size:bool = False, 
                 metrics_full:bool = True):
    logical_gpus, gpus = configure_gpus(gpus=gpus, memory_limit=memory_limit)
    strategy = tf.distribute.MirroredStrategy(devices=logical_gpus)


    # get list of anchors 
    if anchors == None:
        anchors = _get_anchors()[model_version][model_type]
    
    #get masks
    if masks == None:
        masks = _get_masks()[model_version][model_type]

    with strategy.scope():
        # build the given model
        model = _build_model(model_version, model_type, classes = classes, masks = masks, anchors = anchors, scales=scales)
        
        # generate the loss functions
        model = model.generate_loss()

        # load datasets 
        train, test = model.preprocess_train_test(dataset = dataset, batch_size = batch_size, train = split_train, val = split_val, jitter = jitter, fixed = fixed_size)

        # generate the metrics 

        # generate the callbacks for learning rate

        # init early stop callback 

        # init check points callback
        # init tensorboard callback
    return

def main(args, argv = None):
    _train_yolo()
    return

if __name__ == "__main__":
    app.run(main)
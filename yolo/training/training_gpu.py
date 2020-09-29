#### ARGUMENT PARSER ####
from absl import app, flags
from absl.flags import argparse_flags
import sys

# parser = argparse_flags.ArgumentParser()

# parser.add_argument(
#     'model',
#     metavar='model',
#     default='regular',
#     choices=('regular', 'spp', 'tiny'),
#     type=str,
#     help='Name of the model. Defaults to regular. The options are ("regular", "spp", "tiny")',
#     nargs='?'
# )

# parser.add_argument(
#     'vidpath',
#     default=None,
#     type=str,
#     help='Path of the video stream to process. Defaults to the webcam.',
#     nargs='?'
# )

# parser.add_argument(
#     '--webcam',
#     default=0,
#     type=int,
#     help='ID number of the webcam to process as a video stream. This is only used if vidpath is not specified. Defaults to 0.',
#     nargs='?'
# )

if __name__ == '__main__':
    if '-h' in sys.argv or '--help' in sys.argv:
        parser.parse_args(sys.argv[1:])
        exit()

#### MAIN CODE ####
import os
import tensorflow as tf
from absl import app
from typing import *


def configure_gpus(gpus: Union[List, str], memory_limit: Union[int, str]):
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
                tf.config.experimental.set_virtual_device_configuration(
                    gpu, [
                        tf.config.experimental.VirtualDeviceConfiguration(
                            memory_limit=memory_limit)
                    ])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except Runtime as e:
        print(e)
    return logical_gpus, gpus


def _get_anchors():
    return {
        "v3": {
            "regular": [(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
                        (59, 119), (116, 90), (156, 198), (373, 326)],
            "spp": [(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
                    (59, 119), (116, 90), (156, 198), (373, 326)],
            "tiny": [(10, 14), (23, 27), (37, 58), (81, 82), (135, 169),
                     (344, 319)]
        },
        "v4": {
            "regular": [(12, 16), (19, 36), (40, 28), (36, 75), (76, 55),
                        (72, 146), (142, 110), (192, 243), (459, 401)]
        }
    }


def _get_masks():
    return {
        "v3": {
            "regular": {
                "1024": [6, 7, 8],
                "512": [3, 4, 5],
                "256": [0, 1, 2]
            },
            "spp": {
                "1024": [6, 7, 8],
                "512": [3, 4, 5],
                "256": [0, 1, 2]
            },
            "tiny": {
                "1024": [3, 4, 5],
                "256": [0, 1, 2]
            }
        },
        "v4": {
            "regular": {
                "1024": [6, 7, 8],
                "512": [3, 4, 5],
                "256": [0, 1, 2]
            }
        }
    }


def _build_model(model_version,
                 model_type,
                 classes,
                 anchors,
                 masks,
                 scales=None,
                 input_shape=(None, None, None, 3)):
    if model_version == "v3":
        from yolo.modeling.yolo_v3 import Yolov3
        model = Yolov3(model=model_type,
                       classes=classes,
                       boxes=anchors,
                       masks=masks,
                       input_shape=input_shape)
    elif model_version == "v4":
        from yolo.modeling.yolo_v4 import Yolov4
        model = Yolov4(model=model_type,
                       classes=classes,
                       boxes=anchors,
                       masks=masks,
                       input_shape=input_shape,
                       scales=scales)
    else:
        raise ImportError("unsupported model")
    return model


def _train_yolo(model_version="v3",
                model_type="regular",
                classes=80,
                masks=None,
                anchors=None,
                scales=None,
                save_weight_directory="weights",
                gpus=None,
                memory_limit=None,
                dataset="coco",
                split_train="train",
                split_val="validation",
                epochs=270,
                batch_size=32,
                jitter=True,
                fixed_size=False,
                metrics_full=True,
                check_point_path="/tmp/checkpoint/yolo_check_point",
                log_dir="./logs"):
    logical_gpus, gpus = configure_gpus(gpus=gpus, memory_limit=memory_limit)
    strategy = tf.distribute.MirroredStrategy(devices=logical_gpus)

    # get list of anchors
    if anchors == None:
        anchors = _get_anchors()[model_version][model_type]

    #get masks
    if masks == None:
        masks = _get_masks()[model_version][model_type]

    print(anchors)
    print(masks)

    with strategy.scope():
        # build the given model
        model = _build_model(model_version,
                             model_type,
                             classes=classes,
                             masks=masks,
                             anchors=anchors,
                             scales=scales)

        # generate the loss functions
        loss = model.generate_loss()

    # load datasets
    train, test = model.preprocess_train_test(dataset=dataset,
                                              batch_size=batch_size,
                                              train=split_train,
                                              val=split_val,
                                              jitter=jitter,
                                              fixed=fixed_size)

    with strategy.scope():
        # generate the metrics

        # generate the callbacks for learning rate

        # treminate on NAN
        term_nan = tf.keras.callbacks.TerminateOnNaN()
        # init check points callback
        check_points = tf.keras.callbacks.ModelCheckpoint(
            filepath=check_point_path,
            monitor="val_loss",
            save_best_only=False,
            save_weights_only=True,
            save_freq="epoch")
        # init tensorboard callback
        # solving cupit error
        # /sbin/ldconfig -N -v $(sed 's/:/ /g' <<< $LD_LIBRARY_PATH) | grep libcupti
        # export LD_LIBRARY_PATH="/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda-10.1/lib64:/usr/local/cuda-10.1/extras/CUPTI/lib64"
        # modprobe nvidia NVreg_RestrictProfilingToAdminUsers=0
        # sudo vim /etc/modprobe.d/nvidia-kernel-common.conf
        # write options nvidia "NVreg_RestrictProfilingToAdminUsers=0"
        tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                     histogram_freq=0,
                                                     write_graph=True,
                                                     update_freq=1000)

        callbacks = [term_nan, check_points, tensorboard]
        # model.fit(train, validation_data=test, shuffle= True, epochs=epochs, callbacks=callbacks)
    return


def main(args, argv=None):
    _train_yolo()
    return


if __name__ == "__main__":
    app.run(main)

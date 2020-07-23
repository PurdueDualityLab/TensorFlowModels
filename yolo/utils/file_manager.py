"""
Manage the downloading of external files that are used in YOLO networks.
"""

import tensorflow.keras as ks
import os

cfg = {
    'yolov3': 'https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg'
}

weights = {
    'yolov3': 'https://pjreddie.com/media/files/yolov3.weights'
}

allData = {
    'cfg': cfg,
    'weights': weights
}

def download(name, type):
    """
    Download a `type` file corresponding to network `name`.
    """
    url = allData[type][name]
    return ks.utils.get_file(url.rsplit('/', 1)[1], url, cache_dir='cache', cache_subdir=type)

def downloadAll(name=None):
    """
    Download all the files that are related to a particular network. You can
    specify the name of the network with the parameter to this function or leave
    it as None to download all files.
    """
    for type, data in allData.items():
        for name_, url in data.items():
            if name is None or name == name_:
                ks.utils.get_file(url.rsplit('/')[1], url, cache_dir='cache', cache_subdir=type)

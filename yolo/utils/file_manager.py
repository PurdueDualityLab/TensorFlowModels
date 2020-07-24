"""
Manage the downloading of external files that are used in YOLO networks.
"""

from http.client import HTTPException
import tensorflow.keras as ks
import os

urls = {
    'yolov3.cfg': ('https://rfaw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg', 'cfg', 'c4dc8d5b76d218e4878ef105d50b39e4f64fb3bb'),
    'yolov3.weights': ('https://pjreddie.com/media/files/yolov3.weights', 'weights', '520878f12e97cf820529daea502acca380f1cb8e'),
}

def download(name: str) -> str:
    """
    Download a predefined file named `name` from the original repository.

    For example, yolov3.weights is a file that defines the pretrained YOLOv3
    model. It can be downloaded from https://pjreddie.com/media/files/yolov3.weights
    so it is downloaded from there.

    Args:
        name: Name of the file that will be downloaded

    Returns:
        The path of the downloaded file as a `str`

    Raises:
        KeyError:       Name of file is not found in the `urls` variable.
        ValueError:     Name or URL stored in the `urls` variable is invalid.
        OSError:        There was a problem saving the file when it was being
                        downloaded.
        HTTPException:  The file was not able to be downloaded.
        Exception:      Any other undocumented error that ks.utils.get_file may
                        have thrown to indicate that the file was inaccessible.
    """
    url, type, hash = urls[name]
    try:
        return ks.utils.get_file(name, url, cache_dir='cache', cache_subdir=type, file_hash=hash, hash_algorithm='sha256')
    except Exception as e:
        if 'URL fetch failure on' in str(e):
            raise HTTPException(str(e)) from e
        else:
            raise

def downloadAll(name: str = None):
    """
    Download all the files that are related to a particular network. You can
    specify the name of the network with the parameter to this function or leave
    it as None to download all files.
    """
    for type, data in allData.items():
        for name_, url in data.items():
            if name is None or name == name_:
                ks.utils.get_file(url.rsplit('/')[1], url, cache_dir='cache', cache_subdir=type)

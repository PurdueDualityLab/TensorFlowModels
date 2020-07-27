"""
Manage the downloading of external files that are used in YOLO networks.
"""

from __future__ import annotations

from http.client import HTTPException
import tensorflow.keras as ks
import os

urls = {
    'yolov3.cfg': (
        'https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg',
        'cfg',
        '22489ea38575dfa36c67a90048e8759576416a79d32dc11e15d2217777b9a953'),
    'yolov3.weights': (
        'https://pjreddie.com/media/files/yolov3.weights',
        'weights',
        '523e4e69e1d015393a1b0a441cef1d9c7659e3eb2d7e15f793f060a21b32f297'),
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
        return ks.utils.get_file(
            name,
            url,
            cache_dir='cache',
            cache_subdir=type,
            file_hash=hash,
            hash_algorithm='sha256')
    except Exception as e:
        if 'URL fetch failure on' in str(e):
            raise HTTPException(str(e)) from e
        else:
            raise
